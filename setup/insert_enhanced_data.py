import sys
import os
from datetime import datetime
import time
import json
from dotenv import load_dotenv
from textwrap import dedent

from tqdm.auto import tqdm
import numpy as np
from pydantic import BaseModel
from faker import Faker
from tenacity import retry, stop_after_attempt, wait_exponential
from llama_index.llms.openai import OpenAI
from llama_index.core import PromptTemplate

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.db.database import session_scope
from src.db import models

load_dotenv()

FAKER = Faker()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL = "gpt-4o-mini"
MAX_TOKENS_SYNTHETIC_SAMPLES = 300
N_FIRMS = 50
N_CONTACTS = 150
N_EMPLOYEES = 20
N_MEETINGS = 300


class DummyMeetingResult(BaseModel):
    title: str
    content: str

    @classmethod
    def create_dummy_meeting(
        cls,
        firm_attended: str,
        firms_discussed: list[str],
        contacts: list[str],
        employees_attending: list[str],
        date: datetime,
    ) -> "DummyMeetingResult":
        interaction_type = np.random.choice(["Meeting", "Call", "Email"])
        title = f"{interaction_type} with {firm_attended}"

        additional_context = {
            "Meeting": dedent(
                """You are {agent_is} and you are an employee at Harvery's & Co, an investment bank. \
                Harvery's & Co specialise in investment banking, mergers and acquisitions, and asset management. \
                Today is {date} and you are attending a meeting with {firm_attended}. \
                Included in the discussion are your colleagues {employees_attending}, and {contacts} from {firm_attended}. \
                The agenda for the meeting is to discuss the following firms: {firms_discussed}. \
                Pretend you are {agent_is} in the meeting, taking notes on the discussion. \
                These notes should be very brief and to the point remarks of interest or concern in the discussion.\n\n
                    
                Your notes should be in the form of bullet points, and each point should be a single short sentence, sometimes in \
                shorthand. The notes should be concise and to the point, capturing the essence of the discussion."""
            ),
            "Call": dedent(
                """You are {agent_is} and you are an employee at Harvery's & Co, an investment bank. \
                Harvery's & Co specialise in investment banking, mergers and acquisitions, and asset management. \
                Today is {date} and you are in a call with representatives at {firm_attended}. \
                Included in the discussion are your colleagues {employees_attending}, and {contacts} from {firm_attended}. \
                The agenda for the call is to discuss the following firms: {firms_discussed}. \
                Pretend you are {agent_is} in the meeting, taking notes on the discussion. \
                These notes should be very brief and to the point remarks of interest or concern in the discussion.\n\n
                    
                Your notes should be in the form of bullet points, and each point should be a single short sentence, sometimes in \
                shorthand. The notes should be concise and to the point, capturing the essence of the discussion."""
            ),
            "Email": dedent(
                """You are {agent_is} and you are an employee at Harvery's & Co, an investment bank. \
                Harvery's & Co specialise in investment banking, mergers and acquisitions, and asset management. \
                Today is {date} and you are reading an email between you and {firm_attended}. \
                Included in the email thread are your colleagues {employees_attending}, and {contacts} from {firm_attended}. \
                The email is discussing the following firms: {firms_discussed}. \
                
                Write out an email thread between the parties involved, discussing the firms and any other relevant information."""
            ),
        }

        agent_is = np.random.choice(employees_attending)

        prompt_template = PromptTemplate(additional_context[interaction_type])
        llm = OpenAI(api_key=OPENAI_API_KEY, model=LLM_MODEL, max_tokens=MAX_TOKENS_SYNTHETIC_SAMPLES)

        content = cls.call_llm(
            prompt_template.format(
                agent_is=agent_is,
                date=date.strftime("%B %d, %Y"),
                firm_attended=firm_attended,
                firms_discussed=", ".join(firms_discussed),
                contacts=", ".join(contacts),
                employees_attending=", ".join(employees_attending),
            ),
            llm,
        )

        return cls(title=title, content=content)

    @staticmethod
    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def call_llm(prompt: str, llm: OpenAI) -> str:
        return llm.complete(prompt).text


def create_firms(n: int):
    with open("setup/firms.json", "r") as f:
        firm_selection_pool: list[dict[str, str]] = json.load(f)

    np.random.shuffle(firm_selection_pool)
    if n > len(firm_selection_pool):
        n = len(firm_selection_pool)

    firm_selection_pool = firm_selection_pool[:n]

    firms: list[models.Firms] = []
    for i in range(n):
        firm_data = firm_selection_pool[i]
        firm = models.Firms(
            name=firm_data["name"],
            sector=firm_data["sector"],
        )
        firms.append(firm)

    with session_scope() as session:
        session.add_all(firms)
        session.commit()


def create_fake_person() -> tuple[str, str]:
    name = FAKER.name()
    first_part = name.replace(" ", np.random.choice(["_", "."])).lower()
    middle_part = (
        f"_{FAKER.word()}"
        if np.random.choice([True, False])
        else str(np.random.randint(1, 100))
        if np.random.choice([True, False])
        else ""
    )
    last_part = "@" + np.random.choice(
        ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com"]
    )
    email = first_part + middle_part + last_part
    address = FAKER.address()
    return name, email, address


def create_contacts(n: int):
    with session_scope() as session:
        firms = session.query(models.Firms).all()

        contacts: list[models.Contacts] = []
        for i in range(n):
            name, email, address = create_fake_person()
            contact = models.Contacts(
                name=name,
                email=email,
                address=address,
                firm_id=np.random.choice(firms).firm_id,
            )
            contacts.append(contact)

        session.add_all(contacts)
        session.commit()


def create_employees(n: int):
    employees: list[models.Employees] = []
    for i in range(n):
        name, email, _ = create_fake_person()
        employee = models.Employees(
            name=name,
            email=email,
        )
        employees.append(employee)

    with session_scope() as session:
        session.add_all(employees)
        session.commit()


def create_meetings(n: int):
    with session_scope() as session:
        firms = session.query(models.Firms).all()
        employees = session.query(models.Employees).all()

        meetings: list[models.Meetings] = []
        pbar = tqdm(total=n)
        try:
            for i in range(n):
                pbar.update(1)
                firm_attended = np.random.choice(firms)
                size = np.random.randint(
                    0, min([5, len([x for x in firms if x != firm_attended])])
                )
                if size > 0:
                    firms_discussed = np.random.choice(
                        [x for x in firms if x != firm_attended],
                        size=size,
                        replace=False,
                    ).tolist()
                else:
                    firms_discussed = []

                firm_contacts = (
                    session.query(models.Contacts)
                    .filter(models.Contacts.firm_id == firm_attended.firm_id)
                    .all()
                )
                if len(firm_contacts) > 0:
                    size = np.random.randint(0, min([3, len(firm_contacts)]))
                    if size > 0:
                        contacts = np.random.choice(
                            firm_contacts, size=size, replace=False
                        ).tolist()
                    else:
                        contacts = []
                else:
                    contacts = []

                employees_attending = np.random.choice(
                    employees,
                    size=np.random.randint(1, min([4, len(employees)])),
                    replace=False,
                ).tolist()

                date = datetime(
                    np.random.randint(2019, 2026),
                    np.random.randint(1, 13),
                    np.random.randint(1, 29),
                )

                meeting_data = DummyMeetingResult.create_dummy_meeting(
                    firm_attended.name,
                    [x.name for x in firms_discussed],
                    [x.name for x in contacts],
                    [x.name for x in employees_attending],
                    date,
                )

                meeting = models.Meetings(
                    title=meeting_data.title,
                    content=meeting_data.content,
                    date=date,
                    firm_attended=firm_attended,
                    contacts=contacts,
                    employees=employees_attending,
                    firms_discussed=firms_discussed,
                )
                meetings.append(meeting)
                time.sleep(np.random.rand() + np.random.randint(0, 2))
        except KeyboardInterrupt:
            pass
        finally:
            session.add_all(meetings)
            session.commit()
            pbar.close()


def create_data(
    n_firms: int = N_FIRMS,
    n_contacts: int = N_CONTACTS,
    n_employees: int = N_EMPLOYEES,
    n_meetings: int = N_MEETINGS,
):
    create_firms(n_firms)
    create_contacts(n_contacts)
    create_employees(n_employees)
    create_meetings(n_meetings)


if __name__ == "__main__":
    create_data()
