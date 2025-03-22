import argparse
import os
import sys
from datetime import datetime

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.db import models
from src.db.database import session_scope

SECTORS = [
    "Technology",
    "Finance",
    "Healthcare",
    "Real Estate",
    "Retail",
    "Manufacturing",
    "Energy",
    "Transportation",
    "Education",
    "Entertainment",
]


def create_firms(n: int):
    firms: list[models.Firms] = []
    for i in range(n):
        firm = models.Firms(
            name=f"Firm {i}",
            sector=np.random.choice(SECTORS),
        )
        firms.append(firm)

    with session_scope() as session:
        session.add_all(firms)
        session.commit()


def create_contacts(n: int):
    with session_scope() as session:
        firms = session.query(models.Firms).all()

        contacts: list[models.Contacts] = []
        for i in range(n):
            contact = models.Contacts(
                name=f"Contact {i}",
                email=f"email {i}",
                firm_id=np.random.choice(firms).firm_id,
            )
            contacts.append(contact)

        session.add_all(contacts)
        session.commit()


def create_employees(n: int):
    employees: list[models.Employees] = []
    for i in range(n):
        employee = models.Employees(
            name=f"Employee {i}",
            email=f"email {i}",
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
        for i in range(n):
            firm_attended = np.random.choice(firms)
            size = np.random.randint(0, min([5, len([x for x in firms if x != firm_attended])]))
            if size > 0:
                firms_discussed = np.random.choice(
                    [x for x in firms if x != firm_attended], size=size, replace=False
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
                employees, size=np.random.randint(1, min([4, len(employees)])), replace=False
            ).tolist()

            meeting = models.Meetings(
                title=f"Meeting {i}",
                content=f"Content {i}",
                date=datetime(
                    np.random.randint(2019, 2026),
                    np.random.randint(1, 13),
                    np.random.randint(1, 29),
                ),
                firm_attended=firm_attended,
                contacts=contacts,
                employees=employees_attending,
                firms_discussed=firms_discussed,
            )
            meetings.append(meeting)

        session.add_all(meetings)
        session.commit()


def main(
    n_firms: int,
    n_contacts: int,
    n_employees: int,
    n_meetings: int,
):
    create_firms(n_firms)
    create_contacts(n_contacts)
    create_employees(n_employees)
    create_meetings(n_meetings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Insert data into the database.")
    parser.add_argument("--n_firms", type=int, default=30, help="Number of firms to create")
    parser.add_argument("--n_contacts", type=int, default=100, help="Number of contacts to create")
    parser.add_argument("--n_employees", type=int, default=20, help="Number of employees to create")
    parser.add_argument("--n_meetings", type=int, default=2000, help="Number of meetings to create")

    args = parser.parse_args()

    main(
        n_firms=args.n_firms,
        n_contacts=args.n_contacts,
        n_employees=args.n_employees,
        n_meetings=args.n_meetings,
    )
