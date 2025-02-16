from typing import Optional

import pandas as pd
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import text
from tenacity import retry, stop_after_attempt

from llama_index.core import PromptTemplate
from llama_index.core.llms.llm import LLM


class InvalidQueryError(Exception):
    pass


class Thoughts(BaseModel):
    """
    Use this class to think through the problem step-by-step towards constructing the SQL query.

    Attributes:
        thoughts: str - Thinking out loud what things you need to consider to faciliate the user's query.
        outcome: str - What will need to be added to the SQL query to get the desired result base on the thoughts.
    """

    thoughts: str
    outcome: str


class TextToSQL(BaseModel):
    """
    Use this to construct the SQL query to retrieve the correct information from the database according to the user's query.

    Attributes:
        step: list[Thoughts] - List of thoughts to consider to construct the SQL query.
        joins: Optional[list[str]] - List of joins you will need to add to the SQL query if required (default is an empty list).
        filters: Optional[list[str]] - List of filters you will need to add to the SQL query if required (default is an empty list).
        fields: list[str] - List of fields to add to the SQL query, constructed as 'table_name.field_name'.
        query: str - The final SQL query to be executed.
        possible: bool - If the query is not possible to execute, set this to False.
    """

    steps: list[Thoughts]
    joins: Optional[list[str]] = []
    filters: Optional[list[str]] = []
    fields: list[str] = []
    query: str
    possible: bool


prompt_template = PromptTemplate(
    """You are a ChatBot built by Harvery's & Co, an investment bank. \
Harvery's & Co specialise in investment banking, mergers and acquisitions, and asset management. \
Given an input question, you are to create a syntactically correct postgresql \
query to run that will provide the user with the data relevant to their question. \

Pay attention to use only the column names that you can see in the provided schema \
below between the <schema></schema> xml tags which is directly from a sqlalchemy python file. \
Be careful to not query for columns that do not exist. \
Pay attention to which column is in which table. Also, qualify column names \
with the table name when needed.

Only use tables listed below.
<schema>{schema}</schema>

Your response must be formatted as according to the structured output format provided.
Ensure that your sql query is well formatted for readability and syntactically correct. \
When returning meetings, you MUST always include the beam_id field. \
You must always order meetings by date in ascending order. When searching free-text fields, \
you must use the ILIKE operator to perform a case-insensitive search.

User Question: {query}
"""
)


class SQLAgent:
    _validation_words: list[str] = ["delete", "update", "insert", "drop"]

    def __init__(
        self,
        llm: LLM,
        schema_file_path: str,
        output_template: type[TextToSQL] = TextToSQL,
        prompt_template: PromptTemplate = prompt_template,
        verbose: bool = False,
    ):
        """
        SQLAgent class for generating and executing SQL queries based on user input.
        The class uses a Llama-Index LLM model to generate the SQL query based on the user input.
        The SQL query is then executed on the provided database session via SQLAlchemy.

        Args:
            llm (LLM): The Llama-Index LLM model to use for generating the SQL query.
            schema_file_path (str): The path to the schema file to use for the SQL query.
            output_template (type[TextToSQL], optional): The output template to use for the SQL query. Defaults to TextToSQL.
            prompt_template (PromptTemplate, optional): The prompt template to use for the LLM model. Defaults to prompt_template.
            verbose (bool, optional): Whether to print verbose output. Defaults to False.
        """
        self.llm = llm.as_structured_llm(output_template)
        self.schemas_file_str: str = open(schema_file_path, "r").read()
        self.output_template = output_template
        self.prompt_template = prompt_template
        self.verbose = verbose

    def _complete(self, query: str) -> TextToSQL:
        response = self.llm.complete(
            self.prompt_template.format(query=query, schema=self.schemas_file_str)
        ).raw
        return response

    def _show_cot(self, response: TextToSQL):
        print("CHAIN OF THOUGHTS:")
        for step in response.steps:
            print(f"Thoughts: {step.thoughts}")
            print(f"Outcome: {step.outcome}", "\n")

    def _show_sql_query(self, response: TextToSQL):
        print("SQL QUERY:")
        print("```\n" + response.query + "\n```")

    def _valid_query(self, response: TextToSQL) -> bool:
        # TODO: Implement regex for better validation.
        for word in self._validation_words:
            if word.lower() in response.query.lower():
                return False
        return True

    def _return_not_possible_thoughts(self, response: TextToSQL) -> str:
        response = (
            "The SQL Agent has determined that the query is not possible to execute. \
        Here are the thoughts that led to this conclusion:\n"
        )
        for step in response.steps:
            response += f"Thoughts: {step.thoughts}\n"
            response += f"Outcome: {step.outcome}\n\n"
        return response

    def _execute_query(self, session: Session, response: TextToSQL) -> pd.DataFrame:
        result = session.execute(text(response.query)).fetchall()
        columns = response.fields
        return pd.DataFrame(result, columns=columns)

    @retry(stop=stop_after_attempt(3))
    def complete(self, session: Session, query: str) -> pd.DataFrame:
        """
        Send the user query to the LLM model to generate the SQL query and execute it on the provided database session.

        Args:
            session (Session): The SQLAlchemy database session to execute the SQL query on.
            query (str): The user query to generate the SQL query from.

        Returns:
            pd.DataFrame: The result of the SQL query executed on the database session.
        """
        session.flush()  # Ensure that the session is clean before executing the query.
        response = self._complete(query)
        if self.verbose:
            self._show_cot(response)
            self._show_sql_query(response)
        if not self._valid_query(response):
            raise InvalidQueryError(
                "Invalid Query Detected, database does not permit deletion, insertion, or update operations."
            )
        if not response.possible:
            return self._return_not_possible_thoughts(response)
        results = self._execute_query(session, response)
        return results
