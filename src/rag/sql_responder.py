from textwrap import dedent

import pandas as pd
from llama_index.core import PromptTemplate
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt

from src.db.database import session_scope
from src.rag.sql_retriever import MeetingsSQLRetrieverAgent

prompt_template = PromptTemplate(
    dedent(
        """\
        You are a ChatBot built by Harvery's & Co, an investment bank. \
        Harvery's & Co specialise in investment banking, mergers and acquisitions, and asset management. \
        Your task is to answer employee's queries relating to company meeting notes held in a database. \
        Given a query by a user, a SQL AI Agent will try to find the relevant data in the database to answer the query. \
        Your task is to:\n
            1. read the user's query and the returned data from the database.\n
            2. analyse the retrieved data and how it might relate to the user's query.\n
            3a. report back to the user how many records were found in the database, and;\n
            3b. write a response back to the user to answer their query.\n\n

        ## IMPORTANT\n
        - **Your output must use the structured output format provided.**\n
        - **If the retrieved data does not answer the user's query, you must tell the user this and ask for more context to help you answer their query.**\n
        - **You must not make information up that does not exist in the database.**\n
        - **When referencing or citing meeting data from the database, you MUST provide the meetings.beam_id of the meeting encased in xml tags <ref>beam_id</ref>**\n
        - **You must stylise your response in markdown to make it easier to read by humans.**\n\n
        - **If there are too many records/meetings to cover in your response, always tell the user this and always focus on the most recent records.**\n\n

        # User Query:\n
        <query>{query}</query>\n\n

        # Retrieved Data:\n
        <data>{data}</data>\n\n
        """
    )
)

query_writer_template = PromptTemplate(
    dedent(
        """\
        You are a ChatBot built by Harvery's & Co, an investment bank. \
        Harvery's & Co specialise in investment banking, mergers and acquisitions, and asset management. \
        Your task is to answer employee's queries relating to company meeting notes held in a database. \
        You must accomplish this task by cooperating with a SQL AI Agent that can retrieve data from the database. \
        Given a query by a user, you must instruct the SQL AI Agent using ONLY natural language to retrieve the relevant data from the database.\n\n

        ## IMPORTANT
        - **It is important that you provide clear and concise instructions to the SQL AI Agent, including any dates, ids, or personal details the \
        user has mentioned that is relevant to their query.**\n
        - **It is always helpful to provide context on why you need the data, this will help the SQL AI Agent retrieve the correct data fields.**\n
        - **Only mention the user to the SQL AI Agent if it is relevant to the query. This should always be their employee_id and never their name.**\n
        - **You must not write SQL queries yourself, ONLY provide natural language instructions to the SQL AI Agent.**\n\n

        ## EXAMPLES\n
        <example>
        *In this query, the user is asking to know two things;*\n
        *1. What meetings were there in the last 2 months?*\n
        *2. Which ones did they not attend?*\n
        *This query can be fulfilled by getting all meetings in the last 2 months and then comparing the user's attendance to the meetings.*\n\n
        **User Query:** "What meetings were there in the last 2 months and which ones did I not attend?"\n
        **Your Output Instruction:** "Retrieve all meetings in the last 2 months."\n\n
        </example

        # User Query:\n
        <query>{query}</query>\n\n{error}
        """
    )
)


class Step(BaseModel):
    """
    Use this class to think about the problem and collect your thoughts before illiciting a response.

    Attributes:
    - thought: (str) - A thought process identifying requirements, challenges, or things needing consideration.
    - conclusion: (str) - Your conclusion on your thoughts and what you must do in your response.
    """

    thought: str
    conclusion: str


class Response(BaseModel):
    """
    Use this class to structure your response.

    Attributes:
    - steps: list[Step] - A list of Step objects.
    - response: (str) - Your response.
    """

    steps: list[Step]
    response: str


class FinalResponse(BaseModel):
    """
    Used to structure the final response to the user.
    This is not passed to a structured output LLM but rather programmatically returned to the user.

    Attributes:
    - response: (str) - Your response.
    - beam_ids: list[str] - A list of beam_ids that are referenced in the response.
    - response_clipped: (bool) - A boolean indicating if the response was clipped due to too many records.
    """

    response: str
    beam_ids: list[str]
    response_clipped: bool


class MeetingsSQLQnAAgent:
    def __init__(
        self,
        llm: LLM,
        agent: MeetingsSQLRetrieverAgent,
        prompt_template: PromptTemplate = prompt_template,
        query_writer_template: PromptTemplate = query_writer_template,
        output_format: type[Response] = Response,
        verbose: bool = False,
        _query_write_max_tokens: int = 250,
        _response_max_tokens: int = 4000,
        _max_query_attempts: int = 2,
        _max_rows_retrieved: int = 20,
    ):
        self.agent = agent
        self.output_format = output_format
        self.llm = llm
        self.prompt_template = prompt_template
        self.query_writer_template = query_writer_template
        self._query_write_max_tokens = _query_write_max_tokens
        self._response_max_tokens = _response_max_tokens
        self._max_query_attempts = _max_query_attempts
        self._verbose = verbose
        self._max_rows_retrieved = _max_rows_retrieved

    def _query_db(self, query: str) -> pd.DataFrame:
        with session_scope() as session:
            response = self.agent.complete(session, query)
        return response

    def _get_response_md(self, query: str) -> tuple[str, pd.DataFrame]:
        response = self._query_db(query)
        if isinstance(response, str):
            return response, pd.DataFrame()
        table_data = response.copy()
        records_found = len(response)
        response_str = "**The database returned {} records.**".format(records_found)
        if records_found > self._max_rows_retrieved:
            response_str += "\n\n**There are too many records to cover in this response, showing the most recent meetings only.**"
            table_data = table_data.head(self._max_rows_retrieved)
        if records_found > 0:
            markdown_table = table_data.to_markdown(index=True)
            response_str += "\n\n{}".format(markdown_table)
            response_str += "\n\n**The database returned {} records.**".format(
                records_found
            )
            if records_found > self._max_rows_retrieved:
                response_str += "\n\n**There are too many records to cover in this response, showing the most recent meetings only.**"
        return response_str, response

    @retry(stop=stop_after_attempt(3))
    def _invoke_llm(
        self,
        output_format: BaseModel,
        prompt_template: PromptTemplate,
        max_tokens: int,
        **kwargs,
    ) -> Response:
        try:
            return (
                self.llm.as_structured_llm(output_format)
                .complete(prompt_template.format(**kwargs), max_tokens=max_tokens)
                .raw
            )
        except Exception as e:
            print("ERROR:", e)
            raise e

    def _sql_instruction(self, query: str) -> tuple[str, list[str]]:
        attempt = 0
        error = ""
        while True:
            try:
                ai_query = self._invoke_llm(
                    self.output_format,
                    self.query_writer_template,
                    self._query_write_max_tokens,
                    query=query,
                    error=error,
                )
                if self._verbose:
                    print("AI QUERY:")
                    print(ai_query.response)
                data, data_raw = self._get_response_md(ai_query.response)
                if "beam_id" in data_raw.columns:
                    beam_ids = list(set(data_raw["beam_id"].astype(str).tolist()))
                return data, beam_ids

            except Exception as e:
                error = """\n\n**Your request on the last attempt failed. \
                    Please try to provide more context (if available) or re-phrase the query to the SQL AI Agent.**\n
                    **Your Last Query**: "{ai_query}"\n
                    **Error Message**: {e}
                """.format(
                    ai_query=ai_query.response, e=e
                )
                attempt += 1
                if attempt >= self._max_query_attempts:
                    return (
                        "Report back to the user that you are struggling to understand the user's query and ask for more context.",
                        [],
                    )
                continue

    def complete(self, query: str) -> FinalResponse:
        try:
            data, beam_ids = self._sql_instruction(query)
            if self._verbose:
                print("RETURNED DATA:")
                print(data)
            response = self._invoke_llm(
                self.output_format,
                self.prompt_template,
                self._response_max_tokens,
                query=query,
                data=data,
            )
            return FinalResponse(
                response=response.response,
                beam_ids=beam_ids if beam_ids else [],
                response_clipped=True
                if len(beam_ids) > self._max_rows_retrieved
                else False,
            )
        except Exception as e:
            print("ERROR:", e)
            return FinalResponse(
                response="Sorry, there seems to be an issue understanding your query. Please can you provide more context?",
                beam_ids=[],
                response_clipped=False,
            )
