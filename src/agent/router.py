import logging
from pathlib import Path
from typing import Any

import pandas as pd
from llama_index.core.base.llms.types import MessageRole
from llama_index.core.llms import ChatMessage
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.workflow import Context, StartEvent, StopEvent, Workflow, step
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.agent.constants import DEFAULT_TOKEN_LIMIT, DENORM_QUERY
from src.agent.events import (
    DataReturnEvent,
    FailureEvent,
    ReasoningEvent,
    SQLExecutionEvent,
    SQLValidationEvent,
    SQLWritingEvent,
)
from src.agent.prompts import (
    FAILURE_OUTPUT,
    REASONING_STEP_PROMPT,
    SQL_WRITING_PROMPT,
    SYSTEM,
    TOOL_ERROR,
    VALIDATION_STEP_PROMPT,
)
from src.agent.pydantics import (
    ReasoningOutput,
    SQLAgentOutput,
    SQLQuery,
    ValidationOutput,
)
from src.invocations import invocation_validator

logger = logging.getLogger(__name__)


class SQLAgent(Workflow):
    _llm_kwargs: dict[str, Any] = {"max_tokens": 500}
    _max_rounds: int = 3

    def __init__(
        self,
        llm: LLM,
        session: Session,
        table_name: str = "meeting_notes",
        schema_path: Path = Path("src/schemas/meetings_denorm_schema.txt"),
        denormalized_query_path: Path = Path("src/schemas/meetings_denorm.sql"),
        system_prompt: str = SYSTEM,
    ):
        super().__init__(timeout=None)
        self.llm = llm
        self.session = session
        self.table_name = table_name
        self.denormalized_query: str = denormalized_query_path.read_text()
        self.system_prompt = system_prompt.format(
            table_name=table_name,
            schema=schema_path.read_text(),
        )

    @step
    async def start_step(self, ctx: Context, ev: StartEvent) -> ReasoningEvent:
        logger.info(f"[{type(self.__class__)}]: start_step.")
        user_query = str(ev.input)
        memory = ChatMemoryBuffer(token_limit=DEFAULT_TOKEN_LIMIT)
        await ctx.set("user_query", user_query)
        await ctx.set("memory", memory)
        await ctx.set("result_df", None)
        await ctx.set("rounds", 0)
        return ReasoningEvent()

    @step
    async def reasoning_step(
        self, ctx: Context, ev: ReasoningEvent
    ) -> SQLValidationEvent | FailureEvent:
        logger.info(f"[{type(self.__class__)}]: reasoning_step.")

        rounds = await ctx.get("rounds")
        if rounds >= self._max_rounds:
            return FailureEvent(
                thoughts="The SQL Agent has failed to find the data after a deep search."
            )
        else:
            rounds += 1
            await ctx.set("rounds", rounds)

        user_query = await ctx.get("user_query")
        memory: ChatMemoryBuffer = await ctx.get("memory")
        messages = memory.get_all()

        prompt = REASONING_STEP_PROMPT.format(
            system=self.system_prompt,
            user_query=user_query,
            history="\n".join([str(m) for m in messages]),
        )

        response: ReasoningOutput = await invocation_validator.structured_invocation(
            llm=self.llm,
            context=prompt,
            pydantic_object=ReasoningOutput,
            llm_kwargs=self._llm_kwargs,
        )

        thoughts = response.get_thoughts()
        memory.put(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=thoughts,
            )
        )
        await ctx.set("memory", memory)

        logger.info(f"[{type(self.__class__)}]: {response}.")

        if not response.possible:
            return FailureEvent(thoughts=thoughts)

        return SQLValidationEvent(thoughts=thoughts)
    
    @step
    async def sql_validation_step(self, ctx: Context, ev: SQLValidationEvent) -> SQLWritingEvent | ReasoningEvent:
        logger.info(f"[{type(self.__class__)}]: sql_validation_step.")
        thoughts = ev.thoughts
        user_query = await ctx.get("user_query")
        
        prompt = VALIDATION_STEP_PROMPT.format(
            system=self.system_prompt,
            plan=thoughts,
            table_name=self.table_name,
            user_query=user_query,
        )

        response: ValidationOutput = await invocation_validator.structured_invocation(
            llm=self.llm,
            context=prompt,
            pydantic_object=ValidationOutput,
            llm_kwargs=self._llm_kwargs,
        )
        
        if not response.valid:
            memory: ChatMemoryBuffer = await ctx.get("memory")
            memory.put(
                ChatMessage(
                    role=MessageRole.ASSISTANT,
                    content=response.get_thoughts(),
                )
            )
            await ctx.set("memory", memory)
            return ReasoningEvent()
        
        return SQLWritingEvent()

    @step
    async def failure_exit_step(self, ev: FailureEvent) -> StopEvent:
        logger.info(f"[{type(self.__class__)}]: failure_exit_step.")
        thoughts = ev.thoughts
        output = FAILURE_OUTPUT.format(thoughts=thoughts)
        return StopEvent(result=SQLAgentOutput(text=output))

    @step
    async def sql_writing_step(
        self, ctx: Context, ev: SQLWritingEvent
    ) -> SQLExecutionEvent:
        logger.info(f"[{type(self.__class__)}]: sql_writing_step.")

        user_query = await ctx.get("user_query")
        memory: ChatMemoryBuffer = await ctx.get("memory")
        messages = memory.get_all()

        prompt = SQL_WRITING_PROMPT.format(
            system=self.system_prompt,
            user_query=user_query,
            history="\n".join([str(m) for m in messages]),
        )

        response: SQLQuery = await invocation_validator.structured_invocation(
            llm=self.llm,
            context=prompt,
            pydantic_object=SQLQuery,
            llm_kwargs=self._llm_kwargs,
        )

        memory.put(
            ChatMessage(
                role=MessageRole.ASSISTANT,
                content=str(response),
            )
        )
        await ctx.set("memory", memory)

        logger.info(
            f"[{type(self.__class__)}]: {response.to_sql_statement(self.table_name)}."
        )

        return SQLExecutionEvent(query=response.to_sql_statement(self.table_name))

    @step
    async def sql_execution_step(
        self, ctx: Context, ev: SQLExecutionEvent
    ) -> ReasoningEvent | DataReturnEvent:
        logger.info(f"[{type(self.__class__)}]: sql_execution_step.")

        query = ev.query
        memory: ChatMemoryBuffer = await ctx.get("memory")

        try:
            processed_query = (
                DENORM_QUERY.format(
                    table_name=self.table_name, subquery=self.denormalized_query
                )
                + query
            )
            result = self.session.execute(text(processed_query)).fetchall()
            result_df = pd.DataFrame(result)

        except Exception as e:
            error_str = str(e)
            memory.put(
                ChatMessage(
                    role=MessageRole.TOOL,
                    content=TOOL_ERROR.format(error=error_str),
                    additional_kwargs={"tool_call_id": self.table_name},
                )
            )
            await ctx.set("memory", memory)
            return ReasoningEvent()

        await ctx.set("result_df", result_df)

        return DataReturnEvent()

    @step
    async def data_exit_step(self, ctx: Context, ev: DataReturnEvent) -> StopEvent:
        logger.info(f"[{type(self.__class__)}]: data_exit_step.")
        data: pd.DataFrame = await ctx.get("result_df")
        logger.info(str(data.to_markdown()))
        return StopEvent(
            result=SQLAgentOutput(
                text="The data returned from the SQL query is valid.",
                results_df=data,
            )
        )
