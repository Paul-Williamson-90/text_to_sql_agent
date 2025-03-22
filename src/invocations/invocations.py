import asyncio
import json
import logging
import re
from ast import literal_eval
from typing import Any, Callable, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.exceptions import OutputParserException
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    MessageRole,
)
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.llms.llm import LLM
from llama_index.core.memory import ChatMemoryBuffer
from pydantic import BaseModel
from tenacity import before_log, retry, stop_after_attempt, wait_fixed

from src.invocations.constants import (
    REPLACEMENT_PATTERNS,
    STOP_AFTER_ATTEMPT,
    WAIT_FIXED,
)
from src.invocations.prompts import (
    DEFAULT_CHOICE_VALIDATION_PROMPT_TEMPLATE,
    DEFAULT_STRUCTURED_PROMPT_TEMPLATE,
)
from src.invocations.pydantics import Choice, Choices, ValidatorChoice
from src.invocations.types import InvocationCallable

logger = logging.getLogger(__name__)


def decode_either(data: str) -> dict:
    """Function for decoding either JSON or YAML data.

    Parameters
    ----------
    data : str
        The data to be decoded.

    Returns
    -------
    dict
        The decoded data.
    """
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        for pattern in REPLACEMENT_PATTERNS:
            data = re.sub(pattern[0], pattern[1], data, flags=re.IGNORECASE)
        return literal_eval(data)


@retry(
    stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
    wait=wait_fixed(WAIT_FIXED),
    before=before_log(logger, logging.INFO),
)
def structured_invocation(
    llm: LLM,
    context: str,
    pydantic_object: type[BaseModel],
    prompt_template: PromptTemplate = DEFAULT_STRUCTURED_PROMPT_TEMPLATE,
    llm_kwargs: dict[str, Any] = {},
    validation_callable: Optional[Callable] = None,
) -> BaseModel:
    """A function for invoking an LLM with a structured output. Defined separately from \
    Llama-Index built-in structured invocation due to some LLM modules structured output \
    methods are yet to be implemented.

    Parameters
    ----------
    llm : LLM
        The LLM module to be invoked.
    context : str
        The context of the prompt.
    pydantic_object : type[BaseModel]
        The Pydantic object to be used for the structured output.
    prompt_template : PromptTemplate, optional
        A prompt template for combining the context and Pydantic schema, by default DEFAULT_STRUCTURED_PROMPT_TEMPLATE
    llm_kwargs : dict[str, Any], optional
        Inference kwargs passed to the LLM, by default {}
    validation_callable : Optional[Callable], optional
        A validation callable to be used for the structured output, by default None

    Returns
    -------
    BaseModel
        The structured output from the LLM.
    """
    parser = PydanticOutputParser(pydantic_object=pydantic_object)

    prompt = prompt_template.format(
        context=context,
        schema=parser.get_format_instructions(),
    )

    parser = PydanticOutputParser(pydantic_object=pydantic_object)

    response = llm.complete(
        prompt=prompt,
        **llm_kwargs,
    )
    logger.info(f"Structured invocation response: {response}")
    response_str = str(response.text).strip()

    try:
        response_object = parser.parse(response_str)
    except OutputParserException:
        response_json = decode_either(response_str)
        response_object = pydantic_object(**response_json)

    if validation_callable:
        response_object = validation_callable(response_object)  # type: ignore

    return response_object  # type: ignore


@retry(
    stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
    wait=wait_fixed(WAIT_FIXED),
    before=before_log(logger, logging.INFO),
)
def non_structured_invocation(
    llm: LLM,
    prompt: str,
    inference_kwargs: dict[str, Any] = {},
) -> str:
    """Simple function for invoking an LLM completion process.

    Parameters
    ----------
    llm : LLM
        The LLM module to be invoked.
    prompt : str
        The prompt to be used for the LLM.
    inference_kwargs : dict[str, Any], optional
        Inference kwargs passed to the LLM, by default {}

    Returns
    -------
    str
        The response from the LLM.
    """
    response: CompletionResponse = llm.complete(prompt=prompt, **inference_kwargs)
    response_str = str(response)

    return response_str


@retry(
    stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
    wait=wait_fixed(WAIT_FIXED),
    before=before_log(logger, logging.INFO),
)
def non_structured_streamed_invocation(
    llm: LLM,
    prompt: str,
    memory: Optional[ChatMemoryBuffer] = None,
    inference_kwargs: dict[str, Any] = {},
) -> StreamingAgentChatResponse:
    """Function for invoking an LLM completion process and streaming the response.

    Parameters
    ----------
    llm : LLM
        The LLM module to be invoked.
    prompt : str
        The prompt to be used for the LLM.
    memory : Optional[ChatMemoryBuffer], optional
        The memory buffer to store the response, by default None
    inference_kwargs : dict[str, Any], optional
        Inference kwargs passed to the LLM, by default {}

    Returns
    -------
    StreamingAgentChatResponse
        The streaming response from the LLM.
    """
    response: CompletionResponse = llm.stream_complete(
        prompt=prompt, **inference_kwargs
    )

    def wrapped_gen(response: CompletionResponseGen) -> ChatResponseGen:
        full_response = ""
        for token in response:
            if token.delta:
                full_response += token.delta
                yield ChatResponse(
                    message=ChatMessage(content=token.text, role=MessageRole.ASSISTANT),
                    delta=token.delta,
                )

        if memory:
            assistant_message = ChatMessage(
                content=full_response, role=MessageRole.ASSISTANT
            )
            memory.put(assistant_message)

    return StreamingAgentChatResponse(
        chat_stream=wrapped_gen(response),
        sources=[],
        source_nodes=[],
        is_writing_to_memory=False,
    )


class MultiInvocationWithValidation:
    _pydantic_object: BaseModel = ValidatorChoice

    def __init__(
        self,
        llm: Optional[LLM] = None,
        choices: int = 3,
        inference_kwargs: dict[str, Any] = {},
        prompt_template: PromptTemplate = DEFAULT_CHOICE_VALIDATION_PROMPT_TEMPLATE,
        timeout: int = 30,
    ):
        """A class for validating multiple invocations.

        Parameters
        ----------
        llm : Optional[LLM], optional
            The validator LLM, by default None (uses the LLM passed in the validate method)
        choices : int, optional
            Number of generated choices to choose from, by default 3
        inference_kwargs : dict[str, Any], optional
            Kwargs passed during validation, by default {}
        prompt_template : PromptTemplate, optional
            The prompt template used for the validation prompt, by default DEFAULT_CHOICE_VALIDATION_PROMPT_TEMPLATE
        timeout : int, optional
            The timeout for each invocation task, by default 30
        """
        self.llm = llm
        self.choices = choices
        self.inference_kwargs = inference_kwargs
        self.prompt_template = prompt_template
        self._timeout = timeout

    @retry(
        stop=stop_after_attempt(STOP_AFTER_ATTEMPT),
        wait=wait_fixed(WAIT_FIXED),
        before=before_log(logger, logging.INFO),
    )
    async def validate(self, llm: LLM, prompt: str, choices: Choices) -> Choice:
        """A function for validating the choices.

        Parameters
        ----------
        llm : LLM
            The LLM module to be invoked.
        prompt : str
            The prompt to be used for the validation.
        choices : Choices
            The choices to be validated.

        Returns
        -------
        Choice
            The validated choice.
        """
        validator_llm = self.llm or llm
        context = self.prompt_template.format(prompt=prompt, choices=str(choices))

        def validation_callable(response: ValidatorChoice) -> Choice:
            return choices.get_choice_by_identifier(response.identifier)

        response: Choice = structured_invocation(
            llm=validator_llm,
            context=context,
            pydantic_object=self._pydantic_object,
            llm_kwargs=self.inference_kwargs,
            validation_callable=validation_callable,
        )
        return response

    async def structured_invocation(
        self,
        llm: LLM,
        context: str,
        pydantic_object: type[BaseModel],
        prompt_template: PromptTemplate = DEFAULT_STRUCTURED_PROMPT_TEMPLATE,
        llm_kwargs: dict[str, Any] = {},
    ) -> BaseModel:
        """A function for invoking an LLM with a structured output.

        Parameters
        ----------
        llm : LLM
            The LLM module to be invoked.
        context : str
            The context of the prompt.
        pydantic_object : type[BaseModel]
            The Pydantic object to be used for the structured output.
        prompt_template : PromptTemplate, optional
            The prompt template for combining the context and Pydantic schema, by default DEFAULT_STRUCTURED_PROMPT_TEMPLATE
        llm_kwargs : dict[str, Any], optional
            Inference kwargs passed to the LLM, by default {}

        Returns
        -------
        BaseModel
            The structured output from the LLM.
        """
        prompt = prompt_template.format(
            context=context,
            schema=json.dumps(pydantic_object.model_json_schema()),
        )

        logger.info(f"Structured invocation prompt: {prompt}")

        tasks = []
        for _ in range(self.choices):
            tasks.append(
                self._invocation_task(
                    structured_invocation,
                    llm,
                    context,
                    pydantic_object,
                    prompt_template,
                    llm_kwargs,
                )
            )
        results = await asyncio.gather(*tasks)

        choices = Choices(choices=[Choice(choice=result) for result in results])
        choice = await self.validate(llm, prompt, choices)
        return choice.choice

    async def non_structured_invocation(
        self,
        llm: LLM,
        prompt: str,
        inference_kwargs: dict[str, Any] = {},
    ) -> str:
        """Simple function for invoking an LLM completion process.

        Parameters
        ----------
        llm : LLM
            The LLM module to be invoked.
        prompt : str
            The prompt to be used for the LLM.
        inference_kwargs : dict[str, Any], optional
            Inference kwargs passed to the LLM, by default {}

        Returns
        -------
        str
            The response from the LLM.
        """
        logger.info(f"Non-structured invocation prompt: {prompt}")
        tasks = []
        for _ in range(self.choices):
            tasks.append(
                self._invocation_task(
                    structured_invocation, llm, prompt, inference_kwargs
                )
            )
        results = await asyncio.gather(*tasks)
        choices = Choices(choices=[Choice(choice=result) for result in results])
        choice = await self.validate(llm, prompt, choices)
        return choice.choice

    async def _invocation_task(self, invocation_call: InvocationCallable, *args):
        """A function for running an invocation call in a separate task.

        Parameters
        ----------
        invocation_call : InvocationCallable
            The invocation call to be run.
        *args
            The arguments to be passed to the invocation call.

        Returns
        -------
        Any
            The result of the invocation call
        """
        return await asyncio.wait_for(
            asyncio.get_event_loop().run_in_executor(None, invocation_call, *args),
            timeout=self._timeout,
        )


invocation_validator = MultiInvocationWithValidation()
