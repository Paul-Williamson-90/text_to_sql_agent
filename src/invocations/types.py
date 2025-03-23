from typing import Any, Callable, Union

from llama_index.core import PromptTemplate
from llama_index.core.llms.llm import LLM
from pydantic import BaseModel

InvocationCallable = Union[
    Callable[[LLM, str, type[BaseModel], PromptTemplate, dict[str, Any]], BaseModel],
    Callable[[LLM, str, dict[str, Any]], str],
]
