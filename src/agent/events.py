from llama_index.core.workflow import Event


class ReasoningEvent(Event):
    pass


class SQLWritingEvent(Event):
    pass


class SQLExecutionEvent(Event):
    query: str


class SQLValidationEvent(Event):
    thoughts: str


class FailureEvent(Event):
    thoughts: str


class DataReturnEvent(Event):
    pass
