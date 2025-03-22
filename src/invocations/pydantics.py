import uuid

from pydantic import BaseModel, Field


class Choice(BaseModel):
    identifier: uuid.UUID = Field(default_factory=uuid.uuid4)
    choice: str | BaseModel

    def __str__(self) -> str:
        return """{}: {}""".format(self.identifier, str(self.choice))


class Choices(BaseModel):
    choices: list[Choice]

    def __str__(self) -> str:
        return "\n\n".join([str(choice) for choice in self.choices])

    def get_choice_by_identifier(self, identifier: uuid.UUID) -> Choice:
        for choice in self.choices:
            if choice.identifier == identifier:
                return choice
        raise ValueError(f"Choice with identifier {identifier} not found.")


class Step(BaseModel):
    """Use this schema to think through the task and provide justification for your choice.

    Parameters
    ----------
    thought : str
        Your thought.
    conclusion : str
        Your conclusion.
    """

    thought: str
    conclusion: str


class ValidatorChoice(BaseModel):
    """Use this schema to think through the choices available and provide justification for your choice, as well as the identifier of the choice.

    Parameters
    ----------
    thoughts : Step
        Your thoughts.
    identifier : uuid.UUID
        The identifier of the choice output.
    """

    thoughts: Step
    identifier: uuid.UUID
