from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field


class Thought(BaseModel):
    thought: str = Field(
        description="A thought around the task you are performing."
    )
    conclusion: str = Field(
        description="A conclusion that you have made regarding your thought."
    )

    def __str__(self) -> str:
        return f"<thought> {self.thought} </thought>\n<conclusion> {self.conclusion} </conclusion>"


class ReasoningOutput(BaseModel):
    thoughts: list[Thought] = Field(
        description="List of thoughts and conclusions that you have made in the reasoning process."
    )
    possible: bool = Field(
        description="Whether you believe the query is possible to execute."
    )

    def get_thoughts(self) -> str:
        return "\n".join([str(t) for t in self.thoughts])


class SQLQuery(BaseModel):
    where_clauses: list[str] = Field(
        description=(
            "List of where clauses that will be inserted into the SQL query. "
            "This should be written as they would appear in the query without the 'WHERE' keyword."
        ),
        examples=[
            "date between '2021-01-01' and '2021-01-31'",
            "country ilike '%USA%'",
            "(firm_attended_name ilike '%Google%' or firm_attended_name ilike '%Facebook%')",
        ],
        default=[],
    )
    order_by_fields: list[str] = Field(
        description=(
            "List of fields that will be inserted into the SQL query for ordering the data. "
            "This should be a list of fields in order of priority for ordering the data."
        ),
        examples=["date", "country"],
        default=[],
    )
    order_by_direction: Optional[str] = Field(
        description="The direction of the ordering of the data.",
        examples=["asc", "desc"],
        default=None,
    )
    limit: Optional[int] = Field(
        description="The number of rows to limit the query to.",
        examples=[10, 20, 50],
        default=None,
    )

    def to_sql_statement(self, table_name: str) -> str:
        statement = "SELECT * FROM " + table_name
        if self.where_clauses:
            statement += "\nWHERE " + " AND ".join(self.where_clauses)
        if self.order_by_fields:
            statement += "\nORDER BY " + ", ".join(self.order_by_fields)
            if self.order_by_direction:
                statement += f" {self.order_by_direction}"
        if self.limit:
            statement += f"\nLIMIT {self.limit}"
        return statement
    
    
class ValidationOutput(BaseModel):
    thoughts: list[Thought] = Field(
        description="List of thoughts and conclusions around how your plan does / does not follow the instructions, concluding whether the instructions are being followed."
    )
    valid: bool = Field(
        description="Whether you believe the plan follows the instructions."
    )
    
    def get_thoughts(self) -> str:
        return "\n".join([str(t) for t in self.thoughts])


class SQLAgentOutput(BaseModel):
    text: str
    results_df: Optional[pd.DataFrame] = None

    class Config:
        arbitrary_types_allowed = True
