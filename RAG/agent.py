# agent.py

from pydantic import BaseModel, Field
from typing import List, Literal
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Maintains the overall state for an agent debating Antarctic ice melt, including:
    - The user question
    - A list of stance grades (each either 'GlobalWarming' or 'NaturalCycle')
    - The final LLM output
    - Retrieved document identifiers
    - Whether the question is on-topic (i.e. about Antarctic ice melting)
    """
    question: str
    grades: List[Literal["GlobalWarming", "NaturalCycle"]]
    llm_output: str
    documents: List[str]
    on_topic: bool


class GradeQuestion(BaseModel):
    """
    Determines which side of the Antarctic ice‐melting debate
    a user’s question pertains to.
    """
    score: Literal["GlobalWarming", "NaturalCycle"] = Field(
        description=(
            "Stance implied by the question: 'GlobalWarming' if it argues that Antarctic ice is melting "
            "primarily due to anthropogenic climate change, or 'NaturalCycle' if it argues that the melting "
            "is part of recurring seasonal or long-term natural cycles of freezing and melting."
        )
    )


class GradeDocuments(BaseModel):
    """
    Indicates which stance a retrieved document supports
    in the Antarctic ice‐melting debate.
    """
    score: Literal["GlobalWarming", "NaturalCycle"] = Field(
        description=(
            "Stance supported by the document: 'GlobalWarming' if it attributes ice melt to human-driven climate change, "
            "or 'NaturalCycle' if it attributes melting to recurring or natural freeze–thaw cycles and climate variability."
        )
    )
