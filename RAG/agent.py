from pydantic import BaseModel, Field
from typing import List, Literal
from typing_extensions import TypedDict


class AgentState(TypedDict):
    """
    Maintains the overall state for an agent analyzing Antarctic ice mass change questions,
    including:
    - The user question text
    - A list of detected stances per question or document (each 'GlobalWarming' or 'NaturalCycle')
    - The generated, evidence-based LLM response
    - Retrieved document contents or identifiers used
    - Whether the question is on topic (about Antarctic ice)
    """
    question: str
    grades: List[Literal["GlobalWarming", "NaturalCycle"]]
    llm_output: str
    documents: List[str]
    on_topic: bool


class GradeQuestion(BaseModel):
    """
    Classifies which perspective a user's question implies about Antarctic ice change.
    """
    score: Literal["GlobalWarming", "NaturalCycle"] = Field(
        description=(
            "Indicates the viewpoint implied by the user question:"
            " 'GlobalWarming' if it attributes ice change to human-driven warming,"
            " 'NaturalCycle' if it frames change as part of natural freeze/melt cycles."
        )
    )


class GradeDocuments(BaseModel):
    """
    Classifies which perspective a retrieved document supports about Antarctic ice change.
    """
    score: Literal["GlobalWarming", "NaturalCycle"] = Field(
        description=(
            "Indicates the perspective supported by the document:"
            " 'GlobalWarming' for human-driven climate impact,"
            " 'NaturalCycle' for recurring natural variability."
        )
    )
