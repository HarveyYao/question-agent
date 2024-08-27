from pydantic import BaseModel, Field
from typing import List



class QuestionDetails(BaseModel):
    id: int = Field(..., description="The unique identifier of the question", gt=0)
    text: str = Field(..., description="The text of the question")
    question_type: str = Field(..., description="The type of the question")
    difficulty: str = Field(..., description="The difficulty level of the question")
    answers: List[str] = Field(..., description="List of possible answers")
    subject: str = Field(..., description="Subject related to the question")


