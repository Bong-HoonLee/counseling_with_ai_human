from pydantic import BaseModel, Field, model_validator

class BaseOutput(BaseModel):
    '''
    # Define your desired data structure.
    class Output_structure(BaseModel):
        setup: str = Field(description="question to set up a joke")
        punchline: str = Field(description="answer to resolve the joke")

        @model_validator(mode="before")
        @classmethod
        def question_ends_with_question_mark(cls, values: dict) -> dict:
            setup = values.get("setup")
            if setup and setup[-1] != "?":
                raise ValueError("Badly formed question!")
            return values
    # You can add custom validation logic easily with Pydantic.
    '''
    pass

class RagOutput(BaseOutput):
    Answer: str = Field(description="Answer the user query")

class ConvOutput(BaseOutput):
    Answer: str = Field(description="Answer the user query")
    

class IntentOutput(BaseOutput):
    intent: str
    confidence: float