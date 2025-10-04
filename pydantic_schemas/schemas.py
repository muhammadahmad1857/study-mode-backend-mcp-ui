from enum import Enum
from typing import Any, List, Literal, Union
from pydantic import BaseModel, root_validator, Field

# Enum for part type
class PartType(str, Enum):
    """Type of part in the agent response."""
    TEXT = "text"
    REASONING = "reasoning"
    UI_RESOURCE = "ui-resource"

# Strict schema for UI resource
class UIResource(BaseModel):
    """
    Represents a UI resource, such as an image or file, referenced by URI.
    """
    uri: str = Field(..., description="The URI of the resource", example="https://example.com/resource.png")
    mimeType: str = Field("text/uri-list", description="The MIME type of the resource", example="image/png")
    text: str = Field(..., description="A description or label for the resource", example="Example image")
    type: Literal["resource", "UIResource"] = "UIResource"

    class Config:
        json_schema_extra = {
            "example": {
                "uri": "https://example.com/resource.png",
                "mimeType": "image/png",
                "text": "Example image",
                "type": "UIResource"
            }
        }

# Schema for each part
class Part(BaseModel):
    """
    Represents a part of the agent's response, which can be text, reasoning, or a UI resource.
    """
    type: PartType = Field(..., description="The type of the part", example="text")
    text: Any = Field(None, description="Text content for text or reasoning parts", example="This is a text part.")
    resource: Union[UIResource, None] = Field(
        None, description="UIResource object for ui-resource parts",
        example={
            "uri": "https://example.com/resource.png",
            "mimeType": "image/png",
            "text": "Example image",
            "type": "resource"
        }
    )

    @root_validator(pre=True)
    def set_correct_key(cls, values):
        t = values.get("type")
        # If type is ui-resource, move text to resource if needed
        if t == PartType.UI_RESOURCE:
            if "text" in values and not isinstance(values.get("resource"), dict):
                values["resource"] = values.pop("text")
        else:
            if "resource" in values and not isinstance(values.get("text"), str):
                values["text"] = values.pop("resource")
        return values

    class Config:
        json_schema_extra = {
            "examples": [
                {
                    "type": "text",
                    "text": "This is a text part.",
                    "resource": None
                },
                {
                    "type": "reasoning",
                    "text": "This is a reasoning part.",
                    "resource": None
                },
                {
                    "type": "ui-resource",
                    "text": None,
                    "resource": {
                        "uri": "https://example.com/resource.png",
                        "mimeType": "image/png",
                        "text": "Example image",
                        "type": "resource"
                    }
                }
            ]
        }

# Schema for the agent response
class AgentResponse(BaseModel):
    """
    The response from the agent, including the main content and a list of parts.
    """
    content: str = Field(..., description="The main content of the agent's response", example="Here is the answer to your question.")
    parts: List[Part] = Field(
        ..., 
        description="A list of parts that make up the agent's response.",
        example=[
            {
                "type": "text",
                "text": "This is a text part.",
                "resource": None
            },
            {
                "type": "ui-resource",
                "text": None,
                "resource": {
                    "uri": "https://example.com/resource.png",
                    "mimeType": "image/png",
                    "text": "Example image",
                    "type": "resource"
                }
            }
        ]
    )

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Here is the answer to your question.",
                "parts": [
                    {
                        "type": "text",
                        "text": "This is a text part.",
                        "resource": None
                    },
                    {
                        "type": "ui-resource",
                        "text": None,
                        "resource": {
                            "uri": "https://example.com/resource.png",
                            "mimeType": "image/png",
                            "text": "Example image",
                            "type": "resource"
                        }
                    }
                ]
            }
        }
