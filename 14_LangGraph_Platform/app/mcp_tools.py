"""Custom MCP tools wrapped as LangChain tools for LangGraph integration."""

from typing import Type, Optional
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


class AddNumbersInput(BaseModel):
    """Input for add_numbers tool."""
    a: float = Field(description="First number to add")
    b: float = Field(description="Second number to add")


class AddNumbersTool(BaseTool):
    """Tool to add two numbers."""
    name: str = "add_numbers"
    description: str = "Add two numbers together. Useful for basic arithmetic calculations."
    args_schema: Type[BaseModel] = AddNumbersInput

    def _run(self, a: float, b: float) -> str:
        """Add two numbers."""
        result = a + b
        return f"The sum of {a} and {b} is {result}"


class ReverseTextInput(BaseModel):
    """Input for reverse_text tool."""
    text: str = Field(description="Text to reverse")


class ReverseTextTool(BaseTool):
    """Tool to reverse text."""
    name: str = "reverse_text"
    description: str = "Reverse the order of characters in a text string."
    args_schema: Type[BaseModel] = ReverseTextInput

    def _run(self, text: str) -> str:
        """Reverse text."""
        reversed_text = text[::-1]
        return f"The reversed text is: {reversed_text}"


class MultiplyNumbersInput(BaseModel):
    """Input for multiply_numbers tool."""
    a: float = Field(description="First number to multiply")
    b: float = Field(description="Second number to multiply")


class MultiplyNumbersTool(BaseTool):
    """Tool to multiply two numbers."""
    name: str = "multiply_numbers"
    description: str = "Multiply two numbers together. Useful for basic arithmetic calculations."
    args_schema: Type[BaseModel] = MultiplyNumbersInput

    def _run(self, a: float, b: float) -> str:
        """Multiply two numbers."""
        result = a * b
        return f"The product of {a} and {b} is {result}"


def get_mcp_tools():
    """Return list of MCP tools wrapped as LangChain tools."""
    return [
        AddNumbersTool(),
        ReverseTextTool(),
        MultiplyNumbersTool(),
    ]
