#!/usr/bin/env python3
"""
Simple MCP Server for LangGraph Platform
Provides basic tools for mathematical operations and text processing.
"""

from fastmcp import FastMCP

# Create the MCP server
mcp = FastMCP("LangGraph Tools Server")

@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The sum of a and b
    """
    return a + b

@mcp.tool()
def reverse_text(text: str) -> str:
    """Reverse the order of characters in a text string.
    
    Args:
        text: The text to reverse
        
    Returns:
        The reversed text
    """
    return text[::-1]

@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together.
    
    Args:
        a: First number
        b: Second number
        
    Returns:
        The product of a and b
    """
    return a * b

if __name__ == "__main__":
    # Run the server
    mcp.run()
