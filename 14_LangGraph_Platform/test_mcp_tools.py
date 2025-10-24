#!/usr/bin/env python3
"""Test script to validate MCP tools integration with LangGraph."""

from langgraph_sdk import get_sync_client


def test_mcp_tools():
    """Test that MCP tools are available and working."""
    client = get_sync_client(url="http://localhost:2024")
    
    print("Testing MCP tools integration...")
    print("=" * 50)
    
    # Test with a query that should trigger our MCP tools
    for chunk in client.runs.stream(
        None,  # Threadless run
        "simple_agent",  # Assistant id from langgraph.json
        input={
            "messages": [
                {
                    "role": "human",
                    "content": "What is 15 plus 27? Please use the add_numbers tool.",
                }
            ]
        },
        stream_mode="updates",
    ):
        print(f"Event type: {chunk.event}")
        if chunk.event == "updates":
            for node_name, node_data in chunk.data.items():
                if "messages" in node_data:
                    for message in node_data["messages"]:
                        if message.get("type") == "ai":
                            print(f"AI Response: {message.get('content', '')}")
                            if message.get("tool_calls"):
                                print(f"Tool calls: {[tc.get('name') for tc in message.get('tool_calls', [])]}")
                        elif message.get("type") == "tool":
                            print(f"Tool result: {message.get('content', '')}")
        print("-" * 30)


if __name__ == "__main__":
    test_mcp_tools()
