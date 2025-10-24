#!/usr/bin/env python3
"""Test script to validate all MCP tools integration with LangGraph."""

from langgraph_sdk import get_sync_client


def test_all_mcp_tools():
    """Test all MCP tools are available and working."""
    client = get_sync_client(url="http://localhost:2024")
    
    test_cases = [
        {
            "name": "Add Numbers Tool",
            "query": "What is 25 plus 17? Use the add_numbers tool.",
            "expected_tool": "add_numbers"
        },
        {
            "name": "Reverse Text Tool", 
            "query": "Reverse the text 'Hello World' using the reverse_text tool.",
            "expected_tool": "reverse_text"
        },
        {
            "name": "Multiply Numbers Tool",
            "query": "What is 8 times 9? Use the multiply_numbers tool.",
            "expected_tool": "multiply_numbers"
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{i}. Testing {test_case['name']}")
        print("=" * 50)
        print(f"Query: {test_case['query']}")
        print("-" * 30)
        
        for chunk in client.runs.stream(
            None,  # Threadless run
            "simple_agent",  # Assistant id from langgraph.json
            input={
                "messages": [
                    {
                        "role": "human",
                        "content": test_case["query"],
                    }
                ]
            },
            stream_mode="updates",
        ):
            if chunk.event == "updates":
                for node_name, node_data in chunk.data.items():
                    if "messages" in node_data:
                        for message in node_data["messages"]:
                            if message.get("type") == "ai" and message.get("tool_calls"):
                                tool_names = [tc.get('name') for tc in message.get('tool_calls', [])]
                                if test_case["expected_tool"] in tool_names:
                                    print(f"✅ Successfully called {test_case['expected_tool']} tool")
                                else:
                                    print(f"❌ Expected {test_case['expected_tool']}, got: {tool_names}")
                            elif message.get("type") == "tool":
                                print(f"Tool result: {message.get('content', '')}")
                            elif message.get("type") == "ai" and not message.get("tool_calls"):
                                print(f"Final response: {message.get('content', '')}")


if __name__ == "__main__":
    test_all_mcp_tools()
