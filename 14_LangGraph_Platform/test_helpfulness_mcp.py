#!/usr/bin/env python3
"""Test MCP tools with the helpfulness agent."""

from langgraph_sdk import get_sync_client


def test_helpfulness_agent_with_mcp():
    """Test MCP tools with the agent_helpful assistant."""
    client = get_sync_client(url="http://localhost:2024")
    
    print("Testing MCP tools with Agent with Helpfulness Check")
    print("=" * 60)
    
    for chunk in client.runs.stream(
        None,  # Threadless run
        "agent_with_helpfulness",  # Graph id from langgraph.json
        input={
            "messages": [
                {
                    "role": "human",
                    "content": "Calculate 12 times 7 and reverse the word 'Python'",
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
                            print(f"🔧 Agent called tools: {tool_names}")
                        elif message.get("type") == "tool":
                            print(f"⚙️  Tool result: {message.get('content', '')}")
                        elif message.get("type") == "ai" and not message.get("tool_calls"):
                            content = message.get('content', '')
                            if "HELPFULNESS:" in content:
                                print(f"📊 Helpfulness check: {content}")
                            else:
                                print(f"💬 Final response: {content}")


if __name__ == "__main__":
    test_helpfulness_agent_with_mcp()
