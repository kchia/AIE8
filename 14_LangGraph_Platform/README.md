<p align = "center" draggable=”false” ><img src="https://github.com/AI-Maker-Space/LLM-Dev-101/assets/37101144/d1343317-fa2f-41e1-8af1-1dbb18399719" 
     width="200px"
     height="auto"/>
</p>

## <h1 align="center" id="heading">Session 14: Build & Serve Agentic Graphs with LangGraph</h1>

| 🤓 Pre-work | 📰 Session Sheet | ⏺️ Recording | 🖼️ Slides | 👨‍💻 Repo | 📝 Homework | 📁 Feedback |
| :---------- | :--------------- | :----------- | :-------- | :------ | :---------- | :---------- |

# Build 🏗️

Run the repository and complete the following:

- 🤝 Breakout Room Part #1 — Building and serving your LangGraph Agent Graph

  - Task 1: Getting Dependencies & Environment
    - Configure `.env` (OpenAI, Tavily, optional LangSmith)
  - Task 2: Serve the Graph Locally
    - `uv run langgraph dev` (API on http://localhost:2024)
  - Task 3: Call the API from a different terminal
    - `uv run test_served_graph.py` (sync SDK example)
  - Task 4: Explore assistants (from `langgraph.json`)
    - `agent` → `simple_agent` (tool-using agent)
    - `agent_helpful` → `agent_with_helpfulness` (separate helpfulness node)

- 🤝 Breakout Room Part #2 — Using LangGraph Studio to visualize the graph
  - Task 1: Open Studio while the server is running
    - https://smith.langchain.com/studio?baseUrl=http://localhost:2024
  - Task 2: Visualize & Stream
    - Start a run and observe node-by-node updates
  - Task 3: Compare Flows
    - Contrast `agent` vs `agent_helpful` (tool calls vs helpfulness decision)

## Activities and Questions 🏗️ &❓

#### ❓ Question 1:

Compare the `agent` and `agent_helpful` assistants defined in `langgraph.json`. Where does the helpfulness evaluator fit in the graph, and under what condition should execution route back to the agent vs. terminate?

##### ✅ Answer:

**Comparison of `agent` and `agent_helpful`:**

1. **`agent` (simple_agent)**: A basic tool-using agent that:

   - Calls the chat model with tools
   - Routes to tool execution if tool calls are present
   - Terminates immediately after the agent responds (no helpfulness check)

2. **`agent_helpful` (agent_with_helpfulness)**: An enhanced agent that:
   - Includes the same basic agent functionality
   - Adds a helpfulness evaluation step after the agent responds
   - Implements a feedback loop for continuous improvement

**Where the helpfulness evaluator fits:**

The helpfulness evaluator (`helpfulness_node`) is positioned as a **post-response evaluation step** in the graph flow:

```
agent → [tool calls?] → action → agent
                ↓
         helpfulness ← (if no tool calls)
                ↓
         [helpful?] → continue/end
```

**Routing conditions:**

The execution routes back to the agent vs. terminates based on:

1. **Route back to agent** (`continue`):

   - When helpfulness evaluation returns "N" (not helpful)
   - The agent gets another chance to improve its response
   - This creates a feedback loop for iterative improvement

2. **Terminate** (`end`):
   - When helpfulness evaluation returns "Y" (helpful)
   - When loop limit is exceeded (>10 messages) - safety mechanism
   - This prevents infinite loops and ensures termination

**Key differences:**

- **Simple agent**: Linear flow (agent → tools → agent → end)
- **Helpful agent**: Cyclical flow with evaluation (agent → tools → agent → helpfulness → continue/end)

The helpfulness evaluator essentially acts as a **quality gate** that ensures the agent's response adequately addresses the user's query before terminating the conversation.

#### 🏗️ Activity #1 Debugging A Graph

Select the `agent_with_helpfulness` and set one or more interrupts (at least one `Before` and one `After`). Try changing values and continuing the turn.

#### ❓ Question 2:

What are your thoughts on when you would use a Before interrupt vs. an After interrupt?

##### ✅ Answer:

**Before interrupts vs After interrupts in LangGraph:**

**Before interrupts** are used when you want to:

- **Inspect and modify input data** before a node executes
- **Validate inputs** and potentially reject or modify them
- **Add preprocessing steps** or data transformation
- **Implement access control** or authentication checks
- **Debug and examine** what data is being passed to a node
- **Inject additional context** or modify the state before processing

**After interrupts** are used when you want to:

- **Inspect the output** of a node after it has completed execution
- **Validate results** and potentially trigger corrective actions
- **Add post-processing** or result transformation
- **Implement quality checks** on the generated content
- **Debug and examine** what a node produced
- **Trigger follow-up actions** based on the results

**Practical examples:**

1. **Before interrupt on `agent` node**:

   - Check if the user query contains sensitive information
   - Add system context or instructions before the agent processes
   - Validate that the query is appropriate for the agent

2. **After interrupt on `helpfulness` node**:

   - Log the helpfulness evaluation results
   - Trigger additional quality checks
   - Modify the decision before routing continues

3. **Before interrupt on `action` node**:

   - Validate tool parameters before execution
   - Add safety checks for tool usage
   - Modify tool calls if needed

4. **After interrupt on `agent` node**:
   - Check response quality before sending to helpfulness evaluation
   - Log agent responses for analysis
   - Trigger additional validation steps

**Key principle**: Use **Before** when you need to control or modify what goes into a node, and use **After** when you need to control or modify what comes out of a node.

<details>
<summary>🚧 Advanced Build 🚧 (OPTIONAL - <i>open this section for the requirements</i>)</summary>

- Create and deploy a locally hosted MCP server with FastMCP.
- Extend your tools in `tools.py` to allow your LangGraph to consume the MCP Server.
</details>

# Ship 🚢

- Running local server (`langgraph dev`)
- Short demo showing both assistants responding

# Share 🚀

- Walk through your graph in Studio
- Share 3 lessons learned and 3 lessons not learned

# Main Homework Assignment

Follow these steps to prepare and submit your homework assignment:

1. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s14-assignment`
2. Complete the Tasks listed in the Breakout Room sections of `Build 🏗️`
3. Complete the activities and questions in `Activities and Questions 🏗️ &❓` by editing the file and replacing "_(enter answer here)_" with your responses
4. Commit, and push your completed notebook to your `origin` repository. _NOTE: Do not merge it into your main branch._
5. Record a Loom video reviewing the content of your completed notebook
6. Make sure to include all of the following on your Homework Submission Form:
   - The GitHub URL to the `README.md` file _on your assignment branch (not main)_
   - The URL to your Loom Video
   - Your Three Lessons Learned/Not Yet Learned
   - The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_

### OPTIONAL: 🚧 Advanced Build Assignment 🚧

<details>
  <summary>(<i>Open this section for the submission instructions.</i>)</summary>

Follow these steps to prepare and submit your homework assignment:

1. Create a branch of your `AIE8` repo to track your changes. Example command: `git checkout -b s14-assignment`
2. Create your MCP server
3. Add it to the existing graph's tools
4. Deploy it **_locally_**
5. Validate the graph uses the MCP server's tools
6. Commit, and push your changes to your `origin` repository. _NOTE: Do not merge it into your main branch._
7. Record a Loom video reviewing the content of your completed notebook.
8. Make sure to include all of the following on your Homework Submission Form:
   - The GitHub URL to the notebook you created for the Advanced Build Assignment _on your assignment branch_
   - The URL to your Loom Video
   - Your Three Lessons Learned/Not Yet Learned
   - The URLs to any social media posts (LinkedIn, X, Discord, etc.) ⬅️ _easy Extra Credit points!_

</details>
