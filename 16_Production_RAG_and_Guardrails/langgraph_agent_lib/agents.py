"""LangGraph agent integration with production features."""

from typing import Dict, Any, List, Optional
import os
import logging

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.tools.arxiv.tool import ArxivQueryRun
from langchain_core.tools import tool
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages

from .models import get_openai_model
from .rag import ProductionRAGChain

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State schema for agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]


class GuardrailsAgentState(TypedDict, total=False):
    """State schema for guardrails-enabled agent graphs."""
    messages: Annotated[List[BaseMessage], add_messages]
    validation_results: Optional[Dict[str, Any]]
    refinement_count: int


def create_rag_tool(rag_chain: ProductionRAGChain):
    """Create a RAG tool from a ProductionRAGChain."""
    
    @tool
    def retrieve_information(query: str) -> str:
        """Use Retrieval Augmented Generation to retrieve information from the student loan documents."""
        try:
            result = rag_chain.invoke(query)
            return result.content if hasattr(result, 'content') else str(result)
        except Exception as e:
            return f"Error retrieving information: {str(e)}"
    
    return retrieve_information


def get_default_tools(rag_chain: Optional[ProductionRAGChain] = None) -> List:
    """Get default tools for the agent.
    
    Args:
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        List of tools
    """
    tools = []
    
    # Add Tavily search if API key is available
    if os.getenv("TAVILY_API_KEY"):
        tools.append(TavilySearchResults(max_results=5))
    
    # Add Arxiv tool
    tools.append(ArxivQueryRun())
    
    # Add RAG tool if provided
    if rag_chain:
        tools.append(create_rag_tool(rag_chain))
    
    return tools


def create_langgraph_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None
):
    """Create a simple LangGraph agent.
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        
    Returns:
        Compiled LangGraph agent
    """
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    def call_model(state: AgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    def should_continue(state: AgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return END
    
    # Build graph
    graph = StateGraph(AgentState)
    tool_node = ToolNode(tools)
    
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {"action": "action", END: END})
    graph.add_edge("action", "agent")
    
    return graph.compile()


def create_guardrails_agent(
    model_name: str = "gpt-4",
    temperature: float = 0.1,
    tools: Optional[List] = None,
    rag_chain: Optional[ProductionRAGChain] = None,
    input_guard: Optional[Any] = None,
    output_guard: Optional[Any] = None,
    max_refinements: int = 2,
    strict_mode: bool = True
):
    """Create a LangGraph agent with Guardrails validation.
    
    This agent includes:
    - Input validation (jailbreak, topic, PII detection)
    - Output validation (content moderation, factuality)
    - Refinement loops for failed validations
    - Graceful error handling
    
    Args:
        model_name: OpenAI model name
        temperature: Model temperature
        tools: List of tools to bind to the model
        rag_chain: Optional RAG chain to include as a tool
        input_guard: Guardrails Guard instance for input validation
        output_guard: Guardrails Guard instance for output validation
        max_refinements: Maximum number of refinement attempts for failed validations
        strict_mode: If True, raises exceptions on validation failure. If False, returns error messages.
        
    Returns:
        Compiled LangGraph agent with guardrails
    """
    # Import guardrails utilities
    try:
        from .guardrails import validate_input, validate_output
    except ImportError:
        logger.warning("Guardrails module not available. Creating agent without guardrails.")
        return create_langgraph_agent(model_name, temperature, tools, rag_chain)
    
    if tools is None:
        tools = get_default_tools(rag_chain)
    
    # Get model and bind tools
    model = get_openai_model(model_name=model_name, temperature=temperature)
    model_with_tools = model.bind_tools(tools)
    
    # Initialize state defaults
    def initialize_state(state: GuardrailsAgentState) -> GuardrailsAgentState:
        """Initialize state with defaults if needed."""
        if "validation_results" not in state:
            state["validation_results"] = {}
        if "refinement_count" not in state:
            state["refinement_count"] = 0
        return state
    
    # Input validation node
    def validate_user_input(state: GuardrailsAgentState) -> Dict[str, Any]:
        """Validate user input before processing."""
        messages = state.get("messages", [])
        validation_results = state.get("validation_results", {})
        refinement_count = state.get("refinement_count", 0)
        
        if not messages:
            return {"validation_results": validation_results}
        
        last_message = messages[-1]
        
        # Only validate HumanMessage inputs
        if isinstance(last_message, HumanMessage) and input_guard:
            try:
                user_input = last_message.content
                logger.info(f"Validating user input: {user_input[:100]}...")
                
                result = validate_input(
                    input_guard,
                    user_input,
                    raise_on_failure=False  # Don't raise, we'll handle routing
                )
                
                validation_results["input"] = {
                    "passed": result["validation_passed"],
                    "validated_output": result.get("validated_output", user_input),
                    "error": result.get("error")
                }
                
                if not result["validation_passed"]:
                    logger.warning(f"Input validation failed: {result.get('error')}")
                    return {
                        "validation_results": validation_results,
                        "refinement_count": refinement_count
                    }
                else:
                    logger.info("Input validation passed")
                    
            except Exception as e:
                logger.error(f"Input validation error: {e}", exc_info=True)
                validation_results["input"] = {
                    "passed": False,
                    "error": str(e)
                }
                if strict_mode:
                    raise
        
        return {"validation_results": validation_results}
    
    # Output validation node
    def validate_agent_output(state: GuardrailsAgentState) -> Dict[str, Any]:
        """Validate agent output before returning to user."""
        messages = state.get("messages", [])
        validation_results = state.get("validation_results", {})
        refinement_count = state.get("refinement_count", 0)
        
        if not messages:
            return {"validation_results": validation_results}
        
        last_message = messages[-1]
        
        # Only validate AIMessage outputs (final responses)
        if isinstance(last_message, AIMessage) and output_guard:
            # Skip if it has tool calls (not final response yet)
            if hasattr(last_message, "tool_calls") and last_message.tool_calls:
                return {"validation_results": validation_results}
            
            try:
                agent_response = last_message.content
                logger.info(f"Validating agent output: {agent_response[:100]}...")
                
                # For factuality checks, try to extract context and original prompt from messages
                context = None
                original_prompt = None
                if len(messages) > 1:
                    # Try to find the original user query
                    for msg in reversed(messages):
                        if isinstance(msg, HumanMessage):
                            original_prompt = msg.content
                            # Get context from RAG if available
                            if rag_chain:
                                try:
                                    rag_result = rag_chain.invoke(msg.content)
                                    context = rag_result.content if hasattr(rag_result, 'content') else str(rag_result)
                                except:
                                    pass
                            break
                
                result = validate_output(
                    output_guard,
                    agent_response,
                    context=context,
                    original_prompt=original_prompt,
                    raise_on_failure=False  # Don't raise, we'll handle refinement
                )
                
                validation_results["output"] = {
                    "passed": result["validation_passed"],
                    "validated_output": result.get("validated_output", agent_response),
                    "error": result.get("error")
                }
                
                if not result["validation_passed"]:
                    logger.warning(f"Output validation failed: {result.get('error')}")
                else:
                    logger.info("Output validation passed")
                    
            except Exception as e:
                logger.error(f"Output validation error: {e}", exc_info=True)
                validation_results["output"] = {
                    "passed": False,
                    "error": str(e)
                }
                if strict_mode and refinement_count >= max_refinements:
                    raise
        
        return {"validation_results": validation_results}
    
    # Agent node (call model)
    def call_model(state: GuardrailsAgentState) -> Dict[str, Any]:
        """Invoke the model with messages."""
        messages = state["messages"]
        
        # Check if input validation failed
        validation_results = state.get("validation_results", {})
        input_validation = validation_results.get("input", {})
        
        if not input_validation.get("passed", True):
            # Input validation failed - create error response
            error_msg = input_validation.get("error", "Input validation failed")
            if strict_mode:
                raise RuntimeError(f"Input validation failed: {error_msg}")
            else:
                error_response = AIMessage(
                    content=f"I'm sorry, but I cannot process that request. {error_msg}. "
                           f"Please rephrase your question about student loans or financial aid."
                )
                return {"messages": [error_response]}
        
        # Normal agent processing
        response = model_with_tools.invoke(messages)
        return {"messages": [response]}
    
    # Refinement node (re-generate response if validation failed)
    def refine_response(state: GuardrailsAgentState) -> Dict[str, Any]:
        """Refine the agent response if validation failed."""
        messages = state.get("messages", [])
        validation_results = state.get("validation_results", {})
        refinement_count = state.get("refinement_count", 0)
        
        if refinement_count >= max_refinements:
            logger.warning(f"Max refinements ({max_refinements}) reached. Returning last response.")
            return {"refinement_count": refinement_count}
        
        output_validation = validation_results.get("output", {})
        
        if not output_validation.get("passed", True):
            # Add system message asking for refinement
            refinement_prompt = f"""The previous response failed validation: {output_validation.get('error', 'Unknown error')}. 
Please provide a revised response that:
1. Stays on-topic (student loans, financial aid, education financing)
2. Is factual and based on the provided context
3. Uses professional language
4. Avoids any inappropriate content"""
            
            refinement_messages = messages + [
                SystemMessage(content=refinement_prompt),
                HumanMessage(content="Please provide a revised response to the original question.")
            ]
            
            response = model_with_tools.invoke(refinement_messages)
            return {
                "messages": [response],
                "refinement_count": refinement_count + 1
            }
        
        return {"refinement_count": refinement_count}
    
    # Routing functions
    def should_continue(state: GuardrailsAgentState):
        """Route to tools if the last message has tool calls."""
        last_message = state["messages"][-1]
        if getattr(last_message, "tool_calls", None):
            return "action"
        return "validate_output"
    
    def should_refine(state: GuardrailsAgentState):
        """Determine if output should be refined."""
        validation_results = state.get("validation_results", {})
        output_validation = validation_results.get("output", {})
        refinement_count = state.get("refinement_count", 0)
        
        if not output_validation.get("passed", True) and refinement_count < max_refinements:
            return "refine"
        return END
    
    def check_input_validation(state: GuardrailsAgentState):
        """Check if input validation passed."""
        validation_results = state.get("validation_results", {})
        input_validation = validation_results.get("input", {})
        
        # If no input guard or no validation result, proceed to agent
        if not input_guard or "input" not in validation_results:
            return "agent"
        
        if input_validation.get("passed", True):
            return "agent"
        else:
            # Input validation failed
            if strict_mode:
                return "error"
            else:
                return END  # Will return error message from call_model
    
    # Build graph
    graph = StateGraph(GuardrailsAgentState)
    tool_node = ToolNode(tools)
    
    # Add nodes
    graph.add_node("initialize", initialize_state)
    graph.add_node("validate_input", validate_user_input)
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("validate_output", validate_agent_output)
    graph.add_node("refine", refine_response)
    
    # Add error handler for strict mode
    if strict_mode:
        def error_handler(state: GuardrailsAgentState) -> Dict[str, Any]:
            """Handle validation errors."""
            validation_results = state.get("validation_results", {})
            error_msg = "Validation failed"
            if "input" in validation_results:
                error_msg = validation_results["input"].get("error", error_msg)
            elif "output" in validation_results:
                error_msg = validation_results["output"].get("error", error_msg)
            
            error_response = AIMessage(
                content=f"I'm sorry, but I cannot process that request due to validation errors: {error_msg}"
            )
            return {"messages": [error_response]}
        
        graph.add_node("error", error_handler)
        graph.add_edge("error", END)
    
    # Set entry point and edges
    graph.set_entry_point("initialize")
    graph.add_edge("initialize", "validate_input")
    if strict_mode:
        graph.add_conditional_edges(
            "validate_input",
            check_input_validation,
            {
                "agent": "agent",
                "error": "error",
                END: END
            }
        )
    else:
        graph.add_conditional_edges(
            "validate_input",
            check_input_validation,
            {
                "agent": "agent",
                END: END
            }
        )
    graph.add_conditional_edges("agent", should_continue, {"action": "action", "validate_output": "validate_output"})
    graph.add_edge("action", "agent")
    graph.add_conditional_edges("validate_output", should_refine, {"refine": "refine", END: END})
    graph.add_edge("refine", "validate_output")
    
    return graph.compile()
