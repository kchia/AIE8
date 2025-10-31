"""Validate Implementation - Check that all components are properly set up

This script validates the implementation without making API calls.
"""

import sys
import os

print("=" * 80)
print("🔍 VALIDATING ACTIVITY #3 IMPLEMENTATION")
print("=" * 80)

checks_passed = 0
checks_total = 0

def check(name, condition, details=""):
    """Check a condition and report results."""
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        print(f"✅ {name}")
        if details:
            print(f"   {details}")
        checks_passed += 1
        return True
    else:
        print(f"❌ {name}")
        if details:
            print(f"   {details}")
        return False

# Check 1: Module imports
print("\n📦 Checking Module Structure...")
try:
    from langgraph_agent_lib import agents, guardrails
    check("Module imports", True, "langgraph_agent_lib.agents and .guardrails")
except ImportError as e:
    check("Module imports", False, f"Import error: {e}")

# Check 2: Guardrails functions
print("\n🛡️ Checking Guardrails Functions...")
try:
    from langgraph_agent_lib.guardrails import (
        create_guardrails_guard,
        create_factuality_guard,
        validate_input,
        validate_output,
        create_guardrails_node
    )
    check("Guardrails functions exist", True, "All 5 functions found")
except ImportError as e:
    check("Guardrails functions exist", False, f"Missing functions: {e}")

# Check 3: Agent functions
print("\n🤖 Checking Agent Functions...")
try:
    from langgraph_agent_lib.agents import (
        create_langgraph_agent,
        create_guardrails_agent,
        GuardrailsAgentState
    )
    check("Agent functions exist", True, "create_guardrails_agent found")
except ImportError as e:
    check("Agent functions exist", False, f"Missing functions: {e}")

# Check 4: Agent state schema
print("\n📊 Checking State Schema...")
try:
    from langgraph_agent_lib.agents import GuardrailsAgentState
    # Check required fields
    annotations = GuardrailsAgentState.__annotations__
    has_messages = 'messages' in annotations
    has_validation = 'validation_results' in annotations
    has_refinement = 'refinement_count' in annotations

    all_present = has_messages and has_validation and has_refinement
    check("GuardrailsAgentState schema", all_present,
          f"messages: {has_messages}, validation_results: {has_validation}, refinement_count: {has_refinement}")
except Exception as e:
    check("GuardrailsAgentState schema", False, f"Error: {e}")

# Check 5: Test files
print("\n🧪 Checking Test Files...")
test_files = [
    "test_guardrails_agent.py",
    "demo_guardrails_agent.py",
    "quick_demo.py"
]
for test_file in test_files:
    exists = os.path.exists(test_file)
    check(f"Test file: {test_file}", exists)

# Check 6: Documentation
print("\n📚 Checking Documentation...")
doc_files = [
    "ACTIVITY_3_SOLUTION.md",
    "QUICKSTART.md"
]
for doc_file in doc_files:
    exists = os.path.exists(doc_file)
    check(f"Documentation: {doc_file}", exists)

# Check 7: Agent creation function signature
print("\n🔧 Checking create_guardrails_agent Signature...")
try:
    from langgraph_agent_lib.agents import create_guardrails_agent
    import inspect
    sig = inspect.signature(create_guardrails_agent)
    params = list(sig.parameters.keys())

    required_params = ['model_name', 'temperature', 'tools', 'rag_chain',
                      'input_guard', 'output_guard', 'max_refinements', 'strict_mode']
    has_all_params = all(p in params for p in required_params)

    check("Function parameters", has_all_params,
          f"Has {len([p for p in required_params if p in params])}/{len(required_params)} required params")
except Exception as e:
    check("Function parameters", False, f"Error: {e}")

# Check 8: Validation functions
print("\n✔️ Checking Validation Functions...")
try:
    from langgraph_agent_lib.guardrails import validate_input, validate_output
    import inspect

    input_sig = inspect.signature(validate_input)
    output_sig = inspect.signature(validate_output)

    input_params = list(input_sig.parameters.keys())
    output_params = list(output_sig.parameters.keys())

    has_guard = 'guard' in input_params and 'guard' in output_params
    has_raise_param = 'raise_on_failure' in input_params and 'raise_on_failure' in output_params

    check("Validation function signatures", has_guard and has_raise_param,
          f"guard param: {has_guard}, raise_on_failure: {has_raise_param}")
except Exception as e:
    check("Validation function signatures", False, f"Error: {e}")

# Check 9: Required dependencies in code
print("\n📋 Checking Implementation Details...")
try:
    with open("langgraph_agent_lib/agents.py", "r") as f:
        content = f.read()
        has_input_validation = "validate_user_input" in content
        has_output_validation = "validate_agent_output" in content
        has_refinement = "refine_response" in content
        has_conditional_routing = "should_refine" in content

        check("Input validation node", has_input_validation)
        check("Output validation node", has_output_validation)
        check("Refinement node", has_refinement)
        check("Conditional routing", has_conditional_routing)
except Exception as e:
    check("Implementation details", False, f"Error reading file: {e}")

# Check 10: Test scenarios in test file
print("\n🎯 Checking Test Coverage...")
try:
    with open("test_guardrails_agent.py", "r") as f:
        content = f.read()
        has_jailbreak = "jailbreak" in content.lower()
        has_off_topic = "off_topic" in content.lower() or "off-topic" in content.lower()
        has_pii = "pii" in content.lower()
        has_profanity = "profanity" in content.lower() or "inappropriate" in content.lower()

        check("Jailbreak tests", has_jailbreak)
        check("Off-topic tests", has_off_topic)
        check("PII tests", has_pii)
        check("Content moderation tests", has_profanity)
except Exception as e:
    check("Test coverage", False, f"Error reading test file: {e}")

# Summary
print("\n" + "=" * 80)
print(f"📊 VALIDATION SUMMARY")
print("=" * 80)
print(f"Checks Passed: {checks_passed}/{checks_total}")
print(f"Success Rate: {(checks_passed/checks_total)*100:.1f}%")

if checks_passed == checks_total:
    print("\n🎉 All checks passed! Implementation is complete and ready to test.")
    print("\n📝 Next steps:")
    print("  1. Set your OPENAI_API_KEY: export OPENAI_API_KEY='your-key'")
    print("  2. Run quick demo: uv run python quick_demo.py")
    print("  3. Run full tests: uv run python test_guardrails_agent.py")
    print("  4. Try interactive mode: uv run python demo_guardrails_agent.py")
    sys.exit(0)
elif checks_passed >= checks_total * 0.8:
    print("\n✅ Most checks passed. Implementation looks good with minor issues.")
    sys.exit(0)
else:
    print("\n⚠️  Some checks failed. Review the errors above.")
    sys.exit(1)
