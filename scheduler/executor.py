import json
import logging
from datetime import datetime
from models import Agents, AgentVersions

logger = logging.getLogger(__name__)


# ─── Executor ─────────────────────────────────────────────────────────────────
# This is the only file that changes in Month 2.
# Right now it's a hardcoded stub that:
#   - Parses the agent's config_json
#   - Echoes the input back as output
#   - Returns a fixed token/credit cost
#
# In Month 2 you replace _run_llm() with a real LangGraph graph
# that reads config_json and executes the agent's tools + LLM calls.
# The interface (inputs/outputs of execute_agent) stays identical —
# tasks.py never needs to change.

async def execute_agent(
    agent: Agents,
    version: AgentVersions,
    input_data: str | None,
) -> dict:
    """
    Execute an agent and return a result dict.

    Returns:
        {
            "output": str,          # JSON string written to runs.output_json
            "tokens_used": int,     # for observability
            "credits_used": int,    # deducted from org's credit balance
        }
    """
    logger.info(f"[executor] Running agent '{agent.name}' v{version.version_num}")

    # ── Parse config ─────────────────────────────────────────────────────────
    try:
        config = json.loads(version.config_json) if version.config_json else {}
    except json.JSONDecodeError:
        logger.warning("[executor] config_json is not valid JSON — using empty config")
        config = {}

    # ── Parse input ───────────────────────────────────────────────────────────
    try:
        input_payload = json.loads(input_data) if input_data else {}
    except json.JSONDecodeError:
        input_payload = {"raw": input_data}

    # ── Execute ───────────────────────────────────────────────────────────────
    # STUB: In Month 2 this calls _run_langgraph(config, input_payload)
    output = await _run_stub(agent_name=agent.name, config=config, input_payload=input_payload)

    return output


# ─── Stub execution ──────────────────────────────────────────────────────────
async def _run_stub(agent_name: str, config: dict, input_payload: dict) -> dict:
    """
    Hardcoded stub — validates the full run pipeline without any LLM.
    Useful for:
      - Testing the Celery → DB → Credits flow end-to-end
      - Load testing the infra layer before adding LLM latency
      - Confirming the executor contract works before Month 2
    """
    output = {
        "agent"      : agent_name,
        "status"     : "executed (stub)",
        "config_keys": list(config.keys()),         # shows what config was loaded
        "echo"       : input_payload,               # echoes the input back
        "executed_at": datetime.utcnow().isoformat(),
        "note"       : "This is a hardcoded stub. Replace _run_stub() with _run_langgraph() in Month 2.",
    }

    return {
        "output"      : json.dumps(output),
        "tokens_used" : 0,      # stub uses no real tokens
        "credits_used": 1,      # still deducts 1 credit to validate billing flow
    }


# ─── Month 2: swap this in ───────────────────────────────────────────────────
# async def _run_langgraph(config: dict, input_payload: dict) -> dict:
#     """
#     Real LLM execution using LangGraph.
#     config_json defines the graph: nodes, edges, tools, system prompt, model.
#     """
#     from langgraph.graph import StateGraph
#     from langchain_openai import ChatOpenAI
#
#     graph = build_graph_from_config(config)   # Month 2 builder layer
#     result = await graph.ainvoke(input_payload)
#
#     return {
#         "output"      : json.dumps(result["output"]),
#         "tokens_used" : result["usage"]["total_tokens"],
#         "credits_used": calculate_credits(result["usage"]),
#     }