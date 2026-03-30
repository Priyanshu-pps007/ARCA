"""
agent_schema.py

Why this exists:
    agent.json is the core artifact of ARCA — the config that the builder
    generates and the executor runs. This file defines:

    1. Pydantic models for every field (validation + type safety)
    2. JSON Schema export (for builder prompt + CLI validation)
    3. A load/dump interface so executor.py can deserialize cleanly

How it maps to LangGraph:
    AgentConfig.graph.nodes   → graph.add_node() calls
    AgentConfig.graph.edges   → graph.add_edge() / add_conditional_edges()
    AgentConfig.graph.state   → TypedDict State class fields
    AgentConfig.memory        → MemorySaver checkpointer
    AgentConfig.tools         → ToolNode bindings
    AgentConfig.model         → ChatOpenAI / ChatAnthropic etc.

Design rules for this schema:
    1. Every field must be serializable to JSON (no Python objects)
    2. Every field the executor reads must have a default (safe deserialization)
    3. The graph block must be expressive enough for 80% of real agent patterns:
          linear chain, router, tool-calling loop, human-in-the-loop
    4. What the builder generates must be what the executor can run — no gap
"""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, model_validator


# ═══════════════════════════════════════════════════════════════════════════════
# Enums — closed vocabularies the builder must pick from
# ═══════════════════════════════════════════════════════════════════════════════

class ModelProvider(str, Enum):
    openai    = "openai"
    anthropic = "anthropic"
    bedrock   = "bedrock"
    deepseek  = "deepseek"
    qwen      = "qwen"
    groq      = "groq"


class MemoryType(str, Enum):
    episodic = "episodic"   # remembers what happened in past runs
    semantic = "semantic"   # distilled facts, survives across resets
    both     = "both"


class NodeType(str, Enum):
    llm    = "llm"      # calls an LLM with a prompt
    tool   = "tool"     # executes one or more tools (ToolNode)
    router = "router"   # conditional branching — routes to next node
    human  = "human"    # interrupt() — waits for human input
    start  = "start"    # entry point marker (virtual, not executed)
    end    = "end"      # terminal node marker (virtual, not executed)


class EdgeType(str, Enum):
    static      = "static"       # always goes A → B
    conditional = "conditional"  # goes to different nodes based on output


class RetryStrategy(str, Enum):
    none        = "none"
    fixed       = "fixed"
    exponential = "exponential"


# ═══════════════════════════════════════════════════════════════════════════════
# Model config
# ═══════════════════════════════════════════════════════════════════════════════

class ModelConfig(BaseModel):
    """Which LLM to use and how to call it."""

    provider    : ModelProvider = ModelProvider.openai
    name        : str           = Field(
        default     = "gpt-4o-mini",
        description = "Exact model string: gpt-4o-mini, claude-3-5-haiku, etc."
    )
    temperature : float         = Field(default=0.3, ge=0.0, le=2.0)
    max_tokens  : int           = Field(default=1024, ge=1, le=128000)
    streaming   : bool          = Field(
        default     = True,
        description = "Stream tokens back in real time via SSE"
    )

    class Config:
        use_enum_values = True


# ═══════════════════════════════════════════════════════════════════════════════
# Memory config
# ═══════════════════════════════════════════════════════════════════════════════

class MemoryConfig(BaseModel):
    """Controls LangGraph checkpointing and long-term memory writes."""

    enabled      : bool       = False
    type         : MemoryType = MemoryType.episodic
    window_size  : int        = Field(
        default     = 10,
        description = "How many past messages to include in context window"
    )
    persist_runs : bool       = Field(
        default     = True,
        description = "Write run outputs to AgentMemories table"
    )

    class Config:
        use_enum_values = True


# ═══════════════════════════════════════════════════════════════════════════════
# Tool binding
# ═══════════════════════════════════════════════════════════════════════════════

class ToolBinding(BaseModel):
    """
    Reference to a tool from the MCP registry.
    The executor resolves these to actual tool functions at runtime.
    """
    name        : str            = Field(description="Must match Tools.name in DB")
    enabled     : bool           = True
    config      : dict[str, Any] = Field(
        default_factory = dict,
        description     = "Tool-specific config: API keys, base URLs, etc."
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Graph: State schema
# ═══════════════════════════════════════════════════════════════════════════════

class StateField(BaseModel):
    """
    One field in the LangGraph State TypedDict.
    The executor reconstructs a TypedDict class from these at runtime.
    """
    name         : str  = Field(description="Field name, e.g. 'messages'")
    type         : str  = Field(
        description = "Python type string: 'list[str]', 'str', 'dict', 'int'"
    )
    is_message_list : bool = Field(
        default     = False,
        description = "If True, uses add_messages reducer (append, not overwrite)"
    )
    default      : Any  = None


# ═══════════════════════════════════════════════════════════════════════════════
# Graph: Nodes
# ═══════════════════════════════════════════════════════════════════════════════

class NodeConfig(BaseModel):
    """
    One node in the LangGraph StateGraph.

    Maps to:
        type=llm    → a function that calls the LLM with system_prompt
        type=tool   → ToolNode(tools=[...]) for the bound tools
        type=router → a function that returns the next node name
        type=human  → interrupt() call — pauses for human input
    """
    id            : str             = Field(description="Unique node ID, used in edges")
    type          : NodeType
    description   : str             = Field(
        description = "What this node does — used in builder prompt and UI"
    )

    # ── LLM node fields ──────────────────────────────────────────────────────
    system_prompt : Optional[str]   = Field(
        default     = None,
        description = "System prompt for this node. Overrides top-level system_prompt."
    )
    model_override: Optional[ModelConfig] = Field(
        default     = None,
        description = "Use a different model for this node (e.g. cheap model for routing)"
    )

    # ── Tool node fields ──────────────────────────────────────────────────────
    tools         : list[str]       = Field(
        default_factory = list,
        description     = "Tool names this node can call. Must be in top-level tools[]."
    )

    # ── Router node fields ────────────────────────────────────────────────────
    router_prompt : Optional[str]   = Field(
        default     = None,
        description = "Prompt that produces a routing decision (one of the edge targets)"
    )

    # ── Human node fields ─────────────────────────────────────────────────────
    interrupt     : bool            = Field(
        default     = False,
        description = "If True, executor calls interrupt() before this node — HITL"
    )
    interrupt_prompt: Optional[str] = Field(
        default     = None,
        description = "Message shown to human when interrupted"
    )

    # ── Retry policy ──────────────────────────────────────────────────────────
    retry         : RetryStrategy   = RetryStrategy.none
    max_retries   : int             = Field(default=3, ge=0, le=10)

    class Config:
        use_enum_values = True


# ═══════════════════════════════════════════════════════════════════════════════
# Graph: Edges
# ═══════════════════════════════════════════════════════════════════════════════

class ConditionalRoute(BaseModel):
    """One branch in a conditional edge."""
    condition : str = Field(
        description = "Python expression evaluated against state, e.g. 'state[\"next\"] == \"tools\"'"
    )
    target    : str = Field(description="Node ID to route to if condition is True")


class EdgeConfig(BaseModel):
    """
    One edge in the LangGraph StateGraph.

    static edge      → graph.add_edge(from, to)
    conditional edge → graph.add_conditional_edges(from, router_fn, routes)
    """
    from_node  : str                    = Field(alias="from")
    to_node    : Optional[str]          = Field(
        default = None,
        alias   = "to",
        description = "Target node for static edges"
    )
    type       : EdgeType               = EdgeType.static
    routes     : list[ConditionalRoute] = Field(
        default_factory = list,
        description     = "Conditional branches. Required when type=conditional."
    )
    default    : Optional[str]          = Field(
        default     = None,
        description = "Fallback node if no condition matches"
    )

    class Config:
        populate_by_name = True
        use_enum_values  = True

    @model_validator(mode="after")
    def validate_edge(self) -> EdgeConfig:
        if self.type == EdgeType.static and not self.to_node:
            raise ValueError("Static edges must have a 'to' field")
        if self.type == EdgeType.conditional and not self.routes:
            raise ValueError("Conditional edges must have at least one route")
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Graph block
# ═══════════════════════════════════════════════════════════════════════════════

class GraphConfig(BaseModel):
    """
    Full LangGraph StateGraph definition in JSON.
    The executor reconstructs a runnable graph from this at runtime.
    """
    state_schema  : list[StateField] = Field(
        default_factory = lambda: [
            StateField(
                name            = "messages",
                type            = "list",
                is_message_list = True,
            )
        ],
        description = "Fields in the LangGraph State TypedDict"
    )
    nodes         : list[NodeConfig]
    edges         : list[EdgeConfig]
    entry_point   : str = Field(description="ID of the first node to execute")

    @model_validator(mode="after")
    def validate_graph(self) -> GraphConfig:
        node_ids = {n.id for n in self.nodes}

        # entry_point must exist
        if self.entry_point not in node_ids:
            raise ValueError(f"entry_point '{self.entry_point}' not in nodes")

        # all edge sources and targets must exist
        valid_targets = node_ids | {"END"}   # END is a LangGraph special
        for edge in self.edges:
            if edge.from_node not in node_ids:
                raise ValueError(f"Edge from unknown node '{edge.from_node}'")
            if edge.to_node and edge.to_node not in valid_targets:
                raise ValueError(f"Edge to unknown node '{edge.to_node}'")
            for route in edge.routes:
                if route.target not in valid_targets:
                    raise ValueError(f"Route target '{route.target}' unknown")

        return self


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level AgentConfig — the full agent.json
# ═══════════════════════════════════════════════════════════════════════════════

class AgentConfig(BaseModel):
    """
    The complete agent.json schema.

    This is what the builder generates, the executor runs,
    the CLI validates, and the fine-tuning dataset is built from.

    Version field: bump this when you make breaking changes to the schema.
    The executor checks this before loading.
    """
    version       : str          = Field(default="1.0", description="Schema version")
    name          : str          = Field(description="Agent name, used as slug")
    description   : str          = Field(description="What this agent does")
    system_prompt : str          = Field(
        description = "Top-level system prompt. Node-level prompts override this."
    )

    model         : ModelConfig  = Field(default_factory=ModelConfig)
    memory        : MemoryConfig = Field(default_factory=MemoryConfig)
    tools         : list[ToolBinding] = Field(default_factory=list)
    graph         : GraphConfig

    # ── Observability ─────────────────────────────────────────────────────────
    tags          : list[str]    = Field(
        default_factory = list,
        description     = "Free-form tags for search/filter in the dashboard"
    )
    langsmith_project: Optional[str] = Field(
        default     = None,
        description = "LangSmith project name for tracing. None = no tracing."
    )

    class Config:
        use_enum_values = True

    # ── Serialization helpers ─────────────────────────────────────────────────

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string — written to AgentVersions.config_json."""
        return self.model_dump_json(indent=indent, by_alias=True)

    def to_dict(self) -> dict:
        return self.model_dump(by_alias=True)

    @classmethod
    def from_json(cls, json_str: str) -> AgentConfig:
        """Deserialize from AgentVersions.config_json — called by executor."""
        return cls.model_validate_json(json_str)

    @classmethod
    def from_file(cls, path: str | Path) -> AgentConfig:
        """Load from a .json file — useful for CLI and testing."""
        return cls.model_validate_json(Path(path).read_text())

    def validate_tools(self, registered_tools: list[str]) -> list[str]:
        """
        Check that all tool names in graph nodes exist in the top-level
        tools list AND in the registered tool registry.
        Returns list of errors (empty = valid).
        """
        errors        = []
        bound_names   = {t.name for t in self.tools}
        enabled_names = {t.name for t in self.tools if t.enabled}

        for node in self.graph.nodes:
            for tool_name in node.tools:
                if tool_name not in bound_names:
                    errors.append(
                        f"Node '{node.id}' references tool '{tool_name}' "
                        f"not in top-level tools[]"
                    )
                if tool_name not in registered_tools:
                    errors.append(
                        f"Tool '{tool_name}' not found in MCP registry"
                    )

        return errors