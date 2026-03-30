from sqlmodel import SQLModel, Field, Column
from sqlalchemy import Text
from datetime import datetime
from uuid import UUID, uuid4
from typing import Optional
from enum import Enum


# ─── Enums ────────────────────────────────────────────────────────────────────

class OrgRole(str, Enum):
    owner  = "owner"
    admin  = "admin"
    member = "member"

class SubscriptionStatus(str, Enum):
    active    = "active"
    cancelled = "cancelled"
    past_due  = "past_due"

class AgentStatus(str, Enum):
    draft    = "draft"
    active   = "active"
    archived = "archived"

class DeploymentStatus(str, Enum):
    pending = "pending"
    running = "running"
    stopped = "stopped"
    failed  = "failed"

class RunStatus(str, Enum):
    queued  = "queued"
    running = "running"
    done    = "done"
    failed  = "failed"

class MemoryType(str, Enum):
    episodic = "episodic"   # what happened in a specific run
    semantic = "semantic"   # distilled facts the agent learned over time

class AuthType(str, Enum):
    none   = "none"
    apikey = "apikey"
    oauth  = "oauth"


# ─── Domain 1: Identity ───────────────────────────────────────────────────────

class Users(SQLModel, table=True):
    __tablename__ = "users"

    id            : UUID     = Field(default_factory=uuid4, primary_key=True)
    email         : str      = Field(index=True, unique=True)
    name          : str      = Field(index=True)
    auth_provider : str      = Field(default="email")           # email / google / github
    created_at    : datetime = Field(default_factory=datetime.now)


class Organizations(SQLModel, table=True):
    __tablename__ = "organizations"

    id         : UUID     = Field(default_factory=uuid4, primary_key=True)
    owner_id   : UUID     = Field(index=True, foreign_key="users.id", ondelete="CASCADE")
    name       : str      = Field(index=True)
    slug       : str      = Field(index=True, unique=True)      # used in URLs: arca.io/acme
    created_at : datetime = Field(default_factory=datetime.now)


class OrgMembers(SQLModel, table=True):
    __tablename__ = "org_members"

    id      : UUID    = Field(default_factory=uuid4, primary_key=True)
    org_id  : UUID    = Field(index=True, foreign_key="organizations.id", ondelete="CASCADE")
    user_id : UUID    = Field(index=True, foreign_key="users.id", ondelete="CASCADE")
    role    : OrgRole = Field(default=OrgRole.member)


# ─── Domain 2: Billing ────────────────────────────────────────────────────────

class Plans(SQLModel, table=True):
    __tablename__ = "plans"

    id                 : UUID = Field(default_factory=uuid4, primary_key=True)
    name               : str  = Field(index=True, unique=True)  # free / pro / teams
    price_usd_cents    : int  = Field(default=0)                # 0 = free, 2000 = $20
    max_agents         : int  = Field(default=2)
    max_runs_per_month : int  = Field(default=100)


class Subscriptions(SQLModel, table=True):
    __tablename__ = "subscriptions"

    id           : UUID               = Field(default_factory=uuid4, primary_key=True)
    org_id       : UUID               = Field(index=True, foreign_key="organizations.id",ondelete="CASCADE")
    plan_id      : UUID               = Field(foreign_key="plans.id",ondelete="CASCADE")
    status       : SubscriptionStatus = Field(default=SubscriptionStatus.active)
    period_start : datetime           = Field(default_factory=datetime.now)
    period_end   : Optional[datetime] = Field(default=None)


class Credits(SQLModel, table=True):
    """One row per org. Balance is decremented atomically on every Run."""
    __tablename__ = "credits"

    id         : UUID     = Field(default_factory=uuid4, primary_key=True)
    org_id     : UUID     = Field(index=True, unique=True, foreign_key="organizations.id",ondelete="CASCADE")
    balance    : int      = Field(default=0)
    updated_at : datetime = Field(default_factory=datetime.now)


# ─── Domain 3: Agent Core ─────────────────────────────────────────────────────

class Agents(SQLModel, table=True):
    __tablename__ = "agents"

    id                 : UUID        = Field(default_factory=uuid4, primary_key=True)
    org_id             : UUID        = Field(index=True, foreign_key="organizations.id",ondelete="CASCADE")
    created_by         : UUID        = Field(foreign_key="users.id",ondelete="CASCADE")
    current_version_id : Optional[UUID] = Field(default=None, foreign_key="agentversions.id",ondelete="CASCADE")
    name               : str         = Field(index=True)
    slug               : str         = Field(index=True, unique=True)
    status             : AgentStatus = Field(default=AgentStatus.draft)
    created_at         : datetime    = Field(default_factory=datetime.now)


class AgentVersions(SQLModel, table=True):
    """
    Immutable/append-only. Never UPDATE a row here — always INSERT a new version.
    This gives you full rollback history and ensures every deployment
    points to a frozen config snapshot.
    """
    __tablename__ = "agentversions"

    id            : UUID     = Field(default_factory=uuid4, primary_key=True)
    agent_id      : UUID     = Field(index=True, foreign_key="agents.id",ondelete="CASCADE")
    created_by    : UUID     = Field(foreign_key="users.id",ondelete="CASCADE")
    version_num   : int      = Field(default=1)                 # auto-increment in app layer
    config_json   : str      = Field(sa_column=Column(Text))    # serialized agent.json
    builder_model : str      = Field(default="qwen2.5-coder")   # which model built this config
    created_at    : datetime = Field(default_factory=datetime.now)


# ─── Domain 4: Runtime ───────────────────────────────────────────────────────

class Deployments(SQLModel, table=True):
    """
    Tracks infra state — is the container alive, on which worker, since when.
    One deployment can serve thousands of Runs.
    """
    __tablename__ = "deployments"

    id           : UUID             = Field(default_factory=uuid4, primary_key=True)
    agent_id     : UUID             = Field(index=True, foreign_key="agents.id",ondelete="CASCADE")
    version_id   : UUID             = Field(foreign_key="agentversions.id",ondelete="CASCADE")
    status       : DeploymentStatus = Field(default=DeploymentStatus.pending)
    container_id : Optional[str]    = Field(default=None)       # Docker container ID
    worker_id    : Optional[str]    = Field(default=None)       # Celery worker hostname
    deployed_at  : datetime         = Field(default_factory=datetime.now)


class Runs(SQLModel, table=True):
    """
    One row per agent invocation. This is the billing unit — credits_used is
    decremented from Credits.balance inside a DB transaction on run completion.
    """
    __tablename__ = "runs"

    id            : UUID          = Field(default_factory=uuid4, primary_key=True)
    agent_id      : UUID          = Field(index=True, foreign_key="agents.id",ondelete="CASCADE")
    deployment_id : UUID          = Field(foreign_key="deployments.id",ondelete="CASCADE")
    triggered_by  : str           = Field(default="api")        # api / webhook / schedule
    status        : RunStatus     = Field(default=RunStatus.queued)
    input_json    : Optional[str] = Field(default=None, sa_column=Column(Text))
    output_json   : Optional[str] = Field(default=None, sa_column=Column(Text))
    tokens_used   : int           = Field(default=0)
    credits_used  : int           = Field(default=0)
    started_at    : datetime      = Field(default_factory=datetime.now)
    completed_at  : Optional[datetime] = Field(default=None)


# ─── Domain 5: Memory ─────────────────────────────────────────────────────────

class AgentMemories(SQLModel, table=True):
    """
    The moat. Every run can write memories here.
    The `embedding` column is a pgvector vector(1536) — enables semantic search
    across everything an agent has ever seen.

    To enable pgvector in Postgres:
        CREATE EXTENSION IF NOT EXISTS vector;
    The column is declared as Text here for SQLModel compatibility.
    Run this migration manually after table creation:
        ALTER TABLE agentmemories ADD COLUMN embedding vector(1536);
        CREATE INDEX ON agentmemories USING ivfflat (embedding vector_cosine_ops);
    """
    __tablename__ = "agentmemories"

    id          : UUID       = Field(default_factory=uuid4, primary_key=True)
    agent_id    : UUID       = Field(index=True, foreign_key="agents.id",ondelete="CASCADE")
    run_id      : UUID       = Field(foreign_key="runs.id",ondelete="CASCADE")
    content     : str        = Field(sa_column=Column(Text))
    memory_type : MemoryType = Field(default=MemoryType.episodic)
    metadata_json: Optional[str] = Field(default=None, sa_column=Column(Text))
    created_at  : datetime   = Field(default_factory=datetime.now)


# ─── Domain 6: Tools ──────────────────────────────────────────────────────────

class Tools(SQLModel, table=True):
    """
    MCP tool registry. is_public=True means any org can bind it (marketplace).
    is_public=False means private to the org that registered it.
    """
    __tablename__ = "tools"

    id          : UUID     = Field(default_factory=uuid4, primary_key=True)
    org_id      : UUID     = Field(index=True, foreign_key="organizations.id",ondelete="CASCADE")
    name        : str      = Field(index=True)
    mcp_url     : str      = Field()                            # MCP server endpoint
    auth_type   : AuthType = Field(default=AuthType.none)
    schema_json : Optional[str] = Field(default=None, sa_column=Column(Text))
    is_public   : bool     = Field(default=False)


class AgentToolBindings(SQLModel, table=True):
    """
    Junction table — resolves the many-to-many between Agents and Tools.
    One agent can bind many tools; one tool can be used by many agents.
    """
    __tablename__ = "agentbindings"

    id          : UUID           = Field(default_factory=uuid4, primary_key=True)
    agent_id    : UUID           = Field(index=True, foreign_key="agents.id",ondelete="CASCADE")
    tool_id     : UUID           = Field(foreign_key="tools.id",ondelete="CASCADE")
    config_json : Optional[str]  = Field(default=None, sa_column=Column(Text))
    enabled     : bool           = Field(default=True)


# ─── Domain 7: Security ───────────────────────────────────────────────────────

class ApiKeys(SQLModel, table=True):
    """
    Raw key is shown to the user ONCE on creation, then discarded.
    Only the SHA-256 hash is stored here — compared on every API request.
    revoked_at=None means the key is still active.
    """
    __tablename__ = "apikeys"

    id           : UUID              = Field(default_factory=uuid4, primary_key=True)
    org_id       : UUID              = Field(index=True, foreign_key="organizations.id",ondelete="CASCADE")
    created_by   : UUID              = Field(foreign_key="users.id",ondelete="CASCADE")
    key_hash     : str               = Field(unique=True, index=True)   # SHA-256 hex
    name         : str               = Field()                          # "Production key", "CI key"
    last_used_at : Optional[datetime] = Field(default=None)
    revoked_at   : Optional[datetime] = Field(default=None)
