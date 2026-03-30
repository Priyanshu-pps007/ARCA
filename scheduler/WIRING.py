# ─── docker-compose.yml (add Redis service) ───────────────────────────────────
#
# services:
#   postgres:
#     image: pgvector/pgvector:pg16
#     environment:
#       POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
#       POSTGRES_DB: ARCA_db
#     ports:
#       - "5432:5432"
#     volumes:
#       - postgres_data:/var/lib/postgresql/data
#
#   redis:
#     image: redis:7-alpine
#     ports:
#       - "6379:6379"
#     volumes:
#       - redis_data:/data
#     command: redis-server --appendonly yes   # persist tasks across restarts
#
# volumes:
#   postgres_data:
#   redis_data:


# ─── .env additions ───────────────────────────────────────────────────────────
#
# REDIS_URL=redis://localhost:6379/0


# ─── How to start the worker ─────────────────────────────────────────────────
#
# Terminal 1 — FastAPI
#   uvicorn main:app --reload
#
# Terminal 2 — Celery worker
#   celery -A worker.celery_app worker --loglevel=info -Q runs
#
# Terminal 3 — Celery flower (optional dashboard at localhost:5555)
#   celery -A worker.celery_app flower


# ─── routers/runs.py — how FastAPI triggers a Celery task ────────────────────

from fastapi import APIRouter, Depends, HTTPException
from sqlmodel.ext.asyncio.session import AsyncSession
from config import get_session
from models import Runs, Agents, RunStatus
from worker.tasks import execute_run
from uuid import UUID
import json

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("/")
async def trigger_run(
    agent_id: str,
    input_data: dict,
    session: AsyncSession = Depends(get_session),
):
    """
    1. Validate the agent exists
    2. Create a Run row (status=queued)
    3. Enqueue a Celery task with the run_id
    4. Return immediately — don't wait for the agent to finish
    """
    # Validate agent
    agent = await session.get(Agents, UUID(agent_id))
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")

    if agent.status != "active":
        raise HTTPException(status_code=400, detail=f"Agent is {agent.status}, not active")

    # Create Run row
    run = Runs(
        agent_id      = agent.id,
        deployment_id = agent.current_version_id,  # update when Deployments are live
        triggered_by  = "api",
        status        = RunStatus.queued,
        input_json    = json.dumps(input_data),
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)

    # Fire Celery task — non-blocking, returns immediately
    execute_run.delay(str(run.id))

    return {
        "run_id" : str(run.id),
        "status" : run.status,
        "message": "Run queued — poll GET /runs/{run_id} for status",
    }


@router.get("/{run_id}")
async def get_run_status(
    run_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Poll this endpoint to check if a run is done."""
    run = await session.get(Runs, UUID(run_id))
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")

    return {
        "run_id"      : str(run.id),
        "status"      : run.status,
        "tokens_used" : run.tokens_used,
        "credits_used": run.credits_used,
        "output"      : run.output_json,
        "started_at"  : run.started_at,
        "completed_at": run.completed_at,
    }