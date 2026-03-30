import asyncio
import logging
from datetime import datetime
from uuid import UUID

from sqlmodel import select
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

from worker.celery_app import celery_app
from worker.executor import execute_agent
from models import Runs, Agents, AgentVersions, Credits, RunStatus
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)

# ─── DB setup for the worker process ─────────────────────────────────────────
# Celery workers are separate processes — they have their own DB connection,
# independent from FastAPI's. Same config, separate engine instance.
postgres_password = os.getenv("POSTGRES_PASSWORD")
DATABASE_URL = f"postgresql+asyncpg://postgres:{postgres_password}@localhost:5432/ARCA_db"

engine = create_async_engine(DATABASE_URL, echo=False)
SessionFactory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


# ─── Helper: run async code inside a sync Celery task ────────────────────────
# Celery tasks are synchronous by default. Our DB calls are async (asyncpg).
# asyncio.run() spins up a fresh event loop for each task execution.
# This is safe because each Celery worker process handles one task at a time.
def run_async(coro):
    return asyncio.run(coro)


# ─── Core task ────────────────────────────────────────────────────────────────
@celery_app.task(
    name="worker.tasks.execute_run",
    bind=True,              # gives access to `self` for retries
    max_retries=3,
    default_retry_delay=10, # seconds between retries
)
def execute_run(self, run_id: str):
    """
    Entry point for every agent run.

    Flow:
        1. Load Run + Agent + AgentVersion from DB
        2. Mark Run as running
        3. Call executor.execute_agent() — hardcoded stub for now, LLM later
        4. Write output back to Run row
        5. Decrement Credits.balance atomically
        6. Mark Run as done / failed

    The whole DB write in step 4+5 is a single transaction —
    if credits deduction fails, the run result is also rolled back.
    """
    logger.info(f"[task] Starting run {run_id}")
    run_async(_execute_run_async(self, run_id))


async def _execute_run_async(task, run_id: str):
    async with SessionFactory() as session:

        # ── 1. Load the Run row ───────────────────────────────────────────
        run = await session.get(Runs, UUID(run_id))
        if not run:
            logger.error(f"[task] Run {run_id} not found — dropping task")
            return

        # ── 2. Load Agent + current AgentVersion ─────────────────────────
        agent = await session.get(Agents, run.agent_id)
        if not agent:
            await _fail_run(session, run, error="Agent not found")
            return

        version = await session.get(AgentVersions, agent.current_version_id)
        if not version:
            await _fail_run(session, run, error="No active version found for agent")
            return

        # ── 3. Check credit balance before running ───────────────────────
        credits_stmt = select(Credits).where(Credits.org_id == agent.org_id)
        result = await session.exec(credits_stmt)
        credits = result.first()

        if not credits or credits.balance <= 0:
            await _fail_run(session, run, error="Insufficient credits")
            return

        # ── 4. Mark as running ────────────────────────────────────────────
        run.status = RunStatus.running
        run.started_at = datetime.utcnow()
        session.add(run)
        await session.commit()

        # ── 5. Execute ────────────────────────────────────────────────────
        try:
            result = await execute_agent(
                agent=agent,
                version=version,
                input_data=run.input_json,
            )

            # ── 6. Write result + deduct credits (one transaction) ────────
            async with session.begin():
                run.status       = RunStatus.done
                run.output_json  = result["output"]
                run.tokens_used  = result["tokens_used"]
                run.credits_used = result["credits_used"]
                run.completed_at = datetime.utcnow()
                session.add(run)

                # Atomic credit deduction
                credits.balance    -= result["credits_used"]
                credits.updated_at  = datetime.utcnow()
                session.add(credits)

            logger.info(f"[task] Run {run_id} completed — credits used: {result['credits_used']}")

        except Exception as exc:
            logger.exception(f"[task] Run {run_id} failed: {exc}")

            # Retry with exponential backoff if transient error
            try:
                raise task.retry(exc=exc, countdown=2 ** task.request.retries)
            except task.MaxRetriesExceededError:
                await _fail_run(session, run, error=str(exc))


# ─── Helper: mark a run as failed ─────────────────────────────────────────────
async def _fail_run(session: AsyncSession, run: Runs, error: str):
    run.status       = RunStatus.failed
    run.output_json  = f'{{"error": "{error}"}}'
    run.completed_at = datetime.utcnow()
    session.add(run)
    await session.commit()
    logger.warning(f"[task] Run {run.id} marked as failed: {error}")