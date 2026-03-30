from celery import Celery
from dotenv import load_dotenv
import os

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# ─── Celery instance ──────────────────────────────────────────────────────────
# broker  = Redis receives the task messages (the queue)
# backend = Redis stores task results so FastAPI can poll run status
celery_app = Celery(
    "arca_worker",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["worker.tasks"],   # auto-discover tasks on worker startup
)

celery_app.conf.update(
    # ── Serialization ────────────────────────────────────────────────────────
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # ── Reliability ──────────────────────────────────────────────────────────
    # ACK the task only AFTER it finishes, not when the worker receives it.
    # If the worker crashes mid-execution, the task goes back to the queue.
    task_acks_late=True,

    # If a worker dies with a task in-flight, put it back in the queue
    # instead of silently dropping it.
    task_reject_on_worker_lost=True,

    # ── Concurrency ──────────────────────────────────────────────────────────
    # One worker process per CPU core by default.
    # For I/O-heavy agent runs (LLM calls, tool calls) switch to:
    #   worker_pool = "gevent"  and  worker_concurrency = 50+
    worker_prefetch_multiplier=1,   # don't grab more tasks than you can handle

    # ── Result expiry ────────────────────────────────────────────────────────
    # Keep task results in Redis for 24h — long enough for polling, not forever.
    result_expires=86400,

    # ── Queues ───────────────────────────────────────────────────────────────
    # Two queues: default for normal runs, priority for webhook/schedule triggers.
    # Start with one queue, split when you have real traffic data.
    task_default_queue="runs",
    task_queues={
        "runs": {"exchange": "runs", "routing_key": "runs"},
    },

    # ── Timezone ─────────────────────────────────────────────────────────────
    timezone="UTC",
    enable_utc=True,
)