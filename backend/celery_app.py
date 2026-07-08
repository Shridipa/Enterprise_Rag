"""
celery_app.py — Celery application factory.
Redis serves as message broker (task queue) and result backend.
"""
from celery import Celery
from kombu import Queue

from .config import settings

celery_app = Celery(
    "enterprise_rag",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["backend.tasks.ingest"],
)

# Task routing configuration
celery_app.conf.task_routes = {
    "tasks.ingest.ingest_document": {"queue": "ingestion", "priority": 5},
}

celery_app.conf.update(
    # Serialization
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],

    # Timezone
    timezone="UTC",
    enable_utc=True,

    # Reliability — redeliver task if worker dies mid-execution
    task_acks_late=True,
    task_reject_on_worker_lost=True,

    # Prevent workers from grabbing too many tasks ahead of time
    worker_prefetch_multiplier=1,

    # Expose STARTED state so the API can report progress
    task_track_started=True,

    # Keep task results for 24 h
    result_expires=86400,

    # Retry configuration
    task_retry_backoff=True,
    task_retry_backoff_max=600,  # Max 10 minutes
    task_retry_jitter=True,

    # Worker configuration
    worker_max_tasks_per_child=1000,  # Restart worker after 1000 tasks to prevent memory leaks
)

# Define queues
celery_app.conf.task_queues = (
    Queue("ingestion", routing_key="ingestion"),
    Queue("default", routing_key="default"),
)