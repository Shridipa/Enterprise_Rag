web: uvicorn backend.main:app --host 0.0.0.0 --port $PORT
worker: celery -A backend.celery_app.celery_app worker --loglevel=info -Q ingestion -c 1
