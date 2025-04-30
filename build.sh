#!/bin/bash
set -e

# No need to set Redis environment variables as they will be provided directly

# Wait for database connections to be ready
echo "Waiting for external database connections..."
sleep 5

# Run your application
echo "Starting FastAPI application..."
uvicorn src.api:app --host 0.0.0.0 --port $PORT 