#!/bin/bash
set -e

# Set environment variables for Redis connection
export REDIS_HOST=$(echo $RENDER_REDIS_INTERNAL_URI | cut -d':' -f1)
export REDIS_PORT=$(echo $RENDER_REDIS_INTERNAL_URI | cut -d':' -f2)

# Wait for PostgreSQL database to be ready
echo "Waiting for PostgreSQL..."
sleep 5

# Run your application
echo "Starting FastAPI application..."
uvicorn src.api:app --host 0.0.0.0 --port $PORT 