# Deploying Cenomi Chatbot on Render

This document provides a step-by-step guide for deploying the Cenomi Chatbot application to Render.com using the Blueprint feature for one-click deployment of the entire application stack.

## Application Stack

- **FastAPI**: Backend API service
- **PostgreSQL**: Database for persistent storage
- **Redis**: Caching and queue management
- **LangChain/LangGraph**: AI orchestration framework
- **Pinecone**: Vector database for semantic search

## Prerequisites

- GitHub account with your code repository
- Render.com account
- Required API keys:
  - OpenAI API key or Google Gemini API key
  - Pinecone API key
  - ElevenLabs API key (for text-to-speech)
  - LangSmith API key (optional, for tracing and debugging)

## Files for Deployment

We've prepared several configuration files for deployment:

1. `render.yaml`: Defines all services to be deployed on Render
2. `build.sh`: Shell script for initializing the application
3. `Dockerfile`: Container definition for local testing
4. `docker-compose.local.yml`: For local testing before deployment

## Local Testing Before Deployment

Before deploying to Render, you can test the setup locally using Docker:

```bash
# Build and start all services locally
docker-compose -f docker-compose.local.yml up --build
```

Verify that all services are working correctly:

- FastAPI service at http://localhost:8000/docs
- PostgreSQL database connection
- Redis connection
- Basic API functionality

## Deployment Steps

### 1. Push Your Code to GitHub

Ensure all the configuration files (`render.yaml`, `build.sh`, etc.) are in your repository.

### 2. Deploy Using Render Blueprint

1. Log in to your Render account
2. Navigate to the "Blueprints" section
3. Click "New Blueprint Instance"
4. Connect your GitHub repository
5. Render will detect the `render.yaml` file and display the services to be created
6. Fill in the required environment variables:
   - `OPENAI_API_KEY` or `GEMINI_API_KEY`
   - `PINECONE_API_KEY`
   - `ELEVENLABS_API_KEY`
   - `LANGSMITH_API_KEY` (optional)
7. Click "Apply" to start the deployment process

### 3. Monitor Deployment Progress

Render will create and configure all services defined in the `render.yaml` file:

- PostgreSQL database
- Redis instance
- Web service (FastAPI application)

The deployment process may take several minutes. You can monitor the progress in the Render dashboard.

### 4. Verify Deployment

Once deployment is complete:

1. Access your web service at the URL provided by Render
2. Test the API endpoints using the Swagger UI at `/docs`
3. Check the logs for any errors or warnings

## Common Issues and Troubleshooting

### Services Not Connecting

If services aren't connecting properly:

- Check the environment variables in the Render dashboard
- Verify that internal service URLs are correctly set
- Review the logs for connection errors

### Cold Starts on Free Tier

Render's free tier services spin down after 15 minutes of inactivity:

- The first request after inactivity will be slow (30-60s)
- Subsequent requests will be faster
- For production use, consider upgrading to paid plans

### Database Migrations

If you need to run database migrations:

- Update the `build.sh` script with your migration commands
- Re-deploy the service

## Scaling Considerations

For production use:

- Upgrade from free tier to paid plans
- Consider adding more compute resources to the web service
- Use a larger PostgreSQL instance for better performance
- Configure autoscaling for the web service

## Monitoring and Maintenance

- Set up health check notifications in Render
- Monitor logs regularly
- Set up usage alerts for your API keys
- Consider integrating with an external monitoring service

---

By following this guide, you should be able to successfully deploy the Cenomi Chatbot application to Render.com and get it running with minimal configuration.
