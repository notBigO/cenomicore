# Deploying Cenomi Chatbot on Render

This document outlines the steps to deploy the Cenomi Chatbot application on Render.com using the free tier.

## Prerequisites

- A GitHub repository with your code
- A Render.com account
- API keys for OpenAI/Gemini, Pinecone, and ElevenLabs

## Deployment Steps

### 1. Fork or Clone the Repository

Make sure your code is pushed to a GitHub repository that Render can access.

### 2. Set Up Render Blueprint

We're using Render Blueprints to deploy all the required services at once:

1. Go to the Render Dashboard
2. Click on "Blueprints" in the sidebar
3. Click "New Blueprint Instance"
4. Connect your GitHub repository
5. Render will automatically detect the `render.yaml` file and show you the services it will create

### 3. Configure Environment Variables

During the Blueprint setup, you'll need to provide values for these environment variables:

- `OPENAI_API_KEY` - Your OpenAI API key
- `GEMINI_API_KEY` - Your Google Gemini API key (if using Gemini)
- `PINECONE_API_KEY` - Your Pinecone API key for vector searches
- `ELEVENLABS_API_KEY` - Your ElevenLabs API key for text-to-speech
- `LANGSMITH_API_KEY` - Your LangSmith API key (optional)

### 4. Deploy the Blueprint

Click "Create Blueprint Instance" to deploy all services:

- **Web Service**: FastAPI application
- **PostgreSQL Database**: For storing conversation history and app data
- **Redis**: For caching and background tasks

### 5. Verification

After the deployment completes (may take a few minutes):

1. Go to your Web Service in the Render dashboard
2. Click on the URL to access your application
3. Verify that the API endpoints are working correctly

## Troubleshooting

If you encounter any issues:

1. Check the logs in the Render dashboard for each service
2. Ensure all API keys are correctly set
3. Verify the PostgreSQL and Redis connections
4. If needed, you can rebuild the web service by clicking "Manual Deploy" → "Clear build cache & deploy"

## Free Tier Limitations

Be aware of Render's free tier limitations:

- Web services on the free plan will spin down after 15 minutes of inactivity
- When a free service spins down, the next request will take a while to respond as the service spins back up
- PostgreSQL databases on the free tier have limited storage (1GB)
- Redis instances on the free tier have memory limitations

For production use, consider upgrading to paid plans for better performance and reliability.
