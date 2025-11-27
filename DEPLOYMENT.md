# Deployment Guide

This guide covers deploying the Deep Agents application with:
- **Backend**: Render (LangGraph API Server)
- **Frontend**: Vercel (Next.js UI)
- **Database**: Supabase (PostgreSQL)

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Vercel        │     │   Render        │     │   Supabase      │
│   (Frontend)    │────▶│   (Backend)     │────▶│   (Database)    │
│   Next.js UI    │     │   LangGraph API │     │   PostgreSQL    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
                        ┌─────────────────┐
                        │   Anthropic     │
                        │   Claude API    │
                        └─────────────────┘
```

## Prerequisites

1. **Anthropic API Key**: Get one from [Anthropic Console](https://console.anthropic.com/)
2. **LangSmith API Key**: Get one from [LangSmith](https://smith.langchain.com/settings)
3. **Supabase Project**: Create at [Supabase](https://supabase.com/)
4. **Render Account**: Sign up at [Render](https://render.com/)
5. **Vercel Account**: Sign up at [Vercel](https://vercel.com/)

## Step 1: Supabase Setup

The database schema has already been created via migrations. Your Supabase project should have:

### Tables
- `users` - User profiles
- `threads` - Conversation threads
- `messages` - Chat messages
- `agent_files` - Files created by the agent
- `checkpoints` - LangGraph state checkpoints
- `checkpoint_blobs` - Checkpoint binary data
- `checkpoint_writes` - Checkpoint write operations

### Get Your Credentials

1. Go to your Supabase project dashboard
2. Navigate to **Settings > API**
3. Copy:
   - **Project URL**: `https://gbplvxvvshumapgmscrk.supabase.co`
   - **anon public key**: For frontend
   - **service_role key**: For backend (keep secret!)

4. Navigate to **Settings > Database**
5. Copy the **Connection string** (URI format) for the backend

## Step 2: Render Backend Deployment

### Option A: Using Render Dashboard

1. Go to [Render Dashboard](https://dashboard.render.com/)
2. Click **New > Web Service**
3. Connect your GitHub repo: `HarambeeAI/axl8-deepagents`
4. Configure:
   - **Name**: `worryless-deepagents`
   - **Root Directory**: `backend`
   - **Runtime**: Docker
   - **Dockerfile Path**: `./backend/Dockerfile`
   - **Docker Context**: `.` (root)

### Option B: Using render.yaml Blueprint

The `render.yaml` file in the repo root defines the service configuration.

### Environment Variables (Required)

Set these in Render Dashboard > Environment:

| Variable | Description |
|----------|-------------|
| `ANTHROPIC_API_KEY` | Your Anthropic API key |
| `LANGSMITH_API_KEY` | Your LangSmith API key |
| `LANGCHAIN_TRACING_V2` | `true` |
| `LANGCHAIN_PROJECT` | `worryless-deepagents` |
| `SUPABASE_URL` | `https://gbplvxvvshumapgmscrk.supabase.co` |
| `SUPABASE_ANON_KEY` | Your Supabase anon key |
| `SUPABASE_SERVICE_ROLE_KEY` | Your Supabase service role key |
| `DATABASE_URL` | Supabase Postgres connection string |

### Verify Deployment

After deployment, your backend will be available at:
```
https://worryless-deepagents.onrender.com
```

Test the health endpoint:
```bash
curl https://worryless-deepagents.onrender.com/ok
```

## Step 3: Vercel Frontend Deployment

### Automatic Deployment

Since Vercel is connected to the GitHub repo, it will auto-deploy on every push to `master`.

### Configure Root Directory

In Vercel project settings:
- **Root Directory**: `deep-agents-ui`
- **Framework Preset**: Next.js
- **Build Command**: `yarn build`
- **Install Command**: `yarn install`

### Environment Variables

Set these in Vercel Dashboard > Settings > Environment Variables:

| Variable | Description |
|----------|-------------|
| `NEXT_PUBLIC_DEPLOYMENT_URL` | `https://worryless-deepagents.onrender.com` |
| `NEXT_PUBLIC_ASSISTANT_ID` | `agent` |
| `NEXT_PUBLIC_LANGSMITH_API_KEY` | Your LangSmith API key |
| `NEXT_PUBLIC_SUPABASE_URL` | `https://gbplvxvvshumapgmscrk.supabase.co` |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Your Supabase anon key |

## Step 4: Verify Full Stack

1. Open your Vercel deployment URL
2. The UI should connect to the Render backend
3. Start a new conversation
4. The agent should respond using Claude

## Local Development

### Backend

```bash
cd backend
cp .env.example .env
# Edit .env with your API keys

# Install dependencies
pip install -e ../libs/deepagents
pip install -r requirements.txt

# Run locally
langgraph dev
```

### Frontend

```bash
cd deep-agents-ui
cp .env.example .env.local
# Edit .env.local with your settings

yarn install
yarn dev
```

Open http://localhost:3000

## Troubleshooting

### Backend not responding
- Check Render logs for errors
- Verify `ANTHROPIC_API_KEY` is set correctly
- Ensure the Docker build completed successfully

### Frontend can't connect to backend
- Check CORS settings
- Verify `NEXT_PUBLIC_DEPLOYMENT_URL` is correct
- Check browser console for errors

### Database connection issues
- Verify `DATABASE_URL` format
- Check Supabase connection pooler settings
- Ensure IP is allowed in Supabase

## URLs

- **Frontend (Vercel)**: Your Vercel deployment URL
- **Backend (Render)**: https://worryless-deepagents.onrender.com
- **Supabase**: https://gbplvxvvshumapgmscrk.supabase.co
- **LangSmith Tracing**: https://smith.langchain.com/
