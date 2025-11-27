"""FastAPI server for Deep Agents."""

import os
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agent import create_agent, SYSTEM_PROMPT


# Global checkpointer
checkpointer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global checkpointer
    
    # Initialize Postgres checkpointer if DATABASE_URL is set
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        try:
            checkpointer = AsyncPostgresSaver.from_conn_string(database_url)
            await checkpointer.setup()
            print("Connected to Postgres for checkpointing")
        except Exception as e:
            print(f"Warning: Could not connect to Postgres: {e}")
            checkpointer = None
    
    yield
    
    # Cleanup
    if checkpointer:
        await checkpointer.conn.close()


app = FastAPI(
    title="Deep Agents API",
    description="API for interacting with Deep Agents",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ThreadCreate(BaseModel):
    """Request to create a new thread."""
    metadata: dict[str, Any] | None = None


class ThreadResponse(BaseModel):
    """Response for thread operations."""
    thread_id: str
    created_at: str | None = None
    updated_at: str | None = None
    status: str = "idle"
    metadata: dict[str, Any] = {}
    values: dict[str, Any] = {}


class MessageInput(BaseModel):
    """Input message from user."""
    content: str
    role: str = "user"


class RunCreate(BaseModel):
    """Request to run the agent."""
    input: list[MessageInput]
    config: dict[str, Any] | None = None
    stream_mode: str | None = "messages"


class AssistantResponse(BaseModel):
    """Response for assistant info."""
    assistant_id: str
    name: str
    config: dict[str, Any] = {}


# In-memory thread storage (for demo, use Supabase in production)
threads: dict[str, dict] = {}


@app.get("/ok")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/info")
async def get_info():
    """Get server info."""
    return {
        "version": "0.1.0",
        "name": "Deep Agents API",
    }


@app.get("/assistants")
async def list_assistants():
    """List available assistants."""
    return [
        {
            "assistant_id": "agent",
            "name": "Deep Agent",
            "config": {},
            "metadata": {"description": "A powerful AI assistant with file and execution tools"},
        }
    ]


@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """Get assistant by ID."""
    if assistant_id != "agent":
        raise HTTPException(status_code=404, detail="Assistant not found")
    return {
        "assistant_id": "agent",
        "name": "Deep Agent",
        "config": {},
        "metadata": {"description": "A powerful AI assistant with file and execution tools"},
    }


@app.post("/threads")
async def create_thread(request: ThreadCreate | None = None):
    """Create a new thread."""
    thread_id = str(uuid.uuid4())
    thread = {
        "thread_id": thread_id,
        "status": "idle",
        "metadata": request.metadata if request and request.metadata else {},
        "values": {"messages": []},
    }
    threads[thread_id] = thread
    return thread


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get thread by ID."""
    if thread_id not in threads:
        # Create thread if it doesn't exist
        threads[thread_id] = {
            "thread_id": thread_id,
            "status": "idle",
            "metadata": {},
            "values": {"messages": []},
        }
    return threads[thread_id]


@app.post("/threads/search")
async def search_threads(
    limit: int = 10,
    offset: int = 0,
    metadata: dict[str, Any] | None = None,
):
    """Search threads."""
    all_threads = list(threads.values())
    return all_threads[offset:offset + limit]


@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: RunCreate):
    """Create a run for a thread."""
    # Ensure thread exists
    if thread_id not in threads:
        threads[thread_id] = {
            "thread_id": thread_id,
            "status": "idle",
            "metadata": {},
            "values": {"messages": []},
        }
    
    thread = threads[thread_id]
    
    # Build messages
    messages = []
    for msg in request.input:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            messages.append(AIMessage(content=msg.content))
        elif msg.role == "system":
            messages.append(SystemMessage(content=msg.content))
    
    # Create agent with checkpointer
    agent = create_agent()
    
    # Run configuration
    config = {"configurable": {"thread_id": thread_id}}
    if request.config:
        config.update(request.config)
    
    async def generate_stream() -> AsyncGenerator[str, None]:
        """Generate streaming response."""
        try:
            thread["status"] = "busy"
            
            async for event in agent.astream_events(
                {"messages": messages},
                config=config,
                version="v2",
            ):
                event_type = event.get("event", "")
                
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        yield f"data: {{'type': 'token', 'content': {repr(chunk.content)}}}\n\n"
                
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    yield f"data: {{'type': 'tool_start', 'name': {repr(tool_name)}, 'input': {repr(str(tool_input))}}}\n\n"
                
                elif event_type == "on_tool_end":
                    tool_output = event.get("data", {}).get("output", "")
                    yield f"data: {{'type': 'tool_end', 'output': {repr(str(tool_output)[:500])}}}\n\n"
            
            thread["status"] = "idle"
            yield "data: {'type': 'done'}\n\n"
            
        except Exception as e:
            thread["status"] = "error"
            yield f"data: {{'type': 'error', 'message': {repr(str(e))}}}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str):
    """Get the current state of a thread."""
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="Thread not found")
    return threads[thread_id]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
