"""FastAPI server for Deep Agents - LangGraph API Compatible."""

import json
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent import create_agent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    yield


app = FastAPI(
    title="Deep Agents API",
    description="LangGraph-compatible API for Deep Agents",
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


# In-memory thread storage
threads: dict[str, dict] = {}


def get_timestamp():
    """Get current ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


@app.get("/ok")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/info")
async def get_info():
    """Get server info."""
    return {"version": "0.1.0"}


# ============ Assistants API ============

@app.get("/assistants")
async def list_assistants():
    """List available assistants."""
    return [
        {
            "assistant_id": "agent",
            "graph_id": "agent",
            "name": "Deep Agent",
            "config": {},
            "metadata": {},
            "created_at": get_timestamp(),
            "updated_at": get_timestamp(),
        }
    ]


@app.post("/assistants/search")
async def search_assistants(request: Request):
    """Search assistants."""
    return [
        {
            "assistant_id": "agent",
            "graph_id": "agent",
            "name": "Deep Agent",
            "config": {},
            "metadata": {},
            "created_at": get_timestamp(),
            "updated_at": get_timestamp(),
        }
    ]


@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """Get assistant by ID."""
    if assistant_id != "agent":
        raise HTTPException(status_code=404, detail="Assistant not found")
    return {
        "assistant_id": "agent",
        "graph_id": "agent",
        "name": "Deep Agent",
        "config": {},
        "metadata": {},
        "created_at": get_timestamp(),
        "updated_at": get_timestamp(),
    }


# ============ Threads API ============

@app.post("/threads")
async def create_thread(request: Request):
    """Create a new thread."""
    body = {}
    try:
        body = await request.json()
    except:
        pass
    
    thread_id = str(uuid.uuid4())
    now = get_timestamp()
    thread = {
        "thread_id": thread_id,
        "status": "idle",
        "created_at": now,
        "updated_at": now,
        "metadata": body.get("metadata", {}),
        "values": {"messages": [], "todos": [], "files": {}},
    }
    threads[thread_id] = thread
    return thread


@app.get("/threads/{thread_id}")
async def get_thread(thread_id: str):
    """Get thread by ID."""
    if thread_id not in threads:
        now = get_timestamp()
        threads[thread_id] = {
            "thread_id": thread_id,
            "status": "idle",
            "created_at": now,
            "updated_at": now,
            "metadata": {},
            "values": {"messages": [], "todos": [], "files": {}},
        }
    return threads[thread_id]


@app.post("/threads/search")
async def search_threads(request: Request):
    """Search threads."""
    body = {}
    try:
        body = await request.json()
    except:
        pass
    
    limit = body.get("limit", 10)
    offset = body.get("offset", 0)
    all_threads = list(threads.values())
    return all_threads[offset:offset + limit]


@app.patch("/threads/{thread_id}")
async def update_thread(thread_id: str, request: Request):
    """Update thread metadata."""
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    body = await request.json()
    thread = threads[thread_id]
    if "metadata" in body:
        thread["metadata"].update(body["metadata"])
    thread["updated_at"] = get_timestamp()
    return thread


# ============ Thread State API ============

@app.get("/threads/{thread_id}/state")
async def get_thread_state(thread_id: str):
    """Get the current state of a thread."""
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    thread = threads[thread_id]
    return {
        "values": thread.get("values", {}),
        "next": [],
        "checkpoint": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": str(uuid.uuid4()),
        },
        "metadata": thread.get("metadata", {}),
        "created_at": thread.get("created_at"),
        "parent_checkpoint": None,
    }


@app.post("/threads/{thread_id}/state")
async def update_thread_state(thread_id: str, request: Request):
    """Update thread state."""
    if thread_id not in threads:
        raise HTTPException(status_code=404, detail="Thread not found")
    
    body = await request.json()
    thread = threads[thread_id]
    
    if "values" in body:
        thread["values"].update(body["values"])
    thread["updated_at"] = get_timestamp()
    
    return {"checkpoint": {"thread_id": thread_id}}


# ============ Runs API ============

@app.post("/threads/{thread_id}/runs")
async def create_run(thread_id: str, request: Request):
    """Create a run (non-streaming)."""
    body = await request.json()
    
    if thread_id not in threads:
        now = get_timestamp()
        threads[thread_id] = {
            "thread_id": thread_id,
            "status": "idle",
            "created_at": now,
            "updated_at": now,
            "metadata": {},
            "values": {"messages": [], "todos": [], "files": {}},
        }
    
    run_id = str(uuid.uuid4())
    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "status": "pending",
        "created_at": get_timestamp(),
    }


@app.post("/threads/{thread_id}/runs/stream")
async def create_run_stream(thread_id: str, request: Request):
    """Create a streaming run - main endpoint for chat."""
    body = await request.json()
    
    # Ensure thread exists
    if thread_id not in threads:
        now = get_timestamp()
        threads[thread_id] = {
            "thread_id": thread_id,
            "status": "idle",
            "created_at": now,
            "updated_at": now,
            "metadata": {"assistant_id": body.get("assistant_id", "agent")},
            "values": {"messages": [], "todos": [], "files": {}},
        }
    
    thread = threads[thread_id]
    input_data = body.get("input", {})
    config = body.get("config", {})
    stream_mode = body.get("stream_mode", ["messages"])
    
    # Extract messages from input
    input_messages = input_data.get("messages", [])
    
    async def generate_stream() -> AsyncGenerator[bytes, None]:
        """Generate SSE stream in LangGraph format."""
        run_id = str(uuid.uuid4())
        
        try:
            thread["status"] = "busy"
            
            # Send metadata event
            metadata_event = {
                "run_id": run_id,
                "thread_id": thread_id,
                "assistant_id": body.get("assistant_id", "agent"),
            }
            yield f"event: metadata\ndata: {json.dumps(metadata_event)}\n\n".encode()
            
            # Build LangChain messages
            messages = []
            for msg in input_messages:
                content = msg.get("content", "")
                msg_type = msg.get("type", "human")
                if msg_type == "human":
                    messages.append(HumanMessage(content=content))
                elif msg_type == "ai":
                    messages.append(AIMessage(content=content))
            
            if not messages:
                # No input, just return current state
                values_event = {"messages": thread["values"].get("messages", [])}
                yield f"event: values\ndata: {json.dumps(values_event)}\n\n".encode()
                yield f"event: end\ndata: null\n\n".encode()
                thread["status"] = "idle"
                return
            
            # Create and run agent
            agent = create_agent()
            run_config = {"configurable": {"thread_id": thread_id}}
            if config:
                run_config["configurable"].update(config.get("configurable", {}))
            
            # Collect full response
            full_response = ""
            tool_calls = []
            
            async for event in agent.astream_events(
                {"messages": messages},
                config=run_config,
                version="v2",
            ):
                event_type = event.get("event", "")
                
                if event_type == "on_chat_model_stream":
                    chunk = event.get("data", {}).get("chunk")
                    if chunk and hasattr(chunk, "content") and chunk.content:
                        content = chunk.content
                        if isinstance(content, str):
                            full_response += content
                            # Send messages/partial event
                            msg_event = [{
                                "type": "ai",
                                "content": content,
                            }]
                            yield f"event: messages/partial\ndata: {json.dumps(msg_event)}\n\n".encode()
                
                elif event_type == "on_tool_start":
                    tool_name = event.get("name", "")
                    tool_input = event.get("data", {}).get("input", {})
                    tool_calls.append({
                        "name": tool_name,
                        "args": tool_input,
                    })
                
                elif event_type == "on_tool_end":
                    pass  # Tool results handled internally
            
            # Store the response in thread
            if full_response:
                ai_message = {
                    "id": str(uuid.uuid4()),
                    "type": "ai",
                    "content": full_response,
                }
                if tool_calls:
                    ai_message["tool_calls"] = tool_calls
                
                # Add user message and AI response to thread
                for msg in input_messages:
                    thread["values"]["messages"].append({
                        "id": msg.get("id", str(uuid.uuid4())),
                        "type": msg.get("type", "human"),
                        "content": msg.get("content", ""),
                    })
                thread["values"]["messages"].append(ai_message)
            
            # Send final values
            values_event = thread["values"]
            yield f"event: values\ndata: {json.dumps(values_event)}\n\n".encode()
            
            # Send end event
            yield f"event: end\ndata: null\n\n".encode()
            
            thread["status"] = "idle"
            thread["updated_at"] = get_timestamp()
            
        except Exception as e:
            thread["status"] = "error"
            error_event = {"message": str(e)}
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n".encode()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/threads/{thread_id}/runs/{run_id}")
async def get_run(thread_id: str, run_id: str):
    """Get run status."""
    return {
        "run_id": run_id,
        "thread_id": thread_id,
        "status": "success",
        "created_at": get_timestamp(),
    }


@app.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """Get thread history."""
    if thread_id not in threads:
        return []
    
    thread = threads[thread_id]
    return [{
        "checkpoint": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": str(uuid.uuid4()),
        },
        "values": thread.get("values", {}),
        "metadata": thread.get("metadata", {}),
        "created_at": thread.get("created_at"),
    }]


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
