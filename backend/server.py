"""FastAPI server for Deep Agents - Full LangGraph API Compatible.

This server implements the LangGraph API specification to work with the
deep-agents-ui frontend. It properly streams:
- messages (AI responses with tool calls)
- todos (planning state)
- files (filesystem state) - synced from sandbox after each tool
- updates (per-node execution updates)
"""

import json
import os
import uuid
import asyncio
import sys
import httpx
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    ToolMessage,
    BaseMessage,
)

from agent import create_agent
from sandbox import LocalSandbox, ModalSandbox, get_sandbox_backend
from skills import (
    get_skills_client,
    SKILLS_SYSTEM_PROMPT,
    is_document_generation_request,
    SkillType,
)

# Disable LangSmith tracing to avoid 403 errors
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"

# Force unbuffered output for real-time logging
sys.stdout.reconfigure(line_buffering=True)


# Global sandbox instance (shared across requests for persistence)
# Will be either LocalSandbox or ModalSandbox depending on environment
_sandbox = None


def get_sandbox():
    """Get or create the global sandbox instance.
    
    Auto-detects sandbox type based on environment:
    - If MODAL_TOKEN_ID and MODAL_TOKEN_SECRET are set, uses Modal
    - Otherwise, uses local subprocess sandbox
    """
    global _sandbox
    if _sandbox is None:
        _sandbox = get_sandbox_backend()
    return _sandbox


def sync_sandbox_files_to_state(sandbox, thread_state: dict) -> bool:
    """Sync files from sandbox filesystem to thread state.
    
    Works with both LocalSandbox and ModalSandbox.
    Returns True if files were updated.
    """
    updated = False
    
    # For LocalSandbox, we can walk the filesystem directly
    if isinstance(sandbox, LocalSandbox):
        workspace = sandbox.working_dir
        
        # Walk the workspace and read all files
        for root, dirs, files in os.walk(workspace):
            # Skip hidden directories
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for filename in files:
                if filename.startswith('.'):
                    continue
                    
                full_path = Path(root) / filename
                rel_path = "/" + str(full_path.relative_to(workspace))
                
                try:
                    # Only read text files under 100KB
                    if full_path.stat().st_size > 100000:
                        continue
                        
                    content = full_path.read_text(errors='ignore')
                    
                    # Check if file is new or changed
                    if rel_path not in thread_state.get("files", {}) or \
                       thread_state["files"].get(rel_path) != content:
                        if "files" not in thread_state:
                            thread_state["files"] = {}
                        thread_state["files"][rel_path] = content
                        updated = True
                except Exception:
                    pass
    
    # For ModalSandbox, use ls + download_files
    elif isinstance(sandbox, ModalSandbox):
        try:
            # List files in workspace
            result = sandbox.execute("find /workspace -type f -size -100k 2>/dev/null | head -50")
            if result.exit_code == 0 and result.output:
                file_paths = [p.strip() for p in result.output.strip().split('\n') if p.strip()]
                
                # Filter out hidden files
                file_paths = [p for p in file_paths if not any(
                    part.startswith('.') for part in p.split('/')
                )]
                
                if file_paths:
                    # Download files
                    responses = sandbox.download_files(file_paths)
                    
                    for resp in responses:
                        if resp.error is None and resp.content is not None:
                            # Convert path to relative
                            rel_path = resp.path.replace("/workspace", "") or "/"
                            if not rel_path.startswith("/"):
                                rel_path = "/" + rel_path
                            
                            try:
                                content = resp.content.decode('utf-8', errors='ignore')
                                
                                if rel_path not in thread_state.get("files", {}) or \
                                   thread_state["files"].get(rel_path) != content:
                                    if "files" not in thread_state:
                                        thread_state["files"] = {}
                                    thread_state["files"][rel_path] = content
                                    updated = True
                            except Exception:
                                pass
        except Exception as e:
            print(f"[Modal] Error syncing files: {e}")
    
    return updated


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup: initialize sandbox
    get_sandbox()
    yield
    # Shutdown: cleanup
    if _sandbox:
        _sandbox.cleanup()


app = FastAPI(
    title="Deep Agents API",
    description="Full LangGraph-compatible API for Deep Agents",
    version="0.2.0",
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


# In-memory thread storage with proper state structure
threads: dict[str, dict] = {}


def get_timestamp():
    """Get current ISO timestamp."""
    return datetime.now(timezone.utc).isoformat()


def create_empty_state() -> dict:
    """Create empty agent state with all required fields."""
    return {
        "messages": [],
        "todos": [],
        "files": {},
    }


def serialize_message(msg: BaseMessage) -> dict:
    """Serialize a LangChain message to dict format for the frontend."""
    result = {
        "id": getattr(msg, "id", str(uuid.uuid4())),
        "type": msg.type,
        "content": msg.content if isinstance(msg.content, str) else "",
    }
    
    # Handle tool calls for AI messages
    if hasattr(msg, "tool_calls") and msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.get("id", str(uuid.uuid4())),
                "name": tc.get("name", ""),
                "args": tc.get("args", {}),
            }
            for tc in msg.tool_calls
        ]
    
    # Handle tool message specifics
    if msg.type == "tool":
        result["tool_call_id"] = getattr(msg, "tool_call_id", "")
        result["name"] = getattr(msg, "name", "")
    
    return result


def extract_state_from_agent_state(agent_state: dict) -> dict:
    """Extract todos and files from agent state."""
    state = create_empty_state()
    
    # Extract messages
    if "messages" in agent_state:
        state["messages"] = [
            serialize_message(m) if isinstance(m, BaseMessage) else m
            for m in agent_state["messages"]
        ]
    
    # Extract todos
    if "todos" in agent_state:
        state["todos"] = agent_state["todos"]
    
    # Extract files
    if "files" in agent_state:
        # Convert FileData to simple content strings for frontend
        # Also handle binary files from Claude Skills
        files = {}
        for path, file_data in agent_state["files"].items():
            print(f"[Files] Processing {path}: type={type(file_data)}, is_dict={isinstance(file_data, dict)}")
            if isinstance(file_data, dict):
                print(f"[Files] Dict keys: {file_data.keys()}")
                print(f"[Files] is_binary={file_data.get('is_binary')}, has_url={bool(file_data.get('download_url'))}, has_base64={bool(file_data.get('content_base64'))}")
                # Check if it's a binary file from Skills
                if file_data.get("is_binary"):
                    # For binary files, include metadata for download/preview
                    binary_file_data = {
                        "content": file_data.get("content", ["[Binary file]"])[0] if file_data.get("content") else "[Binary file]",
                        "is_binary": True,
                        "content_type": file_data.get("content_type", "application/octet-stream"),
                        "size": file_data.get("size", 0),
                    }
                    # Prefer download_url over base64
                    if file_data.get("download_url"):
                        binary_file_data["download_url"] = file_data["download_url"]
                        print(f"[Files] Binary file with download URL")
                    elif file_data.get("content_base64"):
                        binary_file_data["content_base64"] = file_data["content_base64"]
                        print(f"[Files] Binary file with base64 fallback")
                    files[path] = binary_file_data
                elif "content" in file_data:
                    # FileData format: {"content": [...lines...], ...}
                    files[path] = "\n".join(file_data["content"])
                    print(f"[Files] Text file with content array")
                else:
                    files[path] = str(file_data)
                    print(f"[Files] Unknown dict format, converting to string")
            elif isinstance(file_data, str):
                files[path] = file_data
                print(f"[Files] Already a string")
            else:
                files[path] = str(file_data)
                print(f"[Files] Other type, converting to string")
        state["files"] = files
    
    return state


# ============ Health Check ============

@app.get("/ok")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok"}


@app.get("/info")
async def get_info():
    """Get server info."""
    features = ["todos", "files", "execute", "subagents"]
    
    # Check if Skills are available
    from skills import get_skills_client
    skills_client = get_skills_client()
    if skills_client:
        features.append("document_generation")
        # Don't forget to close the client
        import asyncio
        asyncio.create_task(skills_client.close())
    
    return {
        "version": "0.2.0",
        "sandbox": "local",
        "features": features,
        "skills_available": skills_client is not None,
    }


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
            "metadata": {"description": "Full-featured deep agent with planning, filesystem, and subagents"},
            "created_at": get_timestamp(),
            "updated_at": get_timestamp(),
        }
    ]


@app.post("/assistants/search")
async def search_assistants(request: Request):
    """Search assistants."""
    return await list_assistants()


@app.get("/assistants/{assistant_id}")
async def get_assistant(assistant_id: str):
    """Get assistant by ID."""
    if assistant_id != "agent":
        raise HTTPException(status_code=404, detail="Assistant not found")
    return (await list_assistants())[0]


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
        "values": create_empty_state(),
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
            "values": create_empty_state(),
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
    # Sort by updated_at descending
    all_threads.sort(key=lambda t: t.get("updated_at", ""), reverse=True)
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
        "values": thread.get("values", create_empty_state()),
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
        # Deep merge values
        for key, value in body["values"].items():
            thread["values"][key] = value
    
    thread["updated_at"] = get_timestamp()
    
    return {"checkpoint": {"thread_id": thread_id}}


# ============ Runs API - Main Streaming Endpoint ============

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
            "values": create_empty_state(),
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
    """Create a streaming run - main endpoint for chat.
    
    This endpoint streams events in LangGraph format:
    - event: metadata (run info)
    - event: values (full state snapshots)
    - event: updates (per-node updates)
    - event: messages/partial (streaming message chunks)
    - event: messages/complete (complete messages)
    - event: end (stream complete)
    """
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
            "values": create_empty_state(),
        }
    
    thread = threads[thread_id]
    input_data = body.get("input", {})
    config = body.get("config", {})
    stream_mode = body.get("stream_mode", ["values", "messages"])
    
    # Extract messages from input
    input_messages = input_data.get("messages", [])
    
    async def generate_stream() -> AsyncGenerator[bytes, None]:
        """Generate SSE stream in LangGraph format."""
        import traceback
        run_id = str(uuid.uuid4())
        event_id = 0
        
        def make_event(event_type: str, data: Any) -> bytes:
            nonlocal event_id
            event_id += 1
            event_str = f"id: {event_id}\nevent: {event_type}\ndata: {json.dumps(data)}\n\n"
            print(f"[SSE] Sending event {event_id}: {event_type}", flush=True)
            return event_str.encode()
        
        try:
            thread["status"] = "busy"
            print(f"[Stream] Starting run {run_id} for thread {thread_id}")
            print(f"[Stream] Input messages: {input_messages}")
            print(f"[Stream] Stream modes: {stream_mode}")
            
            # Send metadata event
            yield make_event("metadata", {
                "run_id": run_id,
                "thread_id": thread_id,
                "assistant_id": body.get("assistant_id", "agent"),
            })
            await asyncio.sleep(0)  # Flush
            
            # Build LangChain messages from input
            messages = []
            for msg in input_messages:
                content = msg.get("content", "")
                msg_type = msg.get("type", "human")
                msg_id = msg.get("id", str(uuid.uuid4()))
                
                if msg_type == "human":
                    messages.append(HumanMessage(content=content, id=msg_id))
                elif msg_type == "ai":
                    messages.append(AIMessage(content=content, id=msg_id))
            
            print(f"[Stream] Built {len(messages)} LangChain messages")
            
            if not messages:
                # No input, just return current state
                print("[Stream] No messages, returning current state")
                yield make_event("values", thread["values"])
                yield make_event("end", None)
                thread["status"] = "idle"
                return
            
            # Add human message to thread state immediately
            for msg in input_messages:
                thread["values"]["messages"].append({
                    "id": msg.get("id", str(uuid.uuid4())),
                    "type": msg.get("type", "human"),
                    "content": msg.get("content", ""),
                })
            
            # Send initial values with human message
            yield make_event("values", thread["values"])
            
            # Create agent with sandbox backend
            print("[Stream] Creating agent with sandbox backend...")
            sandbox = get_sandbox()
            agent = create_agent(sandbox)
            print(f"[Stream] Agent created with sandbox at {sandbox.working_dir}")
            
            # Configure run with high recursion limit for complex tasks
            run_config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 500,  # Allow up to 500 tool calls per run
            }
            if config:
                run_config["configurable"].update(config.get("configurable", {}))
            
            # Track state during execution
            current_ai_message = {
                "id": str(uuid.uuid4()),
                "type": "ai",
                "content": "",
                "tool_calls": [],
            }
            current_node = None
            accumulated_content = ""
            # Track tool call IDs for matching tool results
            pending_tool_calls: dict[str, dict] = {}  # tool_call_id -> {name, args}
            
            # Track run hierarchy to filter subagent events
            # We only want to stream messages from the main agent, not nested subagents
            main_run_id: Optional[str] = None
            active_task_tools: set[str] = set()  # Track active 'task' tool run_ids
            
            print("[Stream] Starting agent stream...")
            async for event in agent.astream_events(
                {"messages": messages},
                config=run_config,
                version="v2",
            ):
                event_type = event.get("event", "")
                event_name = event.get("name", "")
                event_data = event.get("data", {})
                run_id = event.get("run_id", "")
                parent_ids = event.get("parent_ids", [])
                tags = event.get("tags", [])
                
                # Capture the main run ID from the first chain start
                if event_type == "on_chain_start" and main_run_id is None:
                    main_run_id = run_id
                
                # Check if this event is from a subagent (nested inside a task tool)
                # We check this BEFORE updating active_task_tools so that task tool
                # start/end events themselves are not filtered
                is_subagent_event = False
                for parent_id in parent_ids:
                    if parent_id in active_task_tools:
                        is_subagent_event = True
                        break
                
                # Track when 'task' tools start/end (these spawn subagents)
                # Do this AFTER checking is_subagent_event so task tool events are not filtered
                if event_type == "on_tool_start" and event_name == "task":
                    active_task_tools.add(run_id)
                    print(f"[Stream] Subagent started: {run_id}", flush=True)
                elif event_type == "on_tool_end" and run_id in active_task_tools:
                    active_task_tools.discard(run_id)
                    print(f"[Stream] Subagent ended: {run_id}", flush=True)
                
                # Track node transitions
                if event_type == "on_chain_start" and event_name:
                    if current_node != event_name:
                        current_node = event_name
                        print(f"[Stream] Node: {current_node}")
                
                # Reset AI message state when model starts a new invocation (main agent only)
                if event_type == "on_chat_model_start" and not is_subagent_event:
                    # If we have a previous AI message with content, finalize it
                    if accumulated_content or current_ai_message["tool_calls"]:
                        final_msg = {
                            "id": current_ai_message["id"],
                            "type": "ai",
                            "content": accumulated_content,
                        }
                        if current_ai_message["tool_calls"]:
                            final_msg["tool_calls"] = current_ai_message["tool_calls"]
                        
                        # Add to thread if not already there
                        if not any(m.get("id") == current_ai_message["id"] for m in thread["values"]["messages"]):
                            thread["values"]["messages"].append(final_msg)
                    
                    # Start fresh AI message for this model invocation
                    current_ai_message = {
                        "id": str(uuid.uuid4()),
                        "type": "ai",
                        "content": "",
                        "tool_calls": [],
                    }
                    accumulated_content = ""
                    print(f"[Stream] New AI message: {current_ai_message['id']}", flush=True)
                
                # Handle chat model streaming (main agent only - skip subagent messages)
                if event_type == "on_chat_model_stream" and not is_subagent_event:
                    chunk = event_data.get("chunk")
                    if chunk:
                        # Extract text content
                        content = chunk.content if hasattr(chunk, "content") else ""
                        text_content = ""
                        
                        if isinstance(content, str):
                            text_content = content
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    text_content += block.get("text", "")
                                elif isinstance(block, str):
                                    text_content += block
                        
                        if text_content:
                            accumulated_content += text_content
                            current_ai_message["content"] = accumulated_content
                            
                            # Send message chunk in LangGraph SDK format
                            # Format: [{content: [{text: "...", type: "text"}], type: "AIMessageChunk", id: "..."}]
                            yield make_event("messages", [{
                                "content": [{"text": text_content, "type": "text"}],
                                "type": "AIMessageChunk",
                                "id": current_ai_message["id"],
                            }])
                            await asyncio.sleep(0)  # Flush for real-time streaming
                        
                        # Handle tool calls in chunk
                        # Check both tool_calls (complete) and tool_call_chunks (streaming)
                        tool_calls_to_process = []
                        
                        if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                            tool_calls_to_process.extend(chunk.tool_calls)
                        
                        # Also check tool_call_chunks for streaming tool calls
                        if hasattr(chunk, "tool_call_chunks") and chunk.tool_call_chunks:
                            tool_calls_to_process.extend(chunk.tool_call_chunks)
                        
                        # Also check additional_kwargs for Anthropic format
                        if hasattr(chunk, "additional_kwargs"):
                            ak = chunk.additional_kwargs
                            if ak.get("tool_calls"):
                                tool_calls_to_process.extend(ak["tool_calls"])
                        
                        for tc in tool_calls_to_process:
                            # Handle both dict and object formats
                            if hasattr(tc, "get"):
                                tc_id = tc.get("id") or tc.get("index") or str(uuid.uuid4())
                                tc_name = tc.get("name", "")
                                tc_args = tc.get("args", {})
                            else:
                                tc_id = getattr(tc, "id", None) or getattr(tc, "index", None) or str(uuid.uuid4())
                                tc_name = getattr(tc, "name", "")
                                tc_args = getattr(tc, "args", {})
                            
                            # Only add if we have a name (skip partial chunks)
                            if tc_name:
                                tool_call = {
                                    "id": str(tc_id),
                                    "name": tc_name,
                                    "args": tc_args if isinstance(tc_args, dict) else {},
                                }
                                # Check if already added
                                if tool_call["id"] not in pending_tool_calls:
                                    current_ai_message["tool_calls"].append(tool_call)
                                    pending_tool_calls[tool_call["id"]] = tool_call
                                    print(f"[Stream] Tool call added: {tc_name} (id={tool_call['id']})", flush=True)
                
                # Handle tool execution (main agent tools only, not subagent internal tools)
                elif event_type == "on_tool_start" and not is_subagent_event:
                    tool_name = event.get("name", "")
                    tool_input = event_data.get("input", {})
                    print(f"[Stream] Tool start: {tool_name} with input keys: {list(tool_input.keys()) if isinstance(tool_input, dict) else 'not dict'}", flush=True)
                    
                    # Extract todos from write_todos tool INPUT (not output)
                    if tool_name == "write_todos" and isinstance(tool_input, dict):
                        todos_input = tool_input.get("todos", [])
                        if todos_input:
                            # Convert to frontend format
                            formatted_todos = []
                            for i, todo in enumerate(todos_input):
                                if isinstance(todo, dict):
                                    formatted_todos.append({
                                        "id": todo.get("id", str(i)),
                                        "content": todo.get("content", todo.get("description", str(todo))),
                                        "status": todo.get("status", "pending"),
                                    })
                                elif isinstance(todo, str):
                                    formatted_todos.append({
                                        "id": str(i),
                                        "content": todo,
                                        "status": "pending",
                                    })
                            thread["values"]["todos"] = formatted_todos
                            print(f"[Stream] Todos extracted from input: {len(formatted_todos)} items", flush=True)
                            yield make_event("values", thread["values"])
                            await asyncio.sleep(0)
                    
                    # If we have accumulated content or tool calls, send the AI message first
                    if accumulated_content or current_ai_message["tool_calls"]:
                        ai_msg_to_send = {
                            "id": current_ai_message["id"],
                            "type": "ai",
                            "content": accumulated_content,
                        }
                        if current_ai_message["tool_calls"]:
                            ai_msg_to_send["tool_calls"] = current_ai_message["tool_calls"]
                        
                        # Add to thread state if not already there
                        if not any(m.get("id") == current_ai_message["id"] for m in thread["values"]["messages"]):
                            thread["values"]["messages"].append(ai_msg_to_send)
                            yield make_event("values", thread["values"])
                            await asyncio.sleep(0)
                    
                    # Send update event for tool start
                    yield make_event("updates", {
                        "tools": {
                            "tool_calls": [{
                                "name": tool_name,
                                "args": tool_input,
                                "status": "running",
                            }]
                        }
                    })
                    await asyncio.sleep(0)  # Flush immediately
                
                elif event_type == "on_tool_end" and not is_subagent_event:
                    tool_name = event.get("name", "")
                    tool_output = event_data.get("output", "")
                    run_id = event.get("run_id", "")
                    print(f"[Stream] Tool end: {tool_name}", flush=True)
                    
                    # Find the tool_call_id for this tool
                    tool_call_id = None
                    for tc_id, tc_info in pending_tool_calls.items():
                        if tc_info.get("name") == tool_name:
                            tool_call_id = tc_id
                            break
                    
                    # Add tool message to thread with tool_call_id
                    tool_msg = {
                        "id": str(uuid.uuid4()),
                        "type": "tool",
                        "name": tool_name,
                        "content": str(tool_output)[:2000] if tool_output else "",
                    }
                    if tool_call_id:
                        tool_msg["tool_call_id"] = tool_call_id
                    
                    thread["values"]["messages"].append(tool_msg)
                    
                    # Remove processed tool call from pending
                    if tool_call_id and tool_call_id in pending_tool_calls:
                        del pending_tool_calls[tool_call_id]
                    
                    # Send update event for tool completion
                    yield make_event("updates", {
                        "tools": {
                            "messages": [tool_msg]
                        }
                    })
                    await asyncio.sleep(0)  # Flush immediately
                    
                    # Sync files from sandbox after file-related tools
                    if tool_name in ("write_file", "edit_file"):
                        if sync_sandbox_files_to_state(sandbox, thread["values"]):
                            print(f"[Stream] Files synced: {list(thread['values'].get('files', {}).keys())}", flush=True)
                            yield make_event("values", thread["values"])
                            await asyncio.sleep(0)  # Flush immediately
                
                # Handle state updates (todos, files)
                elif event_type == "on_chain_end":
                    output = event_data.get("output", {})
                    if isinstance(output, dict):
                        # Check for todos update
                        if "todos" in output:
                            thread["values"]["todos"] = output["todos"]
                            print(f"[Stream] Todos updated: {len(output['todos'])} items", flush=True)
                            yield make_event("values", thread["values"])
                            await asyncio.sleep(0)  # Flush immediately
                        
                        # Check for files update
                        if "files" in output:
                            for path, file_data in output["files"].items():
                                if isinstance(file_data, dict):
                                    # Check if it's a binary file from Claude Skills
                                    if file_data.get("is_binary"):
                                        # Preserve binary file metadata for frontend
                                        binary_data = {
                                            "content": file_data.get("content", ["[Binary file]"])[0] if isinstance(file_data.get("content"), list) else file_data.get("content", "[Binary file]"),
                                            "is_binary": True,
                                            "content_type": file_data.get("content_type", "application/octet-stream"),
                                            "size": file_data.get("size", 0),
                                        }
                                        # Include download_url if available
                                        if file_data.get("download_url"):
                                            binary_data["download_url"] = file_data["download_url"]
                                        elif file_data.get("content_base64"):
                                            binary_data["content_base64"] = file_data["content_base64"]
                                        thread["values"]["files"][path] = binary_data
                                        print(f"[Stream] Binary file stored: {path} (is_binary=True, has_url={bool(file_data.get('download_url'))})")
                                    elif "content" in file_data:
                                        # Regular text file with content array
                                        content = file_data["content"]
                                        if isinstance(content, list):
                                            thread["values"]["files"][path] = "\n".join(content)
                                        else:
                                            thread["values"]["files"][path] = str(content)
                                    else:
                                        thread["values"]["files"][path] = str(file_data)
                                elif file_data is not None:
                                    thread["values"]["files"][path] = str(file_data)
                            print(f"[Stream] Files updated: {list(output['files'].keys())}")
                            yield make_event("values", thread["values"])
            
            print(f"[Stream] Agent finished. Response length: {len(accumulated_content)}", flush=True)
            
            # Finalize AI message
            if accumulated_content or current_ai_message["tool_calls"]:
                # Clean up tool_calls if empty
                if not current_ai_message["tool_calls"]:
                    del current_ai_message["tool_calls"]
                
                # Add final AI message to thread state
                final_ai_msg = {
                    "id": current_ai_message["id"],
                    "type": "ai",
                    "content": accumulated_content,
                }
                if current_ai_message.get("tool_calls"):
                    final_ai_msg["tool_calls"] = current_ai_message["tool_calls"]
                
                thread["values"]["messages"].append(final_ai_msg)
            
            # Send final state with all messages
            yield make_event("values", thread["values"])
            await asyncio.sleep(0)
            
            # Send end event (no data needed)
            yield make_event("end", {})
            
            thread["status"] = "idle"
            thread["updated_at"] = get_timestamp()
            print(f"[Stream] Run {run_id} completed successfully")
            
        except Exception as e:
            print(f"[Stream] ERROR: {str(e)}")
            print(f"[Stream] Traceback: {traceback.format_exc()}")
            thread["status"] = "error"
            yield make_event("error", {"message": str(e), "code": "AGENT_ERROR"})
    
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
        "values": thread.get("values", create_empty_state()),
        "metadata": thread.get("metadata", {}),
        "created_at": thread.get("created_at"),
    }]


# =============================================================================
# CLAUDE SKILLS ENDPOINTS
# =============================================================================

@app.post("/skills/generate")
async def skills_generate(request: Request):
    """Generate documents using Claude Skills.
    
    This endpoint uses Claude's native document generation skills (xlsx, pptx, pdf, docx)
    to create professional documents based on user requests.
    
    Request body:
    {
        "messages": [{"role": "user", "content": "Create a spreadsheet..."}],
        "thread_id": "optional-thread-id",
        "system_prompt": "optional custom system prompt"
    }
    
    Returns SSE stream with:
    - content_block_* events (Claude's response)
    - file_ready events (generated documents with base64 content)
    """
    body = await request.json()
    messages = body.get("messages", [])
    thread_id = body.get("thread_id", str(uuid.uuid4()))
    system_prompt = body.get("system_prompt", SKILLS_SYSTEM_PROMPT)
    
    if not messages:
        raise HTTPException(status_code=400, detail="No messages provided")
    
    # Get skills client
    skills_client = get_skills_client()
    if not skills_client:
        raise HTTPException(
            status_code=503,
            detail="Claude Skills not available - ANTHROPIC_API_KEY not configured"
        )
    
    # Convert messages to Anthropic format
    anthropic_messages = []
    for msg in messages:
        role = msg.get("role") or msg.get("type", "user")
        content = msg.get("content", "")
        
        # Map our types to Anthropic roles
        if role in ("human", "user"):
            role = "user"
        elif role in ("ai", "assistant"):
            role = "assistant"
        
        anthropic_messages.append({"role": role, "content": content})
    
    async def generate_stream():
        """Stream skills generation events."""
        try:
            print(f"[Skills] Starting generation for thread {thread_id}")
            
            # Track generated files
            generated_files = []
            accumulated_text = ""
            
            async for event in skills_client.generate_with_skills(
                anthropic_messages,
                system_prompt=system_prompt,
            ):
                event_type = event.get("type", event.get("_event_type", ""))
                
                # Handle file_ready events (our custom event)
                if event_type == "file_ready":
                    generated_files.append({
                        "file_id": event.get("file_id"),
                        "file_name": event.get("file_name"),
                        "skill": event.get("skill"),
                        "size": event.get("size"),
                    })
                    
                    # Send file_ready event to frontend
                    yield f"event: file_ready\ndata: {json.dumps(event)}\n\n"
                    continue
                
                # Handle file_error events
                if event_type == "file_error":
                    yield f"event: file_error\ndata: {json.dumps(event)}\n\n"
                    continue
                
                # Handle text content
                if event_type == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        accumulated_text += text
                        yield f"event: text_delta\ndata: {json.dumps({'text': text})}\n\n"
                
                # Handle message completion
                if event_type == "message_stop":
                    yield f"event: message_complete\ndata: {json.dumps({'text': accumulated_text, 'files': generated_files})}\n\n"
                
                # Forward other events for debugging/transparency
                if event_type in ("message_start", "content_block_start", "content_block_stop"):
                    yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
            
            # Send final summary
            yield f"event: done\ndata: {json.dumps({'thread_id': thread_id, 'files_generated': len(generated_files)})}\n\n"
            
            print(f"[Skills] Generation complete: {len(generated_files)} files generated")
            
        except httpx.HTTPStatusError as e:
            error_msg = f"Claude API error: {e.response.status_code}"
            print(f"[Skills] Error: {error_msg}")
            yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"
        except Exception as e:
            error_msg = str(e)
            print(f"[Skills] Error: {error_msg}")
            yield f"event: error\ndata: {json.dumps({'error': error_msg})}\n\n"
        finally:
            await skills_client.close()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/skills/status")
async def skills_status():
    """Check if Claude Skills are available."""
    client = get_skills_client()
    available = client is not None
    
    if client:
        await client.close()
    
    return {
        "available": available,
        "skills": [s.value for s in SkillType] if available else [],
        "message": "Claude Skills ready" if available else "ANTHROPIC_API_KEY not configured",
    }


@app.post("/skills/detect")
async def skills_detect(request: Request):
    """Detect if a message should use Claude Skills.
    
    This endpoint helps the frontend decide whether to route
    a request to the skills endpoint or the regular agent.
    """
    body = await request.json()
    message = body.get("message", "")
    
    should_use_skills = is_document_generation_request(message)
    
    return {
        "use_skills": should_use_skills,
        "message": message[:100] + "..." if len(message) > 100 else message,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "10000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
