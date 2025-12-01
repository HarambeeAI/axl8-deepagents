"""Deep Agent configuration for LangGraph deployment.

This module creates a full-featured deep agent with:
- Planning (TodoListMiddleware)
- Filesystem access (ls, read_file, write_file, edit_file, glob, grep)
- Shell execution (execute) - requires SandboxBackendProtocol
- Sub-agent delegation (task tool)
- Document generation (create_document) - via Claude Skills
"""

import os
from typing import Optional
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.backends.protocol import SandboxBackendProtocol
from deepagents.middleware.skills import SkillsMiddleware, skills_available

# System prompt for the deep agent
SYSTEM_PROMPT = """You are a helpful AI assistant powered by Deep Agents.

You have access to powerful tools including:
- **Planning**: Use `write_todos` to create task lists before complex work
- **Filesystem**: `ls`, `read_file`, `write_file`, `edit_file`, `glob`, `grep`
- **Shell execution**: `execute` to run shell commands
- **Sub-agents**: `task` to delegate complex work to isolated sub-agents
- **Document generation**: `create_document` to generate professional documents (xlsx, pptx, pdf, docx)

## Guidelines:
1. For complex tasks, ALWAYS start by creating a todo list with `write_todos`
2. Break down work into clear, actionable steps
3. Use sub-agents for independent, parallelizable work
4. Keep the user informed of your progress
5. Be thorough and methodical
6. When the user needs a document output (spreadsheet, presentation, PDF, Word doc), 
   use `create_document` to generate a professional document

## Working Directory:
Your workspace is at /workspace. All file operations should use paths relative to this.
"""


def create_agent(backend: Optional[SandboxBackendProtocol] = None):
    """Create and return the deep agent.
    
    Args:
        backend: Optional sandbox backend for filesystem and shell execution.
                If None, uses StateBackend (in-memory, no shell execution).
                Pass a SandboxBackendProtocol implementation for full features.
    
    Returns:
        Compiled LangGraph agent with all deep agent capabilities.
    """
    # Use provided backend or fall back to filesystem/state backend
    if backend is None:
        use_filesystem = os.getenv("USE_FILESYSTEM_BACKEND", "false").lower() == "true"
        if use_filesystem:
            root_dir = os.getenv("FILESYSTEM_ROOT_DIR", "/tmp/deepagent_workspace")
            backend = FilesystemBackend(root_dir=root_dir)
        # If no backend specified and not using filesystem, StateBackend is used by default
    
    # Build middleware list
    middleware = []
    
    # Add Skills middleware if available (for document generation)
    if skills_available():
        print("[Agent] Claude Skills available - adding document generation capability")
        middleware.append(SkillsMiddleware(workspace_path="/workspace"))
    else:
        print("[Agent] Claude Skills not available (ANTHROPIC_API_KEY not set)")
    
    agent = create_deep_agent(
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
        middleware=middleware,
    )
    
    return agent


# For LangGraph CLI compatibility - create default agent
# Note: This uses StateBackend by default. For shell execution,
# the server should call create_agent(sandbox) with a proper backend.
def get_default_agent():
    """Get the default agent for LangGraph CLI."""
    return create_agent()


# Export for langgraph.json
agent = get_default_agent()
