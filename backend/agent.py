"""Deep Agent configuration for LangGraph deployment."""

import os
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

# System prompt for the deep agent
SYSTEM_PROMPT = """You are a helpful AI assistant powered by Deep Agents.

You have access to powerful tools including:
- File system operations (read, write, edit files)
- Shell command execution
- Task planning with todo lists
- Sub-agent delegation for complex tasks

Be thorough, methodical, and helpful. Break down complex tasks into manageable steps.
Always explain your reasoning and keep the user informed of your progress.
"""


def create_agent():
    """Create and return the deep agent.
    
    This function is called by LangGraph to instantiate the agent.
    """
    # Check if we should use filesystem backend (for local development)
    use_filesystem = os.getenv("USE_FILESYSTEM_BACKEND", "false").lower() == "true"
    root_dir = os.getenv("FILESYSTEM_ROOT_DIR", "/tmp/deepagent_workspace")
    
    backend = None
    if use_filesystem:
        backend = FilesystemBackend(root_dir=root_dir)
    
    agent = create_deep_agent(
        system_prompt=SYSTEM_PROMPT,
        backend=backend,
    )
    
    return agent


# Export the agent graph for LangGraph
agent = create_agent()
