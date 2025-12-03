"""Middleware for providing Claude Skills document generation to an agent.

This middleware adds a `create_document` tool that uses Claude's native Skills
to generate professional documents (xlsx, pptx, pdf, docx) and saves them
to the agent's file state, making them accessible in the UI's "Files" panel.

The agent can call this tool when it needs to produce a final document output,
seamlessly integrating document generation into the existing workflow.
"""

import os
import json
import base64
import httpx
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Literal
from enum import Enum

from langchain.agents.middleware.types import AgentMiddleware
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import tool
from langgraph.types import Command


class DocumentType(str, Enum):
    """Supported document types for generation."""
    XLSX = "xlsx"
    PPTX = "pptx"
    PDF = "pdf"
    DOCX = "docx"


# Content type mapping for documents
CONTENT_TYPES = {
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


CREATE_DOCUMENT_DESCRIPTION = """Create a professional document using Claude Skills.

This tool generates high-quality documents in various formats:
- **xlsx**: Excel spreadsheets with data, formulas, charts
- **pptx**: PowerPoint presentations with slides, layouts, graphics
- **pdf**: PDF documents with formatting, tables, images
- **docx**: Word documents with rich text, styles, sections

Use this tool when you need to produce a final deliverable document for the user.
The generated document will be saved to the workspace and appear in the Files panel.

Parameters:
- document_type: The format to generate (xlsx, pptx, pdf, docx)
- filename: Name for the output file (without extension - it will be added)
- description: Detailed description of what the document should contain.
  Be specific about:
  - Content structure (sections, slides, sheets)
  - Data to include (tables, charts, text)
  - Formatting preferences (colors, styles)
  - Any specific requirements

Examples:
- create_document(document_type="xlsx", filename="sales_report", 
    description="Create a sales report spreadsheet with monthly revenue data for Q1 2024...")
- create_document(document_type="pptx", filename="project_proposal",
    description="Create a 10-slide presentation about our new product launch...")
- create_document(document_type="pdf", filename="user_guide",
    description="Create a user guide document with installation instructions...")
"""

SKILLS_SYSTEM_PROMPT = """## Document Generation Tool `create_document`

You have access to a `create_document` tool that uses Claude Skills to generate professional documents.

When the user asks for a document output (spreadsheet, presentation, PDF, Word doc), use this tool
to create a high-quality, professionally formatted document.

Available document types:
- xlsx: Excel spreadsheets (data, charts, formulas)
- pptx: PowerPoint presentations (slides, graphics)
- pdf: PDF documents (formatted reports)
- docx: Word documents (text documents, reports)

The generated document will be saved to the workspace and accessible in the Files panel."""


class SkillsClient:
    """Synchronous client for Claude Skills API used by the middleware."""
    
    ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
    ANTHROPIC_FILES_URL = "https://api.anthropic.com/v1/files"
    BETA_FEATURES = "code-execution-2025-08-25,skills-2025-10-02,files-api-2025-04-14"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def _get_headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "anthropic-beta": self.BETA_FEATURES,
        }
    
    def generate_document(
        self,
        document_type: str,
        description: str,
    ) -> tuple[Optional[bytes], Optional[str], Optional[str]]:
        """Generate a document using Claude Skills (synchronous).
        
        Returns:
            Tuple of (file_content, filename, error_message)
        """
        # Build the prompt for document generation
        prompt = f"""Create a {document_type.upper()} document with the following specifications:

{description}

Please generate this document now using the appropriate skill. Make it professional and complete."""

        body = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "container": {
                "skills": [
                    {"type": "anthropic", "skill_id": document_type, "version": "latest"}
                ]
            },
            "tools": [
                {"type": "code_execution_20250825", "name": "code_execution"}
            ],
        }
        
        try:
            file_id = None
            file_name = None
            
            # Use synchronous httpx client with streaming
            with httpx.Client(timeout=300.0) as client:
                with client.stream(
                    "POST",
                    self.ANTHROPIC_API_URL,
                    headers=self._get_headers(),
                    json=body,
                ) as response:
                    response.raise_for_status()
                    
                    buffer = ""
                    for chunk in response.iter_text():
                        buffer += chunk
                        
                        while "\n\n" in buffer:
                            event_str, buffer = buffer.split("\n\n", 1)
                            
                            # Parse SSE event
                            data = None
                            for line in event_str.split("\n"):
                                if line.startswith("data:"):
                                    try:
                                        data = json.loads(line[5:].strip())
                                    except json.JSONDecodeError:
                                        continue
                            
                            if not data:
                                continue
                            
                            # Look for file_id in various event structures
                            file_info = self._extract_file_info(data)
                            if file_info:
                                file_id = file_info.get("file_id")
                                file_name = file_info.get("file_name")
                
                if file_id:
                    # Download the file
                    content = self._download_file(client, file_id)
                    return content, file_name, None
                else:
                    return None, None, "No document was generated. Claude may not have used the skill."
                
        except httpx.HTTPStatusError as e:
            return None, None, f"API error: {e.response.status_code}"
        except Exception as e:
            return None, None, str(e)
    
    def _extract_file_info(self, event: Dict[str, Any]) -> Optional[Dict[str, str]]:
        """Extract file information from SSE event."""
        event_type = event.get("type", "")
        
        # Check content_block_start
        if event_type == "content_block_start":
            content_block = event.get("content_block", {})
            block_type = content_block.get("type", "")
            
            if block_type in ("bash_code_execution_tool_result", "text_editor_code_execution_tool_result"):
                content = content_block.get("content", {})
                
                if isinstance(content, dict) and "content" in content:
                    for item in content.get("content", []):
                        if isinstance(item, dict) and item.get("file_id"):
                            return {
                                "file_id": item["file_id"],
                                "file_name": item.get("file_name", "document"),
                            }
                
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("file_id"):
                            return {
                                "file_id": item["file_id"],
                                "file_name": item.get("file_name", "document"),
                            }
        
        # Check content_block_delta
        if event_type == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("file_id"):
                return {
                    "file_id": delta["file_id"],
                    "file_name": delta.get("file_name", "document"),
                }
        
        return None
    
    def _download_file(self, client: httpx.Client, file_id: str) -> bytes:
        """Download file from Claude Files API."""
        url = f"{self.ANTHROPIC_FILES_URL}/{file_id}/content"
        response = client.get(url, headers=self._get_headers())
        response.raise_for_status()
        return response.content


def _get_skills_client() -> Optional[SkillsClient]:
    """Get a skills client if API key is available."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return SkillsClient(api_key)


class SkillsMiddleware(AgentMiddleware):
    """Middleware that provides document generation capabilities via Claude Skills.
    
    This middleware adds a `create_document` tool that generates professional
    documents (xlsx, pptx, pdf, docx) and saves them to the agent's file state.
    
    Usage:
        ```python
        from deepagents.middleware.skills import SkillsMiddleware
        
        agent = create_deep_agent(
            middleware=[
                SkillsMiddleware(),
                # ... other middleware
            ],
        )
        ```
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        workspace_path: str = "/workspace",
    ):
        """Initialize the Skills middleware.
        
        Args:
            system_prompt: Additional system prompt text to append.
            workspace_path: Base path for saving generated documents.
        """
        self._custom_system_prompt = system_prompt
        self.workspace_path = workspace_path
        
        # Create and store the tools - this is required by the middleware interface
        self.tools = self._create_tools()
        print(f"[SkillsMiddleware] Initialized with {len(self.tools)} tools: {[t.name for t in self.tools]}")
    
    def _create_tools(self) -> List:
        """Create the document generation tools."""
        workspace_path = self.workspace_path
        
        @tool(description=CREATE_DOCUMENT_DESCRIPTION)
        def create_document(
            document_type: Literal["xlsx", "pptx", "pdf", "docx"],
            filename: str,
            description: str,
            runtime: ToolRuntime,
        ) -> str:
            """Create a professional document using Claude Skills."""
            print(f"[Skills] create_document called: type={document_type}, filename={filename}")
            result = _create_document_sync(
                document_type=document_type,
                filename=filename,
                description=description,
                runtime=runtime,
                workspace_path=workspace_path,
            )
            print(f"[Skills] create_document result type: {type(result)}")
            return result
        
        return [create_document]
    
    def wrap_model_call(
        self,
        request,
        handler,
    ):
        """Update the system prompt to include document generation instructions.
        
        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.
            
        Returns:
            The model response from the handler.
        """
        # Build system prompt
        system_prompt = SKILLS_SYSTEM_PROMPT
        if self._custom_system_prompt:
            system_prompt += f"\n\n{self._custom_system_prompt}"
        
        # Append to existing system prompt
        if system_prompt:
            existing_prompt = request.system_prompt or ""
            new_prompt = f"{existing_prompt}\n\n{system_prompt}" if existing_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)
        
        return handler(request)
    
    async def awrap_model_call(
        self,
        request,
        handler,
    ):
        """(async) Update the system prompt to include document generation instructions.
        
        Args:
            request: The model request being processed.
            handler: The handler function to call with the modified request.
            
        Returns:
            The model response from the handler.
        """
        # Build system prompt
        system_prompt = SKILLS_SYSTEM_PROMPT
        if self._custom_system_prompt:
            system_prompt += f"\n\n{self._custom_system_prompt}"
        
        # Append to existing system prompt
        if system_prompt:
            existing_prompt = request.system_prompt or ""
            new_prompt = f"{existing_prompt}\n\n{system_prompt}" if existing_prompt else system_prompt
            request = request.override(system_prompt=new_prompt)
        
        return await handler(request)


def _create_document_sync(
    document_type: str,
    filename: str,
    description: str,
    runtime: ToolRuntime,
    workspace_path: str,
) -> str:
    """Synchronous implementation of document creation."""
    client = _get_skills_client()
    if not client:
        return "Error: Claude Skills not available - ANTHROPIC_API_KEY not configured"
    
    try:
        # Validate document type
        if document_type not in ["xlsx", "pptx", "pdf", "docx"]:
            return f"Error: Invalid document type '{document_type}'. Must be one of: xlsx, pptx, pdf, docx"
        
        # Clean filename and add extension
        clean_filename = filename.replace(" ", "_").replace("/", "_")
        if not clean_filename.endswith(f".{document_type}"):
            clean_filename = f"{clean_filename}.{document_type}"
        
        print(f"[Skills] Generating {document_type.upper()} document: {clean_filename}")
        
        # Generate the document
        content, generated_name, error = client.generate_document(
            document_type=document_type,
            description=description,
        )
        
        if error:
            return f"Error generating document: {error}"
        
        if not content:
            return "Error: No document content was generated"
        
        print(f"[Skills] Document generated: {len(content)} bytes")
        
        # Determine the file path
        file_path = f"{workspace_path}/{clean_filename}"
        timestamp = datetime.now(timezone.utc).isoformat()
        content_type = CONTENT_TYPES.get(document_type, "application/octet-stream")
        
        # Try to upload to Supabase Storage for public URL access
        public_url = None
        try:
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
            
            if supabase_url and supabase_key:
                from supabase import create_client
                supabase = create_client(supabase_url, supabase_key)
                
                # Upload to generated-files bucket
                bucket_name = "generated-files"
                storage_path = f"{int(datetime.now().timestamp())}-{clean_filename}"
                
                # Upload the file
                result = supabase.storage.from_(bucket_name).upload(
                    storage_path,
                    content,
                    file_options={"content-type": content_type}
                )
                
                # Get public URL
                url_result = supabase.storage.from_(bucket_name).get_public_url(storage_path)
                public_url = url_result
                print(f"[Skills] Uploaded to Supabase Storage: {public_url}")
        except Exception as upload_error:
            print(f"[Skills] Supabase upload failed (falling back to base64): {upload_error}")
        
        # Build file data for state
        file_data = {
            "content": [f"[Binary {document_type.upper()} file - {len(content)} bytes]"],
            "content_type": content_type,
            "created_at": timestamp,
            "modified_at": timestamp,
            "is_binary": True,
            "size": len(content),
        }
        
        # Add URL if uploaded, otherwise include base64 as fallback
        if public_url:
            file_data["download_url"] = public_url
        else:
            file_data["content_base64"] = base64.b64encode(content).decode("utf-8")
        
        files_update = {file_path: file_data}
        
        # Success message for the tool response
        if public_url:
            success_message = f"Successfully created {document_type.upper()} document: {clean_filename} ({len(content)} bytes). The file is available for download."
        else:
            success_message = f"Successfully created {document_type.upper()} document: {file_path} ({len(content)} bytes). The file is now available in the Files panel."
        
        # Return a Command to update state with the required ToolMessage
        return Command(
            update={
                "files": files_update,
                "messages": [
                    ToolMessage(
                        content=success_message,
                        tool_call_id=runtime.tool_call_id,
                    )
                ],
            },
        )
        
    except Exception as e:
        print(f"[Skills] Error: {str(e)}")
        return f"Error creating document: {str(e)}"


# Convenience function to check if skills are available
def skills_available() -> bool:
    """Check if Claude Skills are available (API key configured)."""
    return os.getenv("ANTHROPIC_API_KEY") is not None
