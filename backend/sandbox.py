"""Local subprocess sandbox for development and simple deployments.

This provides a simple sandbox that executes commands in a subprocess.
For production with proper isolation, use Modal, Daytona, or Runloop.
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Optional

from deepagents.backends.protocol import (
    ExecuteResponse,
    FileDownloadResponse,
    FileUploadResponse,
)
from deepagents.backends.sandbox import BaseSandbox


class LocalSandbox(BaseSandbox):
    """Local subprocess-based sandbox for development.
    
    This sandbox executes commands in a local subprocess with a dedicated
    working directory. It's suitable for development and simple deployments
    where full container isolation isn't required.
    
    For production deployments requiring isolation, use:
    - Modal (create_modal_sandbox)
    - Daytona (create_daytona_sandbox)  
    - Runloop (create_runloop_sandbox)
    """
    
    def __init__(
        self,
        working_dir: Optional[str] = None,
        timeout: int = 300,
    ):
        """Initialize the local sandbox.
        
        Args:
            working_dir: Working directory for command execution.
                        If None, creates a temp directory.
            timeout: Command execution timeout in seconds.
        """
        if working_dir:
            self._working_dir = Path(working_dir)
            self._working_dir.mkdir(parents=True, exist_ok=True)
            self._owns_dir = False
        else:
            self._working_dir = Path(tempfile.mkdtemp(prefix="deepagent_sandbox_"))
            self._owns_dir = True
        
        self._timeout = timeout
        self._sandbox_id = f"local-{os.getpid()}-{id(self)}"
    
    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        return self._sandbox_id
    
    @property
    def working_dir(self) -> Path:
        """Get the sandbox working directory."""
        return self._working_dir
    
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the sandbox.
        
        Args:
            command: Shell command to execute.
            
        Returns:
            ExecuteResponse with output, exit code, and truncation flag.
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self._working_dir),
                capture_output=True,
                text=True,
                timeout=self._timeout,
                env={**os.environ, "HOME": str(self._working_dir)},
            )
            
            output = result.stdout
            if result.stderr:
                output += "\n" + result.stderr if output else result.stderr
            
            # Truncate very long output
            max_output = 100000
            truncated = len(output) > max_output
            if truncated:
                output = output[:max_output] + "\n... [output truncated]"
            
            return ExecuteResponse(
                output=output,
                exit_code=result.returncode,
                truncated=truncated,
            )
            
        except subprocess.TimeoutExpired:
            return ExecuteResponse(
                output=f"Command timed out after {self._timeout} seconds",
                exit_code=-1,
                truncated=False,
            )
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {str(e)}",
                exit_code=-1,
                truncated=False,
            )
    
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the sandbox.
        
        Args:
            files: List of (path, content) tuples.
            
        Returns:
            List of FileUploadResponse objects.
        """
        responses = []
        for path, content in files:
            try:
                # Resolve path relative to working dir
                if path.startswith("/"):
                    full_path = self._working_dir / path.lstrip("/")
                else:
                    full_path = self._working_dir / path
                
                # Create parent directories
                full_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write file
                full_path.write_bytes(content)
                
                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        
        return responses
    
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the sandbox.
        
        Args:
            paths: List of file paths to download.
            
        Returns:
            List of FileDownloadResponse objects.
        """
        responses = []
        for path in paths:
            try:
                # Resolve path relative to working dir
                if path.startswith("/"):
                    full_path = self._working_dir / path.lstrip("/")
                else:
                    full_path = self._working_dir / path
                
                if not full_path.exists():
                    responses.append(FileDownloadResponse(
                        path=path,
                        content=None,
                        error=f"File not found: {path}"
                    ))
                    continue
                
                content = full_path.read_bytes()
                responses.append(FileDownloadResponse(
                    path=path,
                    content=content,
                    error=None
                ))
            except Exception as e:
                responses.append(FileDownloadResponse(
                    path=path,
                    content=None,
                    error=str(e)
                ))
        
        return responses
    
    def cleanup(self):
        """Clean up the sandbox working directory."""
        if self._owns_dir and self._working_dir.exists():
            shutil.rmtree(self._working_dir, ignore_errors=True)
    
    def __del__(self):
        """Cleanup on garbage collection."""
        self.cleanup()


def create_local_sandbox(
    working_dir: Optional[str] = None,
    timeout: int = 300,
) -> LocalSandbox:
    """Create a local subprocess sandbox.
    
    Args:
        working_dir: Working directory for command execution.
        timeout: Command execution timeout in seconds.
        
    Returns:
        LocalSandbox instance.
    """
    return LocalSandbox(working_dir=working_dir, timeout=timeout)
