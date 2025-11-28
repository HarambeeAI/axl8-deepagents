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


class ModalSandbox(BaseSandbox):
    """Modal cloud sandbox for production deployments.
    
    This sandbox executes commands in Modal's cloud containers,
    providing proper isolation for multi-user production environments.
    
    Requires MODAL_TOKEN_ID and MODAL_TOKEN_SECRET environment variables.
    """
    
    def __init__(
        self,
        sandbox_id: Optional[str] = None,
        timeout: int = 1800,  # 30 minutes default
    ):
        """Initialize the Modal sandbox.
        
        Args:
            sandbox_id: Optional existing sandbox ID to reuse.
            timeout: Command execution timeout in seconds.
        """
        self._timeout = timeout
        self._sandbox_id = sandbox_id
        self._sandbox = None
        self._app = None
        self._initialized = False
    
    def _ensure_initialized(self):
        """Lazily initialize the Modal sandbox on first use."""
        if self._initialized:
            return
        
        try:
            import modal
        except ImportError:
            raise ImportError(
                "Modal SDK not installed. Install with: pip install modal"
            )
        
        # Create Modal app
        self._app = modal.App("deepagents-sandbox")
        
        # Start the app context
        self._app_context = self._app.run()
        self._app_context.__enter__()
        
        if self._sandbox_id:
            # Reuse existing sandbox
            self._sandbox = modal.Sandbox.from_id(
                sandbox_id=self._sandbox_id, 
                app=self._app
            )
            print(f"[Modal] Connected to existing sandbox: {self._sandbox_id}")
        else:
            # Create new sandbox
            self._sandbox = modal.Sandbox.create(
                app=self._app,
                workdir="/workspace",
                timeout=3600,  # 1 hour sandbox lifetime
            )
            self._sandbox_id = self._sandbox.object_id
            print(f"[Modal] Created new sandbox: {self._sandbox_id}")
            
            # Wait for sandbox to be ready
            self._wait_for_ready()
        
        self._initialized = True
    
    def _wait_for_ready(self, max_attempts: int = 30):
        """Wait for sandbox to be ready."""
        import time
        
        for attempt in range(max_attempts):
            try:
                # Check if sandbox terminated
                if self._sandbox.poll() is not None:
                    raise RuntimeError("Modal sandbox terminated unexpectedly")
                
                # Try a simple command
                process = self._sandbox.exec("echo", "ready", timeout=10)
                process.wait()
                if process.returncode == 0:
                    print("[Modal] Sandbox ready")
                    return
            except Exception as e:
                if attempt < max_attempts - 1:
                    time.sleep(2)
                else:
                    raise RuntimeError(f"Modal sandbox failed to start: {e}")
    
    @property
    def id(self) -> str:
        """Unique identifier for the sandbox backend."""
        self._ensure_initialized()
        return self._sandbox_id
    
    @property
    def working_dir(self) -> Path:
        """Get the sandbox working directory."""
        return Path("/workspace")
    
    def execute(self, command: str) -> ExecuteResponse:
        """Execute a command in the Modal sandbox.
        
        Args:
            command: Shell command to execute.
            
        Returns:
            ExecuteResponse with output, exit code, and truncation flag.
        """
        self._ensure_initialized()
        
        try:
            process = self._sandbox.exec(
                "bash", "-c", command,
                timeout=self._timeout
            )
            process.wait()
            
            stdout = process.stdout.read() or ""
            stderr = process.stderr.read() or ""
            
            output = stdout
            if stderr:
                output += "\n" + stderr if output else stderr
            
            # Truncate very long output
            max_output = 100000
            truncated = len(output) > max_output
            if truncated:
                output = output[:max_output] + "\n... [output truncated]"
            
            return ExecuteResponse(
                output=output,
                exit_code=process.returncode,
                truncated=truncated,
            )
            
        except Exception as e:
            return ExecuteResponse(
                output=f"Error executing command: {str(e)}",
                exit_code=-1,
                truncated=False,
            )
    
    def upload_files(self, files: list[tuple[str, bytes]]) -> list[FileUploadResponse]:
        """Upload files to the Modal sandbox.
        
        Args:
            files: List of (path, content) tuples.
            
        Returns:
            List of FileUploadResponse objects.
        """
        self._ensure_initialized()
        
        responses = []
        for path, content in files:
            try:
                # Ensure path is absolute
                if not path.startswith("/"):
                    path = f"/workspace/{path}"
                
                with self._sandbox.open(path, "wb") as f:
                    f.write(content)
                
                responses.append(FileUploadResponse(path=path, error=None))
            except Exception as e:
                responses.append(FileUploadResponse(path=path, error=str(e)))
        
        return responses
    
    def download_files(self, paths: list[str]) -> list[FileDownloadResponse]:
        """Download files from the Modal sandbox.
        
        Args:
            paths: List of file paths to download.
            
        Returns:
            List of FileDownloadResponse objects.
        """
        self._ensure_initialized()
        
        responses = []
        for path in paths:
            try:
                # Ensure path is absolute
                if not path.startswith("/"):
                    path = f"/workspace/{path}"
                
                with self._sandbox.open(path, "rb") as f:
                    content = f.read()
                
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
        """Terminate the Modal sandbox."""
        if self._sandbox:
            try:
                self._sandbox.terminate()
                print(f"[Modal] Sandbox terminated: {self._sandbox_id}")
            except Exception as e:
                print(f"[Modal] Error terminating sandbox: {e}")
        
        if self._app_context:
            try:
                self._app_context.__exit__(None, None, None)
            except Exception:
                pass
        
        self._initialized = False
        self._sandbox = None
        self._app = None
    
    def __del__(self):
        """Cleanup on garbage collection."""
        # Don't cleanup automatically - let the sandbox persist
        pass


def create_modal_sandbox(
    sandbox_id: Optional[str] = None,
    timeout: int = 1800,
) -> ModalSandbox:
    """Create a Modal cloud sandbox.
    
    Args:
        sandbox_id: Optional existing sandbox ID to reuse.
        timeout: Command execution timeout in seconds.
        
    Returns:
        ModalSandbox instance.
    """
    return ModalSandbox(sandbox_id=sandbox_id, timeout=timeout)


def get_sandbox_backend(sandbox_type: Optional[str] = None) -> BaseSandbox:
    """Get the appropriate sandbox backend based on configuration.
    
    Args:
        sandbox_type: Type of sandbox ("modal", "local", or None for auto-detect).
        
    Returns:
        Sandbox backend instance.
    """
    # Auto-detect based on environment
    if sandbox_type is None:
        if os.getenv("MODAL_TOKEN_ID") and os.getenv("MODAL_TOKEN_SECRET"):
            sandbox_type = "modal"
        else:
            sandbox_type = "local"
    
    if sandbox_type == "modal":
        print("[Sandbox] Using Modal cloud sandbox")
        return create_modal_sandbox()
    else:
        workspace_dir = os.getenv("SANDBOX_WORKSPACE", "/workspace")
        print(f"[Sandbox] Using local sandbox at {workspace_dir}")
        return create_local_sandbox(working_dir=workspace_dir)
