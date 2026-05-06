import docker
from docker.errors import DockerException, ContainerError
from app.utils.logger import setup_logger
from typing import Dict, Any, Optional
import time
import uuid

logger = setup_logger(__name__)


class CodeExecutor:
    """
    Secure code execution in Docker sandbox.
    
    Features:
    - Isolated execution environment
    - Resource limits (CPU, memory, time)
    - No network access
    - Read-only filesystem
    """
    
    # Docker image for code execution
    IMAGE = "python:3.11-slim"
    
    # Resource limits
    MEMORY_LIMIT = "512m"
    CPU_PERIOD = 100000
    CPU_QUOTA = 50000  # 50% of one CPU
    TIMEOUT_SECONDS = 30
    
    def __init__(self):
        try:
            self.client = docker.from_env()
            self._ensure_image()
        except DockerException as e:
            logger.error(f"Docker initialization failed: {e}")
            self.client = None
    
    def _ensure_image(self):
        """Ensure execution image is available."""
        try:
            self.client.images.get(self.IMAGE)
            logger.info(f"Docker image {self.IMAGE} is available")
        except docker.errors.ImageNotFound:
            logger.info(f"Pulling Docker image {self.IMAGE}...")
            self.client.images.pull(self.IMAGE)
            logger.info("Image pulled successfully")
    
    def execute_python(
        self,
        code: str,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Execute Python code in isolated sandbox.
        
        Args:
            code: Python code to execute
            timeout: Execution timeout in seconds
        
        Returns:
            Dictionary with output, errors, and execution info
        """
        if not self.client:
            return {
                "success": False,
                "output": "",
                "error": "Docker not available",
                "execution_time_ms": 0
            }
        
        timeout = timeout or self.TIMEOUT_SECONDS
        execution_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Executing code [{execution_id}]: {len(code)} chars")
        
        start_time = time.time()
        
        container = None

        try:
            # Create container
            container = self.client.containers.run(
                image=self.IMAGE,
                command=["python", "-c", code],
                detach=True,
                mem_limit=self.MEMORY_LIMIT,
                cpu_period=self.CPU_PERIOD,
                cpu_quota=self.CPU_QUOTA,
                network_disabled=True,
                read_only=True,
                remove=False,
                stdout=True,
                stderr=True
            )
            
            # Wait for completion with timeout
            exit_code = container.wait(timeout=timeout)
            
            # Get output
            output = container.logs(stdout=True, stderr=False).decode('utf-8')
            error = container.logs(stdout=False, stderr=True).decode('utf-8')
            
            execution_time = (time.time() - start_time) * 1000
            
            success = exit_code['StatusCode'] == 0
            
            logger.info(
                f"Code execution [{execution_id}] "
                f"{'succeeded' if success else 'failed'} "
                f"in {execution_time:.0f}ms"
            )
            
            return {
                "success": success,
                "output": output.strip(),
                "error": error.strip() if error else None,
                "execution_time_ms": execution_time,
                "exit_code": exit_code['StatusCode']
            }
            
        except ContainerError as e:
            logger.error(f"Container error [{execution_id}]: {e}")
            return {
                "success": False,
                "output": "",
                "error": f"Container error: {str(e)}",
                "execution_time_ms": (time.time() - start_time) * 1000
            }
            
        except Exception as e:
            logger.error(f"Execution error [{execution_id}]: {e}")
            return {
                "success": False,
                "output": "",
                "error": str(e),
                "execution_time_ms": (time.time() - start_time) * 1000
            }
        finally:
            if container is not None:
                try:
                    container.remove(force=True)
                except DockerException as e:
                    logger.warning(f"Failed to remove container [{execution_id}]: {e}")


def get_code_executor() -> CodeExecutor:
    """Get code executor instance."""
    return CodeExecutor()
