import subprocess
import socket
import time
import os
import psutil
import logging
from dspy.clients.provider import Provider
from dspy.clients.huggingface import HFProvider

logger = logging.getLogger(__name__)


def assign_local_lm(
        model: str,
        api_base: str,
        provider: Provider,
):
    import dspy
    lm = dspy.LM(
        model=model,
        api_base=api_base,
        api_key="local",
        provider=provider,
    )
    dspy.configure(lm=lm)


def get_sglang_process():
    """Fetch the process instance of the sglang server running on localhost."""
    for process in psutil.process_iter(attrs=['pid', 'name', 'cmdline']):
        try:
            cmdline = process.info.get('cmdline', [])
            if cmdline and any("sglang" in arg for arg in cmdline):
                return process
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


def wait_for_server(port: int, timeout: int = 60):
    """
    Wait for a server to start listening on the specified port.

    Args:
        port (int): Port number to check.
        timeout (int): Maximum time to wait in seconds.

    Raises:
        TimeoutError: If the server does not start within the timeout period.
    """
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(("localhost", port), timeout=5):
                logger.info(f"Server is running on port {port}.")
                return
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(1)  # Retry after a short delay
    raise TimeoutError(f"Server did not start on port {port} within {timeout} seconds.")


def stop_server_and_clean_resources(port: int, cuda_device: int = 0, retry_attempts: int = 5, retry_delay: int = 2):
    """
    Stop any server running on the specified port and clean up GPU resources.

    Args:
        port (int): Port number to stop the server on.
        cuda_device (int): CUDA device ID to clean up processes.
        retry_attempts (int): Number of attempts to verify the port and GPU are cleared.
        retry_delay (int): Delay in seconds between verification attempts.

    Raises:
        RuntimeError: If the server processes or GPU resources cannot be cleared.
    """
    current_pid = os.getpid()  # Get the PID of the current process

    # Stop any server processes on the port
    try:
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pids = result.stdout.decode().strip().splitlines()

        if not pids:
            logger.info(f"No process found on port {port}.")
        else:
            for pid in pids:
                try:
                    if int(pid) == current_pid:
                        logger.info(f"Skipping termination of the current process (PID: {current_pid}).")
                        continue
                    logger.info(f"Attempting to terminate process with PID: {pid} on port {port}")
                    process = psutil.Process(int(pid))
                    process.terminate()  # Send SIGTERM

                    try:
                        process.wait(timeout=15)  # Increased timeout to allow draining requests
                        logger.info(f"Process (PID: {pid}) terminated successfully.")
                    except psutil.TimeoutExpired:
                        logger.info(f"Process (PID: {pid}) did not terminate in time. Sending SIGKILL.")
                        process.kill()  # Send SIGKILL
                        process.wait(timeout=5)
                        logger.info(f"Process (PID: {pid}) forcefully terminated.")
                except psutil.NoSuchProcess:
                    logger.info(f"Process (PID: {pid}) no longer exists.")
                except Exception as e:
                    logger.info(f"Error terminating process with PID {pid}: {e}")

        # Verify the port is cleared
        for attempt in range(retry_attempts):
            logger.info(f"Verifying port {port} is cleared (attempt {attempt + 1}/{retry_attempts})")
            result = subprocess.run(["lsof", "-t", f"-i:{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            remaining_pids = result.stdout.decode().strip().splitlines()

            if not remaining_pids:
                logger.info(f"Port {port} is now cleared.")
                break
            else:
                logger.info(f"Processes still running on port {port}: {', '.join(remaining_pids)}")
                time.sleep(retry_delay)
        else:
            raise RuntimeError(f"[Re-deployment] Failed to clear port {port} after {retry_attempts} attempts.")

    except subprocess.SubprocessError as e:
        logger.info(f"Error running lsof command: {e}")
    except Exception as e:
        logger.info(f"Unexpected error stopping server on port {port}: {e}")

    # Clean up GPU resources
    try:
        logger.info(f"Cleaning up GPU resources on device {cuda_device}")
        gpu_processes = []
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_pids = result.stdout.decode().strip().splitlines()

        for pid in gpu_pids:
            try:
                if int(pid) == current_pid:
                    logger.info(f"Skipping termination of the current process (PID: {current_pid}).")
                    continue
                process = psutil.Process(int(pid))
                logger.info(f"Found GPU-bound process (PID: {pid}), attempting to terminate.")
                process.terminate()
                try:
                    process.wait(timeout=10)
                    logger.info(f"GPU-bound process (PID: {pid}) terminated.")
                except psutil.TimeoutExpired:
                    logger.info(f"GPU process (PID: {pid}) did not terminate in time. Sending SIGKILL.")
                    process.kill()
                    logger.info(f"GPU-bound process (PID: {pid}) forcefully terminated.")
                gpu_processes.append(pid)
            except psutil.NoSuchProcess:
                logger.info(f"GPU-bound process (PID: {pid}) no longer exists.")
            except Exception as e:
                logger.info(f"Error terminating GPU-bound process (PID: {pid}): {e}")

        if gpu_processes:
            logger.info(f"Cleaned up GPU processes: {', '.join(gpu_processes)}")
        else:
            logger.info(f"No GPU-bound processes found for device {cuda_device}.")

    except Exception as e:
        logger.info(f"Unexpected error while cleaning up GPU resources: {e}")


def deploy_sglang_model(model_path: str, log_file: str, port: int = 7501, cuda_device: int = 0):
    """
    Re-deploy the fine-tuned model with SGLang.

    Args:
        model_path (str): Path to the fine-tuned model directory.
        port (int): Port for the SGLang server.
        cuda_device (int): CUDA device ID to use.
        log_file (str): Log file for the server output.
    """
    logger.info(f"Deploying {model_path} model...")
    command = f"nohup env CUDA_VISIBLE_DEVICES={cuda_device} python -m sglang.launch_server " \
              f"--model-path {model_path} --port {port} > {log_file} 2>&1 &"
    subprocess.Popen(command, shell=True)

    # Wait for the server to start
    wait_for_server(port)
    logger.info(f"Deployment completed.")


if __name__ == "__main__":
    sglang_process = get_sglang_process()
    if sglang_process:
        logger.info(f"sglang server is running with PID: {sglang_process.pid}")
    else:
        logger.info(f"sglang server is not running.")
