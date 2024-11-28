import subprocess
import psutil
import socket
import time
import os


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
                print(f"[Re-deployment] Server is running on port {port}.")
                return
        except (socket.timeout, ConnectionRefusedError):
            time.sleep(1)  # Retry after a short delay
    raise TimeoutError(f"Server did not start on port {port} within {timeout} seconds.")


import os


def stop_server_and_clean_resources(port: int, cuda_device: int = 1, retry_attempts: int = 5, retry_delay: int = 2):
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
            print(f"[Re-deployment] No process found on port {port}.")
        else:
            for pid in pids:
                try:
                    if int(pid) == current_pid:
                        print(f"[Re-deployment] Skipping termination of the current process (PID: {current_pid}).")
                        continue
                    print(f"[Re-deployment] Attempting to terminate process with PID: {pid} on port {port}")
                    process = psutil.Process(int(pid))
                    process.terminate()  # Send SIGTERM

                    try:
                        process.wait(timeout=15)  # Increased timeout to allow draining requests
                        print(f"[Re-deployment] Process (PID: {pid}) terminated successfully.")
                    except psutil.TimeoutExpired:
                        print(f"[Re-deployment] Process (PID: {pid}) did not terminate in time. Sending SIGKILL.")
                        process.kill()  # Send SIGKILL
                        process.wait(timeout=5)
                        print(f"[Re-deployment] Process (PID: {pid}) forcefully terminated.")
                except psutil.NoSuchProcess:
                    print(f"[Re-deployment] Process (PID: {pid}) no longer exists.")
                except Exception as e:
                    print(f"[Re-deployment] Error terminating process with PID {pid}: {e}")

        # Verify the port is cleared
        for attempt in range(retry_attempts):
            print(f"[Re-deployment] Verifying port {port} is cleared (attempt {attempt + 1}/{retry_attempts})")
            result = subprocess.run(["lsof", "-t", f"-i:{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            remaining_pids = result.stdout.decode().strip().splitlines()

            if not remaining_pids:
                print(f"[Re-deployment] Port {port} is now cleared.")
                break
            else:
                print(f"[Re-deployment] Processes still running on port {port}: {', '.join(remaining_pids)}")
                time.sleep(retry_delay)
        else:
            raise RuntimeError(f"[Re-deployment] Failed to clear port {port} after {retry_attempts} attempts.")

    except subprocess.SubprocessError as e:
        print(f"[Re-deployment] Error running lsof command: {e}")
    except Exception as e:
        print(f"[Re-deployment] Unexpected error stopping server on port {port}: {e}")

    # Clean up GPU resources
    try:
        print(f"[Re-deployment] Cleaning up GPU resources on device {cuda_device}")
        gpu_processes = []
        result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        gpu_pids = result.stdout.decode().strip().splitlines()

        for pid in gpu_pids:
            try:
                if int(pid) == current_pid:
                    print(f"[Re-deployment] Skipping termination of the current process (PID: {current_pid}).")
                    continue
                process = psutil.Process(int(pid))
                print(f"[Re-deployment] Found GPU-bound process (PID: {pid}), attempting to terminate.")
                process.terminate()
                try:
                    process.wait(timeout=10)
                    print(f"[Re-deployment] GPU-bound process (PID: {pid}) terminated.")
                except psutil.TimeoutExpired:
                    print(f"[Re-deployment] GPU process (PID: {pid}) did not terminate in time. Sending SIGKILL.")
                    process.kill()
                    print(f"[Re-deployment] GPU-bound process (PID: {pid}) forcefully terminated.")
                gpu_processes.append(pid)
            except psutil.NoSuchProcess:
                print(f"[Re-deployment] GPU-bound process (PID: {pid}) no longer exists.")
            except Exception as e:
                print(f"[Re-deployment] Error terminating GPU-bound process (PID: {pid}): {e}")

        if gpu_processes:
            print(f"[Re-deployment] Cleaned up GPU processes: {', '.join(gpu_processes)}")
        else:
            print(f"[Re-deployment] No GPU-bound processes found for device {cuda_device}.")

    except Exception as e:
        print(f"[Re-deployment] Unexpected error while cleaning up GPU resources: {e}")


def redeploy_sglang_model(model_path: str, port: int = 7501, cuda_device: int = 1,
                          log_file: str = "trained_llama_run.log"):
    """
    Re-deploy the fine-tuned model with SGLang.

    Args:
        model_path (str): Path to the fine-tuned model directory.
        port (int): Port for the SGLang server.
        cuda_device (int): CUDA device ID to use.
        log_file (str): Log file for the server output.
    """
    print(f"[Re-deployment] Stopping any existing SGLang server on port {port}...")
    stop_server_and_clean_resources(port, cuda_device)

    print("[Re-deployment] Deploying the fine-tuned model...")
    command = f"nohup env CUDA_VISIBLE_DEVICES={cuda_device} python -m sglang.launch_server " \
              f"--model-path {model_path} --port {port} > {log_file} 2>&1 &"
    subprocess.Popen(command, shell=True)

    # Wait for the server to start
    wait_for_server(port)
    print("[Re-deployment] Deployment completed.")
