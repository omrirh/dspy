import subprocess
import psutil
import socket
import time


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


def stop_server_on_port(port: int):
    """
    Stop any server running on the specified port.

    Args:
        port (int): Port number to stop the server on.
    """
    try:
        result = subprocess.run(["lsof", "-t", f"-i:{port}"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        pid = result.stdout.decode().strip()
        if pid:
            process = psutil.Process(int(pid))
            process.terminate()  # Gracefully terminate the process
            process.wait(timeout=10)  # Wait for the process to exit
            print(f"[Re-deployment] SGLang server process (PID: {pid}) terminated.")
    except psutil.NoSuchProcess:
        print(f"[Re-deployment] No process found on port {port}.")
    except Exception as e:
        print(f"[Re-deployment] Error terminating process on port {port}: {e}")


def redeploy_sglang_model(model_path: str, port: int = 7501, cuda_device: int = 1, log_file: str = "llama_run.log"):
    """
    Re-deploy the fine-tuned model with SGLang.

    Args:
        model_path (str): Path to the fine-tuned model directory.
        port (int): Port for the SGLang server.
        cuda_device (int): CUDA device ID to use.
        log_file (str): Log file for the server output.
    """
    print(f"[Re-deployment] Stopping any existing SGLang server on port {port}...")
    stop_server_on_port(port)

    print("[Re-deployment] Deploying the fine-tuned model...")
    command = f"nohup env CUDA_VISIBLE_DEVICES={cuda_device} python -m sglang.launch_server " \
              f"--model-path {model_path} --port {port} > {log_file} 2>&1 &"
    subprocess.Popen(command, shell=True)

    # Wait for the server to start
    wait_for_server(port)
    print("[Re-deployment] Deployment completed.")
