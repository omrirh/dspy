"""
Remote setup utilities: LM assignment, SGLang server management.
"""
import logging
import socket
import subprocess
import time

import psutil

logger = logging.getLogger(__name__)


def assign_local_lm(model: str, api_base: str, api_key: str = "local", provider=None):
    """Configure dspy.settings with a locally served SGLang model."""
    import dspy

    kwargs = dict(model=model, api_base=api_base, api_key=api_key)
    if provider is not None:
        kwargs["provider"] = provider
    lm = dspy.LM(**kwargs)
    dspy.configure(lm=lm)
    return lm


def get_sglang_process():
    """Return the psutil.Process for the running SGLang server, or None."""
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("sglang" in arg for arg in cmdline):
                return proc
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    return None


def is_server_up(host: str = "localhost", port: int = 30000, timeout: int = 3) -> bool:
    """Return True if a server is already listening on host:port."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def wait_for_server(port: int, timeout: int = 180):
    """Block until the server at *port* accepts connections, or raise TimeoutError."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if is_server_up(port=port):
            logger.info(f"Server is ready on port {port}.")
            return
        time.sleep(1)
    raise TimeoutError(f"Server on port {port} did not start within {timeout}s.")


def stop_server_and_clean_resources(port: int = 30000):
    """Kill the SGLang server listening on *port* and free GPU memory."""
    import gc
    import torch

    proc = get_sglang_process()
    if proc:
        logger.info(f"Stopping SGLang server (PID {proc.pid}) on port {port}")
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except psutil.TimeoutExpired:
            logger.warning("Server did not terminate gracefully; killing it.")
            proc.kill()
        logger.info("SGLang server stopped.")
    else:
        logger.info("No running SGLang server found.")

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("GPU memory and resources cleaned up.")


def deploy_sglang_model(
    model_path: str,
    log_file: str = "sglang_run.log",
    port: int = 30000,
    cuda_device: int = 0,
):
    """Launch an SGLang server for *model_path* in the background."""
    logger.info(f"Deploying {model_path} on port {port} (CUDA:{cuda_device}) ...")

    extra_flags = ""
    if "Qwen3" in model_path:
        extra_flags = "--reasoning-parser qwen3"
    elif "gemma-3" in model_path:
        extra_flags = "--context-length 8192"

    cmd = (
        f"nohup env CUDA_VISIBLE_DEVICES={cuda_device} "
        f"CUDA_HOME=/usr/local/cuda-12.4 "
        f"PATH=/usr/local/cuda-12.4/bin:$PATH "
        f"LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH "
        f"python -m sglang.launch_server "
        f"--model-path {model_path} --port {port} {extra_flags} "
        f"> {log_file} 2>&1 &"
    )
    subprocess.Popen(cmd, shell=True)
    wait_for_server(port)
    logger.info("SGLang deployment complete.")


if __name__ == "__main__":
    proc = get_sglang_process()
    if proc:
        print(f"SGLang server running: PID {proc.pid}")
    else:
        print("SGLang server not running.")
