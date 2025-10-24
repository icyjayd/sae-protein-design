"""
Auto-installs the correct PyTorch build (CUDA, MPS, ROCm, or CPU)
during Poetry install. Reinstalls if the wrong variant is detected.
"""

import sys
import subprocess
import platform
import importlib

TORCH_VERSION = "2.4.0"

def _run(cmd):
    """Run shell command safely."""
    print(f"[torch-installer] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)

def _uninstall_torch():
    try:
        _run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"])
    except Exception as e:
        print(f"[torch-installer] Warning: failed to uninstall existing torch: {e}")

def _detect_env():
    """Detects best-fit accelerator environment."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    # --- Check for NVIDIA GPU ---
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "cuda"
    except Exception:
        pass

    # --- Check for ROCm (AMD) ---
    try:
        subprocess.run(["rocminfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "rocm"
    except Exception:
        pass

    # --- Check for Apple M-series (Metal/MPS backend) ---
    if system == "darwin" and "arm" in machine:
        return "mps"

    # --- Default to CPU ---
    return "cpu"

def _torch_variant_installed():
    """Detect what torch variant is currently installed, if any."""
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        elif hasattr(torch.version, "hip") and torch.version.hip is not None:
            return "rocm"
        else:
            return "cpu"
    except Exception:
        return None

def _install_torch(target):
    """Install the appropriate torch build."""
    if target == "cuda":
        url = "https://download.pytorch.org/whl/cu121"
        wheel = f"torch=={TORCH_VERSION}+cu121"
    elif target == "rocm":
        url = "https://download.pytorch.org/whl/rocm6.1"
        wheel = f"torch=={TORCH_VERSION}+rocm6.1"
    elif target == "mps":
        # Apple M-series uses default Mac wheels (CPU + MPS support baked in)
        url = "https://download.pytorch.org/whl/cpu"
        wheel = f"torch=={TORCH_VERSION}"
    else:  # CPU fallback
        url = "https://download.pytorch.org/whl/cpu"
        wheel = f"torch=={TORCH_VERSION}"

    print(f"[torch-installer] Installing {wheel} from {url}")
    _run([sys.executable, "-m", "pip", "install", wheel, "--index-url", url])

def _main():
    target = _detect_env()
    installed = _torch_variant_installed()

    print(f"[torch-installer] Detected environment: {target}")
    print(f"[torch-installer] Currently installed: {installed or 'none'}")

    if installed != target:
        if installed:
            print(f"[torch-installer] Uninstalling mismatched torch ({installed})...")
            _uninstall_torch()
        print(f"[torch-installer] Installing correct torch variant ({target})...")
        _install_torch(target)
    else:
        print(f"[torch-installer] Correct torch variant ({target}) already installed — skipping.")

if __name__ == "__main__":
    _main()

# Run automatically when imported by Poetry build system
_main()
