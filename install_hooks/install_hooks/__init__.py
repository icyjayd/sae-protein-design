"""
Auto-installs the correct PyTorch build (CUDA, MPS, ROCm, or CPU)
during Poetry install. Also installs extra dependencies like interPLM
and the local model-scout submodule.
"""

import sys
import subprocess
import platform
import importlib
import os

TORCH_VERSION = "2.4.0"


def _run(cmd):
    """Run shell command safely."""
    print(f"[installer] Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


# ---------------------------------------------------------------------
# TORCH INSTALLER
# ---------------------------------------------------------------------
def _uninstall_torch():
    try:
        _run([sys.executable, "-m", "pip", "uninstall", "-y", "torch"])
    except Exception as e:
        print(f"[torch-installer] Warning: failed to uninstall existing torch: {e}")


def _detect_env():
    """Detects best-fit accelerator environment."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "cuda"
    except Exception:
        pass

    try:
        subprocess.run(["rocminfo"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return "rocm"
    except Exception:
        pass

    if system == "darwin" and "arm" in machine:
        return "mps"

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
        url = "https://download.pytorch.org/whl/cpu"
        wheel = f"torch=={TORCH_VERSION}"
    else:
        url = "https://download.pytorch.org/whl/cpu"
        wheel = f"torch=={TORCH_VERSION}"

    print(f"[torch-installer] Installing {wheel} from {url}")
    _run([sys.executable, "-m", "pip", "install", wheel, "--index-url", url])


def _install_torch_variant():
    """Main logic for installing correct torch variant."""
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


# ---------------------------------------------------------------------
# EXTRAS INSTALLER (interPLM + model-scout)
# ---------------------------------------------------------------------
def _install_extras():
    """Install additional dependencies and local submodules (interPLM, model-scout)."""
    repo_url = "https://github.com/ElanaPearl/interPLM.git"
    repo_dir = os.path.join(os.getcwd(), "interPLM")

    # --- Install interPLM ---
    try:
        if not os.path.exists(repo_dir):
            print(f"[extra-installer] Cloning interPLM from {repo_url}...")
            _run(["git", "clone", repo_url, repo_dir])
        else:
            print("[extra-installer] interPLM repo already exists — pulling latest changes...")
            _run(["git", "-C", repo_dir, "pull"])

        print("[extra-installer] Installing interPLM in editable mode...")
        _run([sys.executable, "-m", "pip", "install", "-e", repo_dir])

    except Exception as e:
        print(f"[extra-installer] Warning: failed to install interPLM: {e}")

    # --- Install model-scout submodule ---
    try:
        print("[extra-installer] Installing local submodule model-scout...")
        _run([sys.executable, "-m", "pip", "install", "-e", "model-scout"])
    except Exception as e:
        print(f"[extra-installer] Warning: failed to install model-scout: {e}")

from poetry.plugins.plugin import Plugin

class SAEProteinDesignInstaller(Plugin):
    def activate(self, poetry, io):
        """Poetry automatically calls this when loading the plugin."""
        print("[installer] Running SAE-protein-design install hooks...")
        _install_torch_variant()
        _install_extras()
        print("[installer] Environment setup complete.")
