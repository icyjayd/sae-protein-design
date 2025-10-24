"""
Auto-installs the correct PyTorch build (CUDA, MPS, ROCm, or CPU)
during Poetry install. Also installs extra dependencies like interPLM
and the local model-scout submodule.
"""

import sys
import subprocess
import platform
import importlib
from pathlib import Path

TORCH_VERSION = "2.7.1"


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

def _repo_is_up_to_date(repo_dir: Path):
    """Return True if the local repo exists and is up to date with origin/HEAD."""
    if not repo_dir.exists():
        return False
    try:
        local_hash = subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
            text=True
        ).strip()
        remote_hash = subprocess.check_output(
            ["git", "-C", str(repo_dir), "ls-remote", "origin", "HEAD"],
            text=True
        ).split()[0]
        return local_hash == remote_hash
    except Exception:
        return False


def _is_editable_install(pkg_name: str, repo_dir: Path) -> bool:
    """Check if a package is already installed in editable mode from this directory."""
    try:
        import pkg_resources
        dist = pkg_resources.get_distribution(pkg_name)
        location = Path(dist.location)
        # For editable installs, "location" is the repo root
        return repo_dir.resolve() in location.resolve().parents or location.resolve() == repo_dir.resolve()
    except Exception:
        return False


def _install_extras():
    """Smart installer for interPLM and model-scout (only reinstall if missing or outdated)."""
    print("[extra-installer] Checking interPLM and model-scout...")

    repos = [
        {
            "name": "interplm",
            "pkg_name": "interplm",
            "url": "https://github.com/ElanaPearl/interPLM.git",
            "dir": Path.cwd() / "interPLM",
        },
        {
            "name": "model-scout",
            "pkg_name": "model-scout",
            "url": "https://github.com/icyjayd/model-scout.git",
            "dir": Path.cwd() / "model-scout",
        },
    ]

    for repo in repos:
        repo_dir = repo["dir"]
        pkg_name = repo["pkg_name"]
        url = repo["url"]

        try:
            if _is_editable_install(pkg_name, repo_dir) and _repo_is_up_to_date(repo_dir):
                print(f"[extra-installer] {pkg_name} is already installed and up to date — skipping.")
                continue

            if repo_dir.exists():
                if not _repo_is_up_to_date(repo_dir):
                    print(f"[extra-installer] Updating {pkg_name}...")
                    subprocess.check_call(["git", "-C", str(repo_dir), "pull"])
                else:
                    print(f"[extra-installer] {pkg_name} exists but not installed — installing in editable mode...")
            else:
                print(f"[extra-installer] Cloning {pkg_name} from {url}...")
                subprocess.check_call(["git", "clone", url, str(repo_dir)])

            subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", str(repo_dir)])
            print(f"[extra-installer] {pkg_name} installation complete.")

        except Exception as e:
            print(f"[extra-installer] Warning: failed to install {pkg_name}: {e}")

# from poetry.plugins.plugin import Plugin

# class SAEProteinDesignInstaller(Plugin):
#     def activate(self, poetry, io):
#         """Poetry automatically calls this when loading the plugin."""
#         print("[installer] Running SAE-protein-design install hooks...")
#         _install_torch_variant()
#         _install_extras()
#         print("[installer] Environment setup complete.")
