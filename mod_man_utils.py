import os
import sys
import inspect

def add_module(module_name: str = None, repo_root_name: str = None):
    """
    Adds a given module (or the repo root) to sys.path dynamically,
    regardless of where this is called from.

    Args:
        module_name (str, optional): Name of the module folder to add (e.g. 'sae').
        repo_root_name (str, optional): Name of the repo’s top-level folder.
                                        If None, this is inferred by searching upward
                                        until a `.git` or `pyproject.toml` is found.

    Example:
        >>> import module_management_utils as mmu
        >>> mmu.add_module('sae')
        >>> from sae.utils import esm_utils
    """
    # --- Step 1: determine caller’s directory ---
    caller_frame = inspect.stack()[1]
    caller_dir = os.path.dirname(os.path.abspath(caller_frame.filename))

    # --- Step 2: walk upward to find repo root ---
    current_dir = caller_dir
    while current_dir and current_dir != os.path.dirname(current_dir):
        if repo_root_name and os.path.basename(current_dir) == repo_root_name:
            repo_root = current_dir
            break
        if any(os.path.exists(os.path.join(current_dir, marker))
               for marker in [".git", "pyproject.toml", "setup.py"]):
            repo_root = current_dir
            break
        current_dir = os.path.dirname(current_dir)
    else:
        repo_root = caller_dir  # fallback

    # --- Step 3: decide which path to add ---
    if module_name:
        path_to_add = os.path.join(repo_root, module_name)
    else:
        path_to_add = repo_root

    # --- Step 4: add to sys.path if not already present ---
    path_to_add = os.path.abspath(path_to_add)
    if path_to_add not in sys.path:
        sys.path.insert(0, path_to_add)
        print(f"[module_management_utils] Added to sys.path: {path_to_add}")
    return path_to_add
