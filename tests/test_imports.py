import pytest

def test_project_imports():
    """Ensure core modules import cleanly."""
    import agentic_lab.tools
    import agentic_adapter
    import sae.utils.grade_reconstructions
    assert True
