import subprocess, json, sys
from pathlib import Path

def test_cli_train_runs(tmp_seq_file, tmp_path):
    out = tmp_path / "cli_out"
    cmd = [
        sys.executable,  # ✅ use current env's python
        "-m", "ml_models.cli_train",
        str(tmp_seq_file),
        "--model", "rf",
        "--encoding", "aac",
        "--task", "regression",
        "--outdir", str(out)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    assert "r2" in data or "accuracy" in data
