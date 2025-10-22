import pandas as pd
from proteng_scout.plotting import make_plots
from proteng_scout.report import make_html_report
from pathlib import Path

def test_plot_and_report(tmp_path):
    df = pd.DataFrame({
        "model": ["ridge", "ridge", "rf"],
        "encoding": ["aac", "aac", "dpc"],
        "rho": [0.5, 0.6, 0.7],
        "p": [0.01, 0.02, 0.03],
        "n_samples": [10, 20, 30],
        "seconds": [1, 2, 3],
    })

    plots = make_plots(df, tmp_path / "plots")
    for p in plots.values():
        assert Path(p).exists()

    ranked = df.groupby(["model", "encoding"], as_index=False).agg({"rho": "max"})
    html = make_html_report(
        outdir=str(tmp_path),
        plots=plots,
        meta={"alpha": 0.01, "models": ["ridge"], "encodings": ["aac"], "sample_grid": [10], "n_jobs": 1, "task": "regression"},
        ranked_df=ranked,
        df_results=df,
    )
    assert Path(html).exists()
    assert html.endswith(".html")
