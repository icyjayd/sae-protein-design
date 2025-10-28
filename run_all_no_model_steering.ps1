# -------------------------------
# Run all no_model_steering experiments (robust, no Invoke-Expression)
# -------------------------------

# === INPUTS (edit as needed) ===
$SequencesCSV  = "C:\Users\juani\Documents\Repos\proteng\model_scout_outs\split_ridge_onehot_10000.csv"
$RidgeModel    = "C:\Users\juani\Documents\Repos\proteng\model_scout_outs\models\ridge_onehot_10000.joblib"
$LatentCorrCSV = "C:\Users\juani\Documents\Repos\proteng\sae-protein-design\outputs\latent_property_correlation_mono.csv"
# If your file lives elsewhere, point to it here.

# === COMMON SETTINGS ===
$Property    = "fluorescence"
$Delta       = 0.4
$TopNLatents = 10
$NWorkers    = 4
$Seed        = 123
$OutDir      = "analysis_results"

# Mode-specific params
$ChosenPos = 37
$NRandom   = 10

function Run-Experiment {
    param(
        [Parameter(Mandatory=$true)][string]$Mode,
        [string[]]$ExtraArgs = @()
    )

    $argsList = @(
        "-m","sae.experiments.no_model_steering",
        "--sequences",$SequencesCSV,
        "--ridge_model",$RidgeModel,
        "--property",$Property,
        "--mode",$Mode,
        "--delta",$Delta,
        "--latent_ranking_csv",$LatentCorrCSV,
        "--ranking_column","spearman",
        "--top_n_latents",$TopNLatents,
        "--n_workers",$NWorkers,
        "--seed",$Seed,
        "--outdir",$OutDir
    ) + $ExtraArgs

    # Log a single-line command for easy copy/paste
    $display = "python " + ($argsList -join ' ')
    Write-Host "`n=== Running $Mode ===" -ForegroundColor Yellow
    Write-Host $display -ForegroundColor DarkGray

    # Execute without Invoke-Expression
    & python @argsList
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[$Mode] Process exited with code $LASTEXITCODE" -ForegroundColor Red
    }
}

Write-Host "Starting all no_model_steering experiments..." -ForegroundColor Cyan

# 1) Global latent (broadcast)
Run-Experiment -Mode "global_latent"

# 2) Random residue (repeat N times)
Run-Experiment -Mode "random_residue" -ExtraArgs @("--n_random", "$NRandom")

# 3) Chosen residue (fixed index)
Run-Experiment -Mode "chosen_residue" -ExtraArgs @("--chosen_pos", "$ChosenPos")

# 4) Filtered residues (top-k by activation)
Run-Experiment -Mode "filtered_residues" -ExtraArgs @("--filter_kind","topk","--filter_value","5")

Write-Host "`nAll experiments complete!" -ForegroundColor Green
