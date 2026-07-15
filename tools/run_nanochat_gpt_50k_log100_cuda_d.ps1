param(
  [string]$NanochatRoot = "C:/Users/ai/nanochat",
  [string]$PfcRoot = (Split-Path -Parent $PSScriptRoot),
  [string]$BaseDir = "C:/Users/ai/nanochat_pfc",
  [string]$RunRoot = "D:/pfc_runs/pc_modal_lm/gpt_50k_schedule_log100_20260716",
  [string]$ModelTag = "pfc_d4_seq1024_50k_schedule_log100_20260716"
)

$ErrorActionPreference = "Stop"
$Python = Join-Path $PfcRoot ".venv/Scripts/python.exe"
$CheckpointDir = Join-Path $RunRoot "checkpoints/$ModelTag"
$CheckpointLink = Join-Path $BaseDir "base_checkpoints/$ModelTag"
$Results = Join-Path $RunRoot "results.jsonl"
$TrainLog = Join-Path $RunRoot "train.log"
$TrainErr = Join-Path $RunRoot "train.err"
$LauncherLog = Join-Path $RunRoot "launcher.log"
$Completion = Join-Path $RunRoot "completion.json"
$RunTag = "nanochat_official_d4_seq1024_50000_bpe4096_mix8"

if (Test-Path $Completion) {
  throw "Refusing to overwrite completed run: $RunRoot"
}
if ((Test-Path $CheckpointDir) -and (Get-ChildItem $CheckpointDir -Force).Count -gt 0) {
  throw "Refusing to overwrite checkpoint directory: $CheckpointDir"
}
if (Test-Path $CheckpointLink) {
  throw "Refusing to replace checkpoint path: $CheckpointLink"
}

New-Item -ItemType Directory -Force $RunRoot, $CheckpointDir | Out-Null
"$(Get-Date -Format o) run directory ready" | Set-Content $LauncherLog -Encoding UTF8
New-Item -ItemType Junction -Path $CheckpointLink -Target $CheckpointDir | Out-Null
"$(Get-Date -Format o) checkpoint junction ready" | Add-Content $LauncherLog
Remove-Item $Results, $TrainLog, $TrainErr -Force -ErrorAction SilentlyContinue

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:TORCH_COMPILE_DISABLE = "1"
$env:NANOCHAT_BASE_DIR = $BaseDir
$env:NANOCHAT_DTYPE = "bfloat16"
$env:PYTHONPATH = "$NanochatRoot;$PfcRoot"
$env:PFC_PROGRESS_RESULTS = $Results
$env:PFC_PROGRESS_RUN_TAG = $RunTag
$env:PFC_PROGRESS_CHECKPOINT_DIR = $CheckpointDir
$env:PFC_PROGRESS_EVAL_EVERY = "100"
$env:PFC_PROGRESS_TOTAL_STEPS = "50000"
$env:PFC_PROGRESS_TOTAL_BATCH_SIZE = "2048"
$env:PFC_PROGRESS_SEQ_LEN = "1024"
"$(Get-Date -Format o) environment ready" | Add-Content $LauncherLog

$trainArgs = @(
  "-u", (Join-Path $PfcRoot "tools/run_nanochat_base_train_progress.py"),
  "--depth=4",
  "--aspect-ratio=64",
  "--head-dim=64",
  "--max-seq-len=1024",
  "--window-pattern=L",
  "--device-batch-size=2",
  "--total-batch-size=2048",
  "--num-iterations=50000",
  "--eval-every=100",
  "--eval-tokens=102400",
  "--core-metric-every=-1",
  "--sample-every=-1",
  "--save-every=1000",
  "--run=dummy",
  "--model-tag=$ModelTag"
)

$startedAt = Get-Date
"$(Get-Date -Format o) starting Python" | Add-Content $LauncherLog
$process = Start-Process `
  -FilePath $Python `
  -ArgumentList $trainArgs `
  -WorkingDirectory $PfcRoot `
  -RedirectStandardOutput $TrainLog `
  -RedirectStandardError $TrainErr `
  -Wait `
  -PassThru
$exitCode = $process.ExitCode
$finishedAt = Get-Date
"$(Get-Date -Format o) Python finished exit=$exitCode" | Add-Content $LauncherLog

if ($exitCode -eq 0) {
  $finalizeArgs = @(
    (Join-Path $PfcRoot "tools/finalize_nanochat_progress_results.py"),
    "--results", $Results,
    "--meta", (Join-Path $CheckpointDir "meta_050000.json")
  )
  $finalizeProcess = Start-Process `
    -FilePath $Python `
    -ArgumentList $finalizeArgs `
    -WorkingDirectory $PfcRoot `
    -Wait `
    -PassThru
  if ($finalizeProcess.ExitCode -ne 0) {
    throw "Failed to finalize training timing in $Results"
  }
  "$(Get-Date -Format o) results timing finalized" | Add-Content $LauncherLog
}

$completionData = @{
  checkpoint_dir = $CheckpointDir
  elapsed_seconds = ($finishedAt - $startedAt).TotalSeconds
  exit_code = $exitCode
  finished_at = $finishedAt.ToString("o")
  results = $Results
  run_tag = $RunTag
  started_at = $startedAt.ToString("o")
}
$completionData | ConvertTo-Json | Set-Content $Completion -Encoding UTF8

if ($exitCode -ne 0) {
  exit $exitCode
}
