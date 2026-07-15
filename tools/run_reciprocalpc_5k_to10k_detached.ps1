$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$log = "D:/pfc_runs/pc_modal_lm/stage_reciprocal_5k_to10k_queue.log"
$done = "D:/pfc_runs/pc_modal_lm/stage_reciprocal_5k_to10k_queue.done.json"
Remove-Item $done -ErrorAction SilentlyContinue

$exitCode = 1
try {
  & (Join-Path $repo "tools/resume_reciprocalpc_cuda_d.ps1") `
    -Parent "message_pc_d256_l4_i1_meta_pc_stage_reciprocal_lr100_d256_f1022_seed2_resume2k_5k_5000" `
    -ResumeStep 5000 `
    -Steps 10000 `
    -RunSuffix "meta_pc_stage_reciprocal_lr100_d256_f1022_seed2_resume5k_10k" `
    -LogInterval 100 `
    -CheckpointInterval 1000 `
    *>> $log
  $exitCode = $LASTEXITCODE
} catch {
  $_ | Out-String | Add-Content $log
  $exitCode = 1
}

@{
  completed_at = (Get-Date).ToString("o")
  exit_code = $exitCode
} | ConvertTo-Json | Set-Content $done

exit $exitCode
