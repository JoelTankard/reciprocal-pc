$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$log = "D:/pfc_runs/pc_modal_lm/blue_exact_20k_to50k_queue.log"
$done = "D:/pfc_runs/pc_modal_lm/blue_exact_20k_to50k_queue.done.json"
Remove-Item $done -ErrorAction SilentlyContinue

$exitCode = 1
try {
  & (Join-Path $repo "tools/resume_message_pc_blue_exact_step20k_to50k_cuda_d.ps1") *>> $log
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
