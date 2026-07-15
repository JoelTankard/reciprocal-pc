$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$runs = @(
  @{ Name = "frozenroute"; RouteLr = 0.0 },
  @{ Name = "learnroute"; RouteLr = 1.0 }
)

foreach ($run in $runs) {
  & (Join-Path $repo "tools/run_message_pc_nanochat_exact_cuda_d.ps1") `
    -Steps 100 `
    -LogInterval 100 `
    -CheckpointInterval 100 `
    -HiddenDim 320 `
    -Layers 4 `
    -Heads 4 `
    -InferenceSteps 4 `
    -Seed 2 `
    -LearningRate 0.001 `
    -DecoderLrScale 1.0 `
    -StateLrScale 1.0 `
    -EmbedLrScale 14.0 `
    -InferenceRate 0.03 `
    -TokenFeedbackScale 0.1 `
    -RouteUpdateMode "local_energy" `
    -RouteLrScale $run.RouteLr `
    -FeedbackLrScale 0.0 `
    -GenerativeLrScale 0.0 `
    -DecoderSupervisionMode "top_only" `
    -DecoderStateMode "rms_norm" `
    -ReadoutLayer 0 `
    -ReadoutObjectiveShare 0.0 `
    -BlockStateMode "identity_residual" `
    -ResidualScale 0.25 `
    -SelfInitGain 0.02 `
    -MessageInitGain 1.0 `
    -InferenceMode "pc_energy" `
    -LearningPhaseSource "settled" `
    -DecoderLearningSource "free" `
    -LearnedFeedbackProposalScale 0.0 `
    -EnergyBacktrackingSteps 16 `
    -MatrixOptimizer "muon" `
    -OptimizerWarmupSteps 40 `
    -MuonLrScale 16.0 `
    -MuonMomentum 0.85 `
    -MuonMomentumFinal 0.97 `
    -MuonMomentumWarmupSteps 400 `
    -MuonBeta2 0.9 `
    -MuonNsSteps 5 `
    -RunSuffix "exactcredit_b01_p0_$($run.Name)_seed2"
  if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
  }
}

exit 0
