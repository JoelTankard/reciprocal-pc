param(
  [Parameter(Mandatory = $true)][string]$Parent,
  [Parameter(Mandatory = $true)][int]$ResumeStep,
  [Parameter(Mandatory = $true)][int]$Steps,
  [Parameter(Mandatory = $true)][string]$RunSuffix,
  [int]$LogInterval = 100,
  [int]$CheckpointInterval = 1000
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$stepTag = $ResumeStep.ToString("D6")
$checkpoint = "D:/pfc_runs/pc_modal_lm/$Parent/message_passing_pc_lm_${Parent}_step${stepTag}.pt"
if (-not (Test-Path $checkpoint)) {
  throw "Missing checkpoint: $checkpoint"
}

& (Join-Path $repo "tools/run_message_pc_nanochat_exact_cuda_d.ps1") `
  -Steps $Steps `
  -LogInterval $LogInterval `
  -CheckpointInterval $CheckpointInterval `
  -HiddenDim 256 `
  -Layers 4 `
  -Heads 4 `
  -InferenceSteps 1 `
  -InferenceRate 1.0 `
  -Seed 2 `
  -LearningRate 0.001 `
  -DecoderLrScale 1.0 `
  -StateLrScale 1.0 `
  -EmbedLrScale 14.0 `
  -TokenFeedbackScale 1.0 `
  -RouteUpdateMode "local_energy" `
  -RouteLrScale 0.1 `
  -FeedbackLrScale 0.0 `
  -GenerativeLrScale 0.0 `
  -DecoderSupervisionMode "top_only" `
  -DecoderStateMode "rms_norm" `
  -ReadoutLayer 0 `
  -BlockStateMode "nanochat_residual" `
  -FfnDim 1022 `
  -FfnLrScale 1.0 `
  -FfnLearningSource "phase" `
  -TokenValueEmbeddings `
  -TokenValueScale 1.0 `
  -TokenValueLrScale 0.5 `
  -SensoryReinjectionStart 0.2 `
  -SensoryReinjectionEnd 0.05 `
  -SensoryReinjectionMode "pre_block" `
  -ResidualStreamScaleStart 1.15 `
  -ResidualStreamScaleEnd 1.05 `
  -LearnResidualStreamScales `
  -ResidualStreamLrScale 1.0 `
  -InferenceMode "meta_pc" `
  -MetaFeedbackMode "stage_reciprocal" `
  -MetaReciprocalLearningRate 1.0 `
  -MetaReciprocalNoiseStd 0.01 `
  -MetaReciprocalUpdateBeforeSettle `
  -MetaOutputFeedbackMode "kp_learned" `
  -MetaOutputFeedbackLrScale 1.0 `
  -LearningPhaseSource "settled" `
  -ForwardUpdateTopLayers 4 `
  -DecoderLearningSource "free" `
  -MatrixOptimizer "muon" `
  -OptimizerWarmupSteps 40 `
  -MuonLrScale 16.0 `
  -MuonMomentum 0.85 `
  -MuonMomentumFinal 0.97 `
  -MuonMomentumWarmupSteps 400 `
  -MuonBeta2 0.9 `
  -MuonNsSteps 5 `
  -ResumeFastForwardData `
  -ResumeCheckpoint $checkpoint `
  -RunRoot "D:/pfc_runs/pc_modal_lm" `
  -RunSuffix $RunSuffix

exit $LASTEXITCODE
