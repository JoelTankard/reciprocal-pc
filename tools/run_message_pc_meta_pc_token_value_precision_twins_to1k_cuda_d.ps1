$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$common = @{
  Steps = 1000
  LogInterval = 100
  CheckpointInterval = 100
  HiddenDim = 256
  Layers = 4
  Heads = 4
  InferenceSteps = 1
  InferenceRate = 1.0
  Seed = 2
  LearningRate = 0.001
  DecoderLrScale = 1.0
  StateLrScale = 1.0
  EmbedLrScale = 14.0
  TokenFeedbackScale = 1.0
  RouteUpdateMode = "local_energy"
  RouteLrScale = 0.1
  FeedbackLrScale = 0.0
  GenerativeLrScale = 0.0
  DecoderSupervisionMode = "top_only"
  DecoderStateMode = "rms_norm"
  ReadoutLayer = 0
  BlockStateMode = "nanochat_residual"
  FfnDim = 1022
  FfnLrScale = 1.0
  FfnLearningSource = "phase"
  TokenValueEmbeddings = $true
  TokenValueScale = 1.0
  TokenValueLrScale = 0.5
  SensoryReinjectionStart = 0.2
  SensoryReinjectionEnd = 0.05
  SensoryReinjectionMode = "pre_block"
  ResidualStreamScaleStart = 1.15
  ResidualStreamScaleEnd = 1.05
  LearnResidualStreamScales = $true
  ResidualStreamLrScale = 1.0
  InferenceMode = "meta_pc"
  LearningPhaseSource = "settled"
  ForwardUpdateTopLayers = 4
  DecoderLearningSource = "free"
  MatrixOptimizer = "muon"
  OptimizerWarmupSteps = 40
  MuonLrScale = 16.0
  MuonMomentum = 0.85
  MuonMomentumFinal = 0.97
  MuonMomentumWarmupSteps = 400
  MuonBeta2 = 0.9
  MuonNsSteps = 5
  RunRoot = "D:/pfc_runs/pc_modal_lm"
}

& (Join-Path $repo "tools/run_message_pc_nanochat_exact_cuda_d.ps1") `
  @common `
  -MetaFeedbackMode "exact_vjp" `
  -MetaOutputFeedbackMode "decoder_transpose" `
  -RunSuffix "meta_pc_tokenvalue_precision_exact_d256_f1022_seed2_1k"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& (Join-Path $repo "tools/run_message_pc_nanochat_exact_cuda_d.ps1") `
  @common `
  -MetaFeedbackMode "learned_reciprocal" `
  -MetaReciprocalLearningRate 0.01 `
  -MetaReciprocalNoiseStd 0.01 `
  -MetaOutputFeedbackMode "kp_learned" `
  -MetaOutputFeedbackLrScale 1.0 `
  -RunSuffix "meta_pc_tokenvalue_precision_strict_d256_f1022_seed2_1k"

exit $LASTEXITCODE
