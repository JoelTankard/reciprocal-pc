param(
  [int]$Steps = 100,
  [int]$LogInterval = 100,
  [int]$CheckpointInterval = 100,
  [int]$HiddenDim = 320,
  [int]$Layers = 4,
  [int]$Heads = 4,
  [int]$InferenceSteps = 4,
  [int]$Seed = 1,
  [double]$LearningRate = 0.001,
  [int]$LrDecayStartStep = 10000,
  [double]$LrDecayFactor = 0.1,
  [double]$DecoderLrScale = 1.0,
  [double]$StateLrScale = 1.0,
  [double]$EmbedLrScale = 1.0,
  [double]$InferenceRate = 0.12,
  [double]$InferenceRateFinal = 0.0,
  [int]$InferenceRateScheduleStart = 0,
  [int]$InferenceRateScheduleEnd = 0,
  [ValidateSet("gradient", "heavy_ball", "conjugate_gradient")]
  [string]$InferenceOptimizer = "gradient",
  [double]$InferenceMomentum = 0.9,
  [double]$TokenFeedbackScale = 0.35,
  [double]$SpikingPrecisionScale = 0.0,
  [int]$FastAttentionWindow = 0,
  [int]$FastAttentionLayers = 0,
  [ValidateSet("cosine_teacher", "local_energy", "token_error_teacher")]
  [string]$RouteUpdateMode = "local_energy",
  [double]$RouteLrScale = 0.25,
  [double]$FeedbackLrScale = 0.25,
  [double]$GenerativeLrScale = 0.25,
  [ValidateSet("all_layers", "top_only")]
  [string]$DecoderSupervisionMode = "all_layers",
  [ValidateSet("raw", "rms_norm")]
  [string]$DecoderStateMode = "raw",
  [int]$ReadoutLayer = 0,
  [double]$ReadoutObjectiveShare = 0.0,
  [double]$WinnerMargin = 0.0,
  [double]$WinnerMarginScale = 0.0,
  [ValidateSet("squashed_state", "identity_residual", "nanochat_residual")]
  [string]$BlockStateMode = "squashed_state",
  [ValidateSet("standard", "depth_mup")]
  [string]$BlockParameterization = "standard",
  [double]$ResidualScale = 0.25,
  [double]$SelfInitGain = 0.02,
  [double]$MessageInitGain = 0.02,
  [int]$FfnDim = 0,
  [double]$FfnInitGain = 0.4,
  [double]$FfnLrScale = 1.0,
  [ValidateSet("phase", "free_to_settled")]
  [string]$FfnLearningSource = "phase",
  [switch]$TokenValueEmbeddings,
  [double]$TokenValueScale = 1.0,
  [double]$TokenValueLrScale = 0.5,
  [switch]$CausalEncoderDecoder,
  [int]$CausalEncoderDepth = 0,
  [double]$CrossAttentionLrScale = 1.0,
  [double]$CrossAttentionGateLrScale = 1.0,
  [double]$CrossAttentionOutputInitGain = 0.0,
  [switch]$MetaCrossMemorySkipCredit,
  [double]$MetaCrossMemorySkipCreditScale = 1.0,
  [double]$SensoryReinjectionStart = 0.0,
  [double]$SensoryReinjectionEnd = 0.0,
  [ValidateSet("post_block", "pre_block")]
  [string]$SensoryReinjectionMode = "post_block",
  [double]$ResidualStreamScaleStart = 1.0,
  [double]$ResidualStreamScaleEnd = 1.0,
  [switch]$LearnResidualStreamScales,
  [ValidateSet("layer_scalar", "channel")]
  [string]$ResidualStreamScaleMode = "layer_scalar",
  [double]$ResidualStreamLrScale = 1.0,
  [double]$ResidualStreamLrScaleFinal = 0.0,
  [int]$ResidualStreamLrScheduleStartStep = 0,
  [int]$ResidualStreamLrScheduleEndStep = 0,
  [ValidateSet("direct_layer", "top_propagated")]
  [string]$TokenFeedbackMode = "direct_layer",
  [ValidateSet("heuristic", "pc_energy", "meta_pc", "bidirectional_pc", "residual_error_pc", "her_error_hierarchy", "difference_target", "recurrent_local", "layerwise_local")]
  [string]$InferenceMode = "heuristic",
  [ValidateSet("exact_vjp", "reciprocal_attention", "learned_reciprocal", "stage_reciprocal")]
  [string]$MetaFeedbackMode = "exact_vjp",
  [double]$MetaReciprocalScale = 0.0,
  [double]$MetaReciprocalLearningRate = 0.0,
  [double]$MetaReciprocalNoiseStd = 0.01,
  [switch]$MetaReciprocalUpdateBeforeSettle,
  [switch]$MetaStateGatedReciprocal,
  [switch]$MetaVarianceNormalizeWeights,
  [switch]$MetaRelinearizeInference,
  [ValidateSet("decoder_transpose", "fixed_random", "kp_learned", "kp_factorized")]
  [string]$MetaOutputFeedbackMode = "decoder_transpose",
  [double]$MetaOutputFeedbackLrScale = 1.0,
  [int]$MetaOutputFeedbackRank = 0,
  [switch]$MetaOutputFeedbackValues,
  [switch]$BidirectionalFreeSettling,
  [switch]$BidirectionalDeploymentSettling,
  [int]$RecurrentSharedSteps = 1,
  [double]$RecurrentStepScale = 1.0,
  [ValidateSet("settled", "free", "dual")]
  [string]$LearningPhaseSource = "settled",
  [int]$ForwardUpdateTopLayers = 0,
  [double]$FreePhaseAmortizationScale = 0.0,
  [double]$FreeLocalTokenCreditScale = 0.0,
  [double]$FreeLocalTokenEmbeddingCreditScale = 0.0,
  [ValidateSet("settled", "free")]
  [string]$DecoderLearningSource = "settled",
  [double]$LearnedFeedbackProposalScale = 0.0,
  [ValidateSet("energy_vjp", "learned_only")]
  [string]$ResidualErrorTransportMode = "energy_vjp",
  [double]$ResidualErrorSuffixDecay = 1.0,
  [ValidateSet("top", "base")]
  [string]$ResidualErrorTargetLayer = "top",
  [double]$ResidualPhaseTransportLrScale = 1.0,
  [int]$ResidualPhaseTransportRefinePasses = 0,
  [double]$DifferenceTargetInverseNoise = 0.05,
  [double]$DifferenceTargetInverseDamping = 0.0,
  [ValidateSet("adjacent", "direct")]
  [string]$DifferenceTargetPathMode = "adjacent",
  [int]$HierarchicalErrorBaseDepth = 0,
  [ValidateSet("factorized", "sign", "identity", "unitary")]
  [string]$HierarchicalErrorBindingMode = "sign",
  [ValidateSet("gated", "direct", "direct_transport", "direct_phase_transport", "recursive_cp")]
  [string]$HierarchicalErrorPredictionMode = "gated",
  [ValidateSet("all_tokens", "fixed_outcome", "mistakes")]
  [string]$HierarchicalErrorTargetMode = "all_tokens",
  [double]$HierarchicalErrorOutcomeCodeScale = 1.0,
  [double]$HierarchicalErrorOutcomeReadoutScale = 0.0,
  [double]$HierarchicalErrorTopDownScale = 1.0,
  [double]$HierarchicalErrorContextCreditScale = 1.0,
  [double]$HierarchicalErrorTokenCreditScale = 0.0,
  [ValidateSet("instant", "persistent", "persistent_hard")]
  [string]$HierarchicalErrorContextMode = "instant",
  [double]$HierarchicalErrorGateLrScale = 1.0,
  [double]$HierarchicalErrorInitialWriteRate = 0.5,
  [double]$HierarchicalErrorMemoryScale = 1.0,
  [int]$HierarchicalErrorLearningStartStep = 0,
  [int]$HierarchicalErrorLevelStartInterval = 0,
  [int]$HierarchicalErrorRecursiveCpRank = 0,
  [double]$HierarchicalErrorRecursiveCpUpperLrScale = 1.0,
  [double]$HierarchicalErrorRecursiveCpComponentNormCap = 0.0,
  [switch]$HierarchicalErrorRecursiveCpDelayedConsolidation,
  [double]$HierarchicalErrorRecursiveCpConsolidationRate = 0.1,
  [double]$HierarchicalErrorRecursiveCpMinImprovement = 0.0,
  [switch]$StatePredictionSeparation,
  [int]$PredictionStreamWindow = 0,
  [double]$SpsPredictionInputScale = 0.0,
  [double]$SpsLayerTokenCreditScale = 0.0,
  [double]$SpsFutureStateCreditScale = 0.0,
  [double]$SpsFutureObservationCreditScale = 0.0,
  [switch]$SpsNodeResidualHierarchy,
  [switch]$HierarchicalErrorOuterProductContract,
  [int]$EnergyBacktrackingSteps = 0,
  [ValidateSet("adam", "muon")]
  [string]$MatrixOptimizer = "adam",
  [int]$OptimizerWarmupSteps = 0,
  [double]$MuonLrScale = 1.0,
  [double]$MuonMomentum = 0.95,
  [double]$MuonMomentumFinal = 0.95,
  [int]$MuonMomentumWarmupSteps = 0,
  [double]$MuonBeta2 = 0.9,
  [int]$MuonNsSteps = 5,
  [string]$RunRoot = "C:/pfc_runs/pc_modal_lm",
  [string]$RunSuffix = "core",
  [string]$ResumeCheckpoint = "",
  [ValidateSet("none", "route", "muon", "context", "recursive_cp", "all")]
  [string]$ResumeOptimizerReset = "none",
  [switch]$ResumeResetHerPredictors,
  [int]$StatePrewritePasses = 0,
  [double]$StatePrewriteLrScale = 1.0,
  [ValidateSet("all", "predictors")]
  [string]$StatePrewriteMode = "all",
  [switch]$ResumeFastForwardData
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$env:TORCH_COMPILE_DISABLE = "1"
$env:NANOCHAT_BASE_DIR = "C:/Users/ai/nanochat_pfc"
$env:PYTHONPATH = "C:/Users/ai/nanochat"

$runName = "message_pc_d${HiddenDim}_l${Layers}_i${InferenceSteps}_${RunSuffix}_${Steps}"
$results = "${RunRoot}/${runName}.jsonl"
$out = "${RunRoot}/${runName}"
if (Test-Path $results) {
  Remove-Item $results
}
New-Item -ItemType Directory -Force $out | Out-Null

$argsList = @(
  "-m", "pfc.pc_modal_lm",
  "--model", "pc-message",
  "--tokenizer", "nanochat",
  "--tokenizer-dir", "runs/pc_modal_lm/tokenizer_4096",
  "--data-mode", "climbmix",
  "--climbmix-loader", "nanochat-bestfit",
  "--data-dir", "data/base_data_climbmix",
  "--train-shards", "8",
  "--mix-train-shards",
  "--steps", $Steps.ToString(),
  "--seed", $Seed.ToString(),
  "--log-interval", $LogInterval.ToString(),
  "--checkpoint-interval", $CheckpointInterval.ToString(),
  "--progress-eval-steps", "50",
  "--eval-steps", "50",
  "--train-eval-steps", "50",
  "--reset-eval-batcher",
  "--restore-selection", "none",
  "--batch-size", "2",
  "--seq-len", "1024",
  "--message-pc-hidden-dim", $HiddenDim.ToString(),
  "--message-pc-layers", $Layers.ToString(),
  "--message-pc-heads", $Heads.ToString(),
  "--message-pc-inference-steps", $InferenceSteps.ToString(),
  "--message-pc-inference-lr", $InferenceRate.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-inference-lr-final", $InferenceRateFinal.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-inference-lr-schedule-start-step", $InferenceRateScheduleStart.ToString(),
  "--message-pc-inference-lr-schedule-end-step", $InferenceRateScheduleEnd.ToString(),
  "--message-pc-inference-optimizer", $InferenceOptimizer,
  "--message-pc-inference-momentum", $InferenceMomentum.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-token-feedback-scale", $TokenFeedbackScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-fast-attention-window", $FastAttentionWindow.ToString(),
  "--message-pc-fast-attention-layers", $FastAttentionLayers.ToString(),
  "--message-pc-prediction-precision", "1.0",
  "--message-pc-spiking-precision-scale", $SpikingPrecisionScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-generative-precision", "0.25",
  "--message-pc-state-delta-rms-cap", "0.20",
  "--message-pc-attention-temperature", "1.0",
  "--message-pc-route-teacher-temperature", "0.25",
  "--message-pc-route-update-mode", $RouteUpdateMode,
  "--message-pc-route-lr-scale", $RouteLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-feedback-lr-scale", $FeedbackLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-generative-lr-scale", $GenerativeLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-position-scale", "0.10",
  "--message-pc-decoder-supervision-mode", $DecoderSupervisionMode,
  "--message-pc-decoder-state-mode", $DecoderStateMode,
  "--message-pc-readout-layer", $ReadoutLayer.ToString(),
  "--message-pc-readout-objective-share", $ReadoutObjectiveShare.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-winner-margin", $WinnerMargin.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-winner-margin-scale", $WinnerMarginScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-block-state-mode", $BlockStateMode,
  "--message-pc-block-parameterization", $BlockParameterization,
  "--message-pc-residual-scale", $ResidualScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-self-init-gain", $SelfInitGain.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-message-init-gain", $MessageInitGain.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-ffn-dim", $FfnDim.ToString(),
  "--message-pc-ffn-init-gain", $FfnInitGain.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-ffn-lr-scale", $FfnLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-ffn-learning-source", $FfnLearningSource,
  "--message-pc-token-value-scale", $TokenValueScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-token-value-lr-scale", $TokenValueLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-cross-attention-lr-scale", $CrossAttentionLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-causal-encoder-depth", $CausalEncoderDepth.ToString(),
  "--message-pc-cross-attention-gate-lr-scale", $CrossAttentionGateLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-cross-attention-output-init-gain", $CrossAttentionOutputInitGain.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-meta-cross-memory-skip-credit-scale", $MetaCrossMemorySkipCreditScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-sensory-reinjection-start", $SensoryReinjectionStart.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-sensory-reinjection-end", $SensoryReinjectionEnd.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-sensory-reinjection-mode", $SensoryReinjectionMode,
  "--message-pc-residual-stream-scale-start", $ResidualStreamScaleStart.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-residual-stream-scale-end", $ResidualStreamScaleEnd.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-residual-stream-scale-mode", $ResidualStreamScaleMode,
  "--message-pc-residual-stream-lr-scale", $ResidualStreamLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-residual-stream-lr-scale-final", $ResidualStreamLrScaleFinal.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-residual-stream-lr-schedule-start-step", $ResidualStreamLrScheduleStartStep.ToString(),
  "--message-pc-residual-stream-lr-schedule-end-step", $ResidualStreamLrScheduleEndStep.ToString(),
  "--message-pc-token-feedback-mode", $TokenFeedbackMode,
  "--message-pc-inference-mode", $InferenceMode,
  "--message-pc-meta-feedback-mode", $MetaFeedbackMode,
  "--message-pc-meta-reciprocal-scale", $MetaReciprocalScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-meta-reciprocal-learning-rate", $MetaReciprocalLearningRate.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-meta-reciprocal-noise-std", $MetaReciprocalNoiseStd.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-meta-output-feedback-mode", $MetaOutputFeedbackMode,
  "--message-pc-meta-output-feedback-lr-scale", $MetaOutputFeedbackLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-meta-output-feedback-rank", $MetaOutputFeedbackRank.ToString(),
  "--message-pc-recurrent-shared-steps", $RecurrentSharedSteps.ToString(),
  "--message-pc-recurrent-step-scale", $RecurrentStepScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-learning-phase-source", $LearningPhaseSource,
  "--message-pc-forward-update-top-layers", $ForwardUpdateTopLayers.ToString(),
  "--message-pc-free-phase-amortization-scale", $FreePhaseAmortizationScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-free-local-token-credit-scale", $FreeLocalTokenCreditScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-free-local-token-embedding-credit-scale", $FreeLocalTokenEmbeddingCreditScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-decoder-learning-source", $DecoderLearningSource,
  "--message-pc-learned-feedback-proposal-scale", $LearnedFeedbackProposalScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-residual-error-transport-mode", $ResidualErrorTransportMode,
  "--message-pc-residual-error-suffix-decay", $ResidualErrorSuffixDecay.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-residual-error-target-layer", $ResidualErrorTargetLayer,
  "--message-pc-residual-phase-transport-lr-scale", $ResidualPhaseTransportLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-residual-phase-transport-refine-passes", $ResidualPhaseTransportRefinePasses.ToString(),
  "--message-pc-difference-target-inverse-noise", $DifferenceTargetInverseNoise.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-difference-target-inverse-damping", $DifferenceTargetInverseDamping.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-difference-target-path-mode", $DifferenceTargetPathMode,
  "--message-pc-hierarchical-error-base-depth", $HierarchicalErrorBaseDepth.ToString(),
  "--message-pc-hierarchical-error-binding-mode", $HierarchicalErrorBindingMode,
  "--message-pc-hierarchical-error-prediction-mode", $HierarchicalErrorPredictionMode,
  "--message-pc-hierarchical-error-target-mode", $HierarchicalErrorTargetMode,
  "--message-pc-hierarchical-error-outcome-code-scale", $HierarchicalErrorOutcomeCodeScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-outcome-readout-scale", $HierarchicalErrorOutcomeReadoutScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-top-down-scale", $HierarchicalErrorTopDownScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-context-credit-scale", $HierarchicalErrorContextCreditScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-token-credit-scale", $HierarchicalErrorTokenCreditScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-context-mode", $HierarchicalErrorContextMode,
  "--message-pc-hierarchical-error-gate-lr-scale", $HierarchicalErrorGateLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-initial-write-rate", $HierarchicalErrorInitialWriteRate.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-memory-scale", $HierarchicalErrorMemoryScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-learning-start-step", $HierarchicalErrorLearningStartStep.ToString(),
  "--message-pc-hierarchical-error-level-start-interval", $HierarchicalErrorLevelStartInterval.ToString(),
  "--message-pc-hierarchical-error-recursive-cp-rank", $HierarchicalErrorRecursiveCpRank.ToString(),
  "--message-pc-hierarchical-error-recursive-cp-upper-lr-scale", $HierarchicalErrorRecursiveCpUpperLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-recursive-cp-component-norm-cap", $HierarchicalErrorRecursiveCpComponentNormCap.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-recursive-cp-consolidation-rate", $HierarchicalErrorRecursiveCpConsolidationRate.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-hierarchical-error-recursive-cp-min-improvement", $HierarchicalErrorRecursiveCpMinImprovement.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-prediction-stream-window", $PredictionStreamWindow.ToString(),
  "--message-pc-sps-prediction-input-scale", $SpsPredictionInputScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-sps-layer-token-credit-scale", $SpsLayerTokenCreditScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-sps-future-state-credit-scale", $SpsFutureStateCreditScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-sps-future-observation-credit-scale", $SpsFutureObservationCreditScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-energy-backtracking-steps", $EnergyBacktrackingSteps.ToString(),
  "--message-pc-matrix-optimizer", $MatrixOptimizer,
  "--message-pc-optimizer-warmup-steps", $OptimizerWarmupSteps.ToString(),
  "--message-pc-muon-lr-scale", $MuonLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-muon-momentum", $MuonMomentum.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-muon-momentum-final", $MuonMomentumFinal.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-muon-momentum-warmup-steps", $MuonMomentumWarmupSteps.ToString(),
  "--message-pc-muon-beta2", $MuonBeta2.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--message-pc-muon-ns-steps", $MuonNsSteps.ToString(),
  "--message-pc-resume-optimizer-reset", $ResumeOptimizerReset,
  "--pc-state-prewrite-passes", $StatePrewritePasses.ToString(),
  "--pc-state-prewrite-lr-scale", $StatePrewriteLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--pc-state-prewrite-mode", $StatePrewriteMode,
  "--message-pc-adam-beta1", "0.9",
  "--message-pc-adam-beta2", "0.99",
  "--message-pc-gradient-rms-cap", "1.0",
  "--message-pc-update-rms-cap", "0.01",
  "--lr", $LearningRate.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--decoder-lr-scale", $DecoderLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--state-lr-scale", $StateLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--embed-lr-scale", $EmbedLrScale.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--state-target-lr-scale", "1.0",
  "--weight-decay", "0.01",
  "--lr-decay-start-step", $LrDecayStartStep.ToString(),
  "--lr-decay-factor", $LrDecayFactor.ToString([Globalization.CultureInfo]::InvariantCulture),
  "--output-dir", $out,
  "--results-jsonl", $results,
  "--run-tag", $runName,
  "--device", "cuda",
  "--timing-warmup-steps", "2",
  "--resource-check-interval", "100",
  "--memory-poll-seconds", "0",
  "--max-system-memory-percent", "99.9",
  "--max-system-load-percent", "99.9",
  "--max-gpu-memory-percent", "95"
)

if ($TokenValueEmbeddings) {
  $argsList += "--message-pc-token-value-embeddings"
}
if ($CausalEncoderDecoder) {
  $argsList += "--message-pc-causal-encoder-decoder"
}
if ($MetaCrossMemorySkipCredit) {
  $argsList += "--message-pc-meta-cross-memory-skip-credit"
}
if ($LearnResidualStreamScales) {
  $argsList += "--message-pc-learn-residual-stream-scales"
}
if ($StatePredictionSeparation) {
  $argsList += "--message-pc-state-prediction-separation"
}
if ($MetaVarianceNormalizeWeights) {
  $argsList += "--message-pc-meta-variance-normalize-weights"
}
if ($MetaReciprocalUpdateBeforeSettle) {
  $argsList += "--message-pc-meta-reciprocal-update-before-settle"
}
if ($MetaStateGatedReciprocal) {
  $argsList += "--message-pc-meta-state-gated-reciprocal"
}
if ($MetaRelinearizeInference) {
  $argsList += "--message-pc-meta-relinearize-inference"
}
if ($MetaOutputFeedbackValues) {
  $argsList += "--message-pc-meta-output-feedback-values"
}
if ($BidirectionalFreeSettling) {
  $argsList += "--message-pc-bidirectional-free-settling"
}
if ($BidirectionalDeploymentSettling) {
  $argsList += "--message-pc-bidirectional-deployment-settling"
}
if ($HierarchicalErrorRecursiveCpDelayedConsolidation) {
  $argsList += "--message-pc-hierarchical-error-recursive-cp-delayed-consolidation"
}
if ($SpsNodeResidualHierarchy) {
  $argsList += "--message-pc-sps-node-residual-hierarchy"
}
if ($HierarchicalErrorOuterProductContract) {
  $argsList += "--message-pc-hierarchical-error-outer-product-contract"
}
if ($ResumeCheckpoint) {
  $argsList += @("--resume-checkpoint", $ResumeCheckpoint)
}
if ($ResumeResetHerPredictors) {
  $argsList += "--message-pc-resume-reset-her-predictors"
}
if ($ResumeFastForwardData) {
  $argsList += "--resume-fast-forward-data"
}

& (Join-Path $repo ".venv/Scripts/python.exe") @argsList 1> "$out/train.log" 2> "$out/train.err"
