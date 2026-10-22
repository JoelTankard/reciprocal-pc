"""Golden inference, persistent RLS state, and unaveraged strict-local training."""

from pfc.reciprocal_pc import (
    ReciprocalPC,
    ReciprocalPCConfig,
    ReciprocalPCTrainer,
    ReciprocalPCTrainingConfig,
    create_reciprocal_pc,
    load_reciprocal_pc_checkpoint,
    model_config,
    training_config,
)
from .golden import (
    GOLDEN_CHECKPOINT_SHA256,
    GOLDEN_CHECKPOINT_URL,
    Golden,
    create_golden,
    load_golden_checkpoint,
)
from .rls_branch import AdditiveRLSBranch, RLSBranchState
from .rls_memory import RLSState
from .shared_prefill import golden_local_update, shared_prefill

__all__ = [
    "Golden",
    "AdditiveRLSBranch",
    "RLSBranchState",
    "RLSState",
    "create_golden",
    "load_golden_checkpoint",
    "GOLDEN_CHECKPOINT_URL",
    "GOLDEN_CHECKPOINT_SHA256",
    "golden_local_update",
    "shared_prefill",
    "ReciprocalPC",
    "ReciprocalPCConfig",
    "ReciprocalPCTrainer",
    "ReciprocalPCTrainingConfig",
    "create_reciprocal_pc",
    "load_reciprocal_pc_checkpoint",
    "model_config",
    "training_config",
]

__version__ = "0.1.0"
