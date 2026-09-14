from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from pfc import (
    ReciprocalPC,
    ReciprocalPCTrainer,
    create_reciprocal_pc,
    load_reciprocal_pc_checkpoint,
    model_config,
    training_config,
)


def test_release_constructs_exact_phase827_model() -> None:
    model, trainer = create_reciprocal_pc()

    assert isinstance(model, ReciprocalPC)
    assert isinstance(trainer, ReciprocalPCTrainer)
    assert (model.config.width, model.config.depth, model.config.ffn_width) == (
        512,
        2,
        1040,
    )
    assert model.config.vocab_size == 32768
    assert model.config.shared_embedding_decoder
    assert model.config.fixed_ternary_hadamard_lexical_predictor
    assert model.config.temporal_hgf_precision_carry
    assert model.config.projected_self_bilinear_token_observation
    assert model.projected_self_bilinear_observation_matrix is not None
    assert sum(parameter.numel() for parameter in model.parameters()) == 23_930_434
    assert not trainer.config.exact_vjp_credit
    assert trainer.config.optimizer_name == "ternary_qat_absmean_g128"


def test_release_inference_returns_finite_full_vocabulary_logits() -> None:
    model, _ = create_reciprocal_pc()
    tokens = torch.tensor([[32759, 11, 23, 47]], dtype=torch.long)

    logits = model(tokens)

    assert logits.shape == (1, 4, 32768)
    assert torch.isfinite(logits).all()


def test_release_training_update_remains_strict_local() -> None:
    _, trainer = create_reciprocal_pc()
    tokens = torch.arange(128, dtype=torch.long).reshape(1, -1)
    tokens[0, 0] = 32759
    targets = tokens.roll(-1, dims=1)

    metrics = trainer.local_update(tokens, targets)

    assert trainer.update_count == 1
    assert metrics["global_backprops"] == 0
    assert metrics["forward_weight_transpose_reads"] == 0


def test_release_checkpoint_round_trip(tmp_path) -> None:
    model, trainer = create_reciprocal_pc()
    path = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_config": asdict(model_config()),
            "trainer_config": asdict(training_config()),
            "model": model.state_dict(),
            "trainer": trainer.state_dict(),
            "step": 0,
            "lifetime_global_backprops": 0,
            "lifetime_forward_weight_transpose_reads": 0,
        },
        path,
    )

    restored_model, restored_trainer, payload = load_reciprocal_pc_checkpoint(path)

    assert payload["step"] == 0
    assert restored_trainer.update_count == 0
    for name, expected in model.state_dict().items():
        assert torch.equal(restored_model.state_dict()[name], expected)
    for name, expected in trainer.state_dict().items():
        actual = restored_trainer.state_dict()[name]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected), name


def test_release_manifest_matches_packaged_files() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "configs/reciprocalpc-release-manifest.json").read_text()
    )

    for item in manifest["release_files"]:
        actual = hashlib.sha256((root / item["path"]).read_bytes()).hexdigest()
        assert actual == item["sha256"], item["path"]
