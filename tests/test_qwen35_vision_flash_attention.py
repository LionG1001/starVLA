from types import SimpleNamespace

import pytest
import torch.nn as nn

from starVLA.model.modules.vlm import qwen35_musa_flash_attention as flash_attention


class Qwen3_5VisionAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config


class TextAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config


class FakeQwen35Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vision_config = SimpleNamespace(_attn_implementation="eager")
        self.vision_layers = nn.ModuleList(
            [Qwen3_5VisionAttention(vision_config) for _ in range(3)]
        )
        self.text = TextAttention(SimpleNamespace(_attn_implementation="eager"))


def _stub_runtime(monkeypatch) -> None:
    monkeypatch.setattr(flash_attention, "_musa_is_available", lambda: True)
    monkeypatch.setattr(
        flash_attention,
        "_register_musa_varlen_attention",
        lambda *, require_mate: None,
    )


def test_vision_flash_patch_is_isolated_and_reversible(monkeypatch) -> None:
    _stub_runtime(monkeypatch)
    model = FakeQwen35Model()

    patched = flash_attention.configure_qwen35_musa_vision_flash_attention(
        model,
        {"attn_implementation": "eager", "musa_vision_flash_attention": True},
    )

    assert patched == 3
    assert all(
        layer.config._attn_implementation == flash_attention.MUSA_VARLEN_ATTENTION
        for layer in model.vision_layers
    )
    assert model.text.config._attn_implementation == "eager"
    assert (
        flash_attention.configure_qwen35_musa_vision_flash_attention(
            model,
            {"attn_implementation": "eager", "musa_vision_flash_attention": True},
        )
        == 3
    )
    assert flash_attention.disable_qwen35_musa_vision_flash_attention(model) == 3
    assert all(
        layer.config._attn_implementation == "eager"
        for layer in model.vision_layers
    )
    assert model.text.config._attn_implementation == "eager"


def test_vision_flash_requires_model_level_eager(monkeypatch) -> None:
    _stub_runtime(monkeypatch)
    with pytest.raises(RuntimeError, match="attn_implementation=eager"):
        flash_attention.configure_qwen35_musa_vision_flash_attention(
            FakeQwen35Model(),
            {"attn_implementation": "sdpa", "musa_vision_flash_attention": True},
        )


def test_disabled_vision_flash_does_not_require_musa(monkeypatch) -> None:
    monkeypatch.setattr(flash_attention, "_musa_is_available", lambda: False)
    assert (
        flash_attention.configure_qwen35_musa_vision_flash_attention(
            FakeQwen35Model(),
            {"attn_implementation": "eager", "musa_vision_flash_attention": False},
        )
        == 0
    )
