from __future__ import annotations

import types

import pytest

pytest.importorskip("transformers")
pytest.importorskip("peft")

import vis_ground_lab.models.florence2 as florence2
from vis_ground_lab.models.florence2 import Florence2Wrapper


class DummyProcessor:
    def push_to_hub(self, repo_name, token):
        self.pushed = (repo_name, token)


class DummyBaseModel:
    def __init__(self):
        self.pushed = None

    def push_to_hub(self, repo_name, token):
        self.pushed = (repo_name, token)


def test_load_model_uses_get_peft_model(monkeypatch):
    dummy_processor = DummyProcessor()
    base_model = DummyBaseModel()
    peft_model = DummyBaseModel()
    called = {}

    monkeypatch.setattr(
        florence2.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: dummy_processor),
    )
    monkeypatch.setattr(
        florence2.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: base_model),
    )

    def fake_get_peft_model(model, config):
        called["model"] = model
        called["config"] = config
        peft_model.print_trainable_parameters = lambda: None
        return peft_model

    monkeypatch.setattr(florence2, "get_peft_model", fake_get_peft_model)

    wrapper = Florence2Wrapper(model_name="dummy/model", use_lora=True)
    wrapper.load_model()

    assert wrapper.processor is dummy_processor
    assert wrapper.model is peft_model
    assert called["model"] is base_model


def test_load_model_with_adapter_path(monkeypatch):
    dummy_processor = DummyProcessor()
    base_model = DummyBaseModel()
    adapter_model = DummyBaseModel()
    called = {}

    monkeypatch.setattr(
        florence2.AutoProcessor,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: dummy_processor),
    )
    monkeypatch.setattr(
        florence2.AutoModelForCausalLM,
        "from_pretrained",
        staticmethod(lambda *args, **kwargs: base_model),
    )

    dummy_peft_type = types.SimpleNamespace()

    def fake_from_pretrained(model, adapter_path_or_repo, is_trainable=False):
        called["model"] = model
        called["adapter"] = adapter_path_or_repo
        called["is_trainable"] = is_trainable
        return adapter_model

    dummy_peft_type.from_pretrained = staticmethod(fake_from_pretrained)
    monkeypatch.setattr(florence2, "PeftModel", dummy_peft_type)

    wrapper = Florence2Wrapper(model_name="dummy/model", use_lora=False)
    wrapper.load_model(adapter_path_or_repo="user/repo", is_trainable_adapter=False)

    assert wrapper.model is adapter_model
    assert called["model"] is base_model
    assert called["adapter"] == "user/repo"


def test_push_to_hub_calls_model_and_processor():
    wrapper = Florence2Wrapper(model_name="dummy/model", use_lora=False)
    wrapper.model = DummyBaseModel()
    wrapper.processor = DummyProcessor()

    wrapper.push_to_hub(token="hf_test", repo_name="user/repo")

    assert wrapper.model.pushed == ("user/repo", "hf_test")
    assert wrapper.processor.pushed == ("user/repo", "hf_test")
