"""One NaVIDA host shared by every role.

The base policy runs with all adapters disabled, so it is the frozen upstream
agent; the judge and the recovery policy are LoRA adapters on the same weights,
activated per call. The backbone class follows the checkpoint's config.json:
Qwen2.5-VL and Qwen3-VL share the vocabulary and the prompt format but differ
in patch size, and the frame contract in navida_io follows the model type.
"""
import json
import os
from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import transformers
from transformers import AutoImageProcessor, AutoProcessor, GenerationConfig

from moon import navida_io
from moon.harness.chunking import STOP_TOKEN_ID
from moon.harness.judge import LABELS


def model_type_of(model_path: str) -> str:
    with open(os.path.join(model_path, "config.json")) as f:
        return json.load(f).get("model_type", "")


def model_class_for(model_type: str):
    if model_type == "qwen2_5_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        return Qwen2_5_VLForConditionalGeneration
    if model_type == "qwen3_vl":
        from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration
        return Qwen3VLForConditionalGeneration
    raise ValueError(f"unsupported backbone model_type {model_type!r}")


def backbone_class(model_path: str):
    """Model class for a checkpoint, from its config.json; also selects the frame
    contract every image loader in this process uses."""
    model_type = model_type_of(model_path)
    navida_io.configure(model_type)
    return model_class_for(model_type), model_type


def neutralize_generation_config(cfg: GenerationConfig) -> None:
    """Reset the checkpoint's sampling fields to the library defaults.

    generate() fills every field the caller left at its default from the
    model's generation_config, and transformers 5.x removed the switch that
    disabled this; a checkpoint shipping temperature 0.7 would silently replace
    a caller's 1.0. At the defaults the merge is a no-op.
    """
    d = GenerationConfig()
    cfg.do_sample = False
    cfg.temperature = d.temperature
    cfg.top_p = d.top_p
    cfg.top_k = d.top_k
    cfg.repetition_penalty = d.repetition_penalty


def generate_kwargs(use_model_defaults: bool, version: Optional[str] = None) -> Dict:
    """The per-call defaults switch, only where generate() still accepts it."""
    major = int((version or transformers.__version__).split(".")[0])
    return {} if major >= 5 else {"use_model_defaults": use_model_defaults}


class MergedAdapter:
    """One LoRA adapter of a PEFT model, merged into a plain copy of the backbone.

    Mounting writes base + delta into the plain copy's weights; unmounting copies
    the saved base weights back, so the base stays bit-identical however many
    times the adapter is mounted (unmerging by subtraction would drift in bf16).
    """

    def __init__(self, plain, peft_model, adapter: str):
        self.adapter = adapter
        self.targets = []
        for name, module in peft_model.base_model.model.named_modules():
            if adapter in getattr(module, "lora_A", {}):
                weight = plain.get_submodule(name).weight
                self.targets.append((module, weight, weight.detach().clone()))

    @torch.no_grad()
    def mount(self) -> None:
        for module, weight, saved in self.targets:
            delta = module.get_delta_weight(self.adapter)
            weight.copy_((saved.float() + delta.float()).to(weight.dtype))

    @torch.no_grad()
    def unmount(self) -> None:
        for _, weight, saved in self.targets:
            weight.copy_(saved)


class NavidaHost:
    def __init__(self, model_path: str, device: str = "cuda:0",
                 adapters: Optional[Dict[str, str]] = None,
                 max_pixels: int = navida_io.JUDGE_MAX_PIXELS,
                 attn_implementation: str = "flash_attention_2", merge_adapters=()):
        """``merge_adapters`` names adapters that generate from a plain second copy of
        the backbone with the adapter merged in, and makes the frozen base run on
        that copy too, free of the LoRA wrappers; the others stay unmerged."""
        self.model_type = model_type_of(model_path)
        navida_io.configure(self.model_type)
        load = lambda: model_class_for(self.model_type).from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation=attn_implementation)
        self.model = load()
        self.adapters = list((adapters or {}).keys())
        if self.adapters:
            from peft import PeftModel
            names = list(adapters.items())
            first, path = names[0]
            self.model = PeftModel.from_pretrained(self.model, path, adapter_name=first)
            for name, p in names[1:]:
                self.model.load_adapter(p, adapter_name=name)
        self.model.to(device).eval()
        neutralize_generation_config(self.model.generation_config)
        self.plain, self.merged, self.mounted = None, {}, None
        if merge_adapters:
            missing = set(merge_adapters) - set(self.adapters)
            if missing:
                raise ValueError(f"cannot merge adapters that are not loaded: {sorted(missing)}")
            self.plain = load().to(device).eval()
            neutralize_generation_config(self.plain.generation_config)
            self.merged = {name: MergedAdapter(self.plain, self.model, name)
                           for name in merge_adapters}
        self.processor = AutoProcessor.from_pretrained(model_path)
        # The slow image processor matches training (the fast one shifts pixel
        # values by up to 0.03); use_fast on AutoProcessor would also swap in the
        # slow tokenizer, which cannot decode the ids above 151664 the model samples.
        self.processor.image_processor = AutoImageProcessor.from_pretrained(
            model_path, use_fast=False)
        # The pixel budget is the Qwen2.5 processor's knob; the Qwen3 processor
        # sizes by edge length and a 352x288 frame is far below its limits.
        if hasattr(self.processor.image_processor, "max_pixels"):
            self.processor.image_processor.max_pixels = max_pixels
        self.device = device
        self.tokenizer = self.processor.tokenizer
        self.label_ids = [self.tokenizer.encode(x, add_special_tokens=False)[0]
                          for x in LABELS]
        stop_ids = self.tokenizer.encode("stop", add_special_tokens=False)
        if stop_ids != [STOP_TOKEN_ID]:
            raise RuntimeError(f"'stop' tokenises to {stop_ids}, not [{STOP_TOKEN_ID}]: "
                               "this tokenizer does not match the NaVIDA backbone")

    @contextmanager
    def role(self, name: Optional[str]):
        """None = frozen base (adapters off); otherwise activate that adapter."""
        if self.plain is not None and (name is None or name in self.merged):
            if self.mounted != name:
                if self.mounted is not None:
                    self.merged[self.mounted].unmount()
                if name is not None:
                    self.merged[name].mount()
                self.mounted = name
            yield self.plain
            return
        if not self.adapters:
            yield self.model
            return
        if name is None:
            with self.model.disable_adapter():
                yield self.model
            return
        prev = getattr(self.model, "active_adapter", self.adapters[0])
        self.model.set_adapter(name)
        try:
            yield self.model
        finally:
            self.model.set_adapter(prev)

    @torch.inference_mode()
    def label_probs(self, text: str, images: List, adapter: str) -> Dict[str, float]:
        """Softmax over the label tokens at the last position, one forward."""
        inputs = self.processor(text=[text], images=[images], return_tensors="pt",
                                padding=False).to(self.device)
        with self.role(adapter) as m:
            out = m(**inputs, output_hidden_states=False)
        sub = out.logits[0, -1, :][self.label_ids].float()
        p = torch.softmax(sub, dim=0).cpu().tolist()
        return dict(zip(LABELS, p))

