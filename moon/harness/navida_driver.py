"""NaVIDA policy driver: prompt construction, generation and the frame history."""
import random
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from transformers import GenerationConfig

from moon.harness.chunking import LEFT, chunk_text_to_atoms
from moon.harness.judge import SYSTEM_PROMPT
from moon.navida_io import (NAVIDA_MAX_HISTORY, RESOLUTION_RATIO, frame_from_sensor,
                            judge_frame_from_sensor)

from moon.harness.settings import DECODE_MODES


def decode_config(mode: str = "greedy") -> GenerationConfig:
    """The upstream generation settings: ``official`` samples at the upstream
    temperature, ``greedy`` is its deterministic variant used for evaluation."""
    if mode not in DECODE_MODES:
        raise ValueError(f"decode mode must be one of {DECODE_MODES}, got {mode!r}")
    if mode == "greedy":
        return GenerationConfig(
            do_sample=False, temperature=1.0, top_p=1.0, top_k=0,
            max_new_tokens=512, use_cache=True, repetition_penalty=1.05,
            num_return_sequences=1)
    # Upstream sets no top_k, so it ran with the transformers 4.x default of 50;
    # 5.x has no default and would take the checkpoint's value instead.
    return GenerationConfig(
        do_sample=True, temperature=0.2, top_p=1.0, top_k=50,
        max_new_tokens=512, use_cache=True, repetition_penalty=1.05,
        num_return_sequences=1)


def sampling_config(temperature: float, top_p: float = 1.0) -> GenerationConfig:
    """Sampled decoding with the upstream length and repetition settings."""
    return GenerationConfig(
        do_sample=True, temperature=temperature, top_p=top_p, top_k=0,
        max_new_tokens=512, use_cache=True, repetition_penalty=1.05,
        num_return_sequences=1)


PROMPT_TEMPLATE = (
    "Imagine you are a robot programmed for navigation tasks. "
    "You have been given a video of historical observations and an image of the "
    "current observation. Your assigned task is: '{}'. Analyze this series of "
    "images to decide your next move, which could involve turning left or right "
    "by a specific degree or moving forward a certain distance.")


def sample_history(data, n=8):
    if len(data) <= n:
        return data
    return [data[round(i * (len(data) - 1) / (n - 1))] for i in range(n)]


def build_navida_messages(instruction: str, rgb_list: List[Image.Image]) -> List[Dict]:
    """The upstream conversation, verbatim."""
    hist = sample_history(rgb_list[:-1]) if len(rgb_list) > 1 else [rgb_list[-1]]
    content = [{"type": "text", "text": "Imagine you are a robot programmed for "
                                        "navigation tasks. You have been given a video "
                                        "of historical observations"}]
    content += [{"type": "image", "image": im} for im in hist]
    content.append({"type": "text", "text": "and an image of the current observation"})
    content.append({"type": "image", "image": rgb_list[-1]})
    content.append({"type": "text",
                    "text": PROMPT_TEMPLATE.format(instruction).split("current observation")[1]})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
            {"role": "user", "content": content}], hist + [rgb_list[-1]]


class NavidaDriver:
    """BasePolicy on a shared NavidaHost.

    ``policy_role`` selects the adapter that generates: None is the frozen base.
    The driver keeps two frame histories, the policy's and the judge's, because
    the two consume different frame contracts (navida_io). ``restart_gen_cfg``
    replaces the decoding after a restart, when given.
    """

    def __init__(self, host, max_history: int = NAVIDA_MAX_HISTORY, decode: str = "greedy",
                 resolution_ratio: float = RESOLUTION_RATIO, restart_gen_cfg=None):
        self.host = host
        self.restart_gen_cfg = restart_gen_cfg
        self.restarted = False
        self.policy_role = None
        self.processor = host.processor
        self.device = host.device
        self.max_history = max_history
        self.resolution_ratio = float(resolution_ratio)
        self.gen_cfg = decode_config(decode)
        self.use_model_defaults = True
        self.history: List[Image.Image] = []
        self.judge_history: List[Image.Image] = []
        self.recent_actions: List[str] = []
        self.reset("")

    def reset(self, instruction: str) -> None:
        self.instruction = instruction
        self.history.clear()
        self.judge_history.clear()
        self.recent_actions.clear()
        self.last_text = ""
        self.restarted = False

    def note_action(self, name: str) -> None:
        self.recent_actions.append(name)

    def restart_fresh(self, instruction: str, step: int) -> None:
        """Clear the frame history; the current frame becomes the new first frame.
        The executed-action record the judge reads is kept."""
        frame = self.history[-1] if self.history else None
        jframe = self.judge_history[-1] if self.judge_history else None
        keep_recent = list(self.recent_actions)
        self.reset(instruction)
        self.restarted = True
        self.recent_actions.extend(keep_recent)
        if frame is not None:
            self.history.append(frame)
            self.judge_history.append(jframe)

    @property
    def rgb_list(self) -> List[Image.Image]:
        return self.history[-self.max_history:]

    @property
    def judge_rgb_list(self) -> List[Image.Image]:
        """Never truncated: the judge's training rows sampled the whole span."""
        return self.judge_history

    def observe(self, rgb: np.ndarray) -> None:
        self.history.append(frame_from_sensor(rgb, self.resolution_ratio))
        self.judge_history.append(judge_frame_from_sensor(rgb))

    def _build(self):
        msgs, images = build_navida_messages(self.instruction, self.rgb_list)
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return text, images

    @torch.inference_mode()
    def _generate(self) -> str:
        text, images = self._build()
        inputs = self.processor(text=[text], images=[images], return_tensors="pt",
                                padding=True).to(self.device)
        from moon.harness.host import generate_kwargs
        with self.host.role(self.policy_role) as m:
            out = m.generate(**inputs, generation_config=self.active_gen_cfg(),
                             output_scores=False, return_dict_in_generate=False,
                             **generate_kwargs(self.use_model_defaults))
        n = inputs["input_ids"].shape[1]
        return self.processor.batch_decode(out[:, n:], skip_special_tokens=True)[0].strip()

    def active_gen_cfg(self):
        if self.restarted and self.restart_gen_cfg is not None:
            return self.restart_gen_cfg
        return self.gen_cfg

    def propose_chunk(self, state) -> List[int]:
        self.last_text = self._generate()
        return chunk_text_to_atoms(self.last_text, rng=random) or [LEFT]

