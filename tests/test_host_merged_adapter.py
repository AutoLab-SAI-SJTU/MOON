import copy
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

peft = pytest.importorskip("peft")

from moon.harness.host import MergedAdapter, NavidaHost  # noqa: E402


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8)
        self.o_proj = nn.Linear(8, 8)
        self.fc = nn.Linear(8, 8)

    def forward(self, x):
        return self.fc(torch.tanh(self.o_proj(torch.tanh(self.q_proj(x)))))


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def fast_host():
    """A host in fast-adapter mode around a toy backbone, without loading a checkpoint."""
    torch.manual_seed(0)
    plain = Toy().eval()
    wrapped = copy.deepcopy(plain)
    cfg = lambda targets: peft.LoraConfig(r=4, lora_alpha=8, target_modules=targets,
                                          init_lora_weights=False)
    wrapped = peft.get_peft_model(wrapped, cfg(["q_proj", "o_proj"]), adapter_name="restart")
    wrapped.add_adapter("judge", cfg(["q_proj"]))
    wrapped.eval()
    host = NavidaHost.__new__(NavidaHost)
    host.adapters = ["restart", "judge"]
    host.model = wrapped
    host.plain = plain
    host.merged = {"restart": MergedAdapter(plain, wrapped, "restart")}
    host.mounted = None
    return host, plain, wrapped


def run_role(host, name, x):
    with torch.no_grad(), host.role(name) as m:
        return m(x)


def test_base_role_runs_the_plain_backbone_bit_identical_to_the_disabled_adapters():
    host, plain, wrapped = fast_host()
    x = torch.randn(5, 8)
    with torch.no_grad(), wrapped.disable_adapter():
        want = wrapped(x)
    with torch.no_grad(), host.role(None) as m:
        assert m is plain
        got = m(x)
    assert torch.equal(got, want)


def test_merged_recovery_role_matches_the_unmerged_adapter():
    host, plain, wrapped = fast_host()
    x = torch.randn(5, 8)
    wrapped.set_adapter("restart")
    with torch.no_grad():
        want = wrapped(x)
    got = run_role(host, "restart", x)
    assert torch.allclose(got, want, atol=1e-5)
    assert not torch.equal(got, run_role(host, None, x))


def test_returning_to_the_base_restores_the_weights_exactly_after_repeated_mounts():
    host, plain, _ = fast_host()
    original = {k: v.clone() for k, v in plain.state_dict().items()}
    x = torch.randn(5, 8)
    first = run_role(host, None, x)
    for _ in range(5):
        run_role(host, "restart", x)
        run_role(host, "restart", x)
        again = run_role(host, None, x)
        assert torch.equal(again, first)
        assert all(torch.equal(v, original[k]) for k, v in plain.state_dict().items())


def test_judge_role_stays_on_the_adapter_model_and_leaves_the_mounted_recovery_alone():
    host, plain, wrapped = fast_host()
    x = torch.randn(5, 8)
    wrapped.set_adapter("judge")
    with torch.no_grad():
        want = wrapped(x)
    wrapped.set_adapter("restart")
    recovery = run_role(host, "restart", x)
    with torch.no_grad(), host.role("judge") as m:
        assert m is wrapped
        assert torch.allclose(m(x), want, atol=1e-6)
    assert host.mounted == "restart"
    assert torch.equal(run_role(host, "restart", x), recovery)

