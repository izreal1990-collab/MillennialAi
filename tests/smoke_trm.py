import sys
from pathlib import Path
import torch
import torch.nn as nn

# Ensure repo root is on path for imports when running tests directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hybrid_training import TRMInjectionWrapper, discover_transformer_layers


class DummyLayer(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)

    def forward(self, x, *args, **kwargs):
        return self.lin(x)


class DummyModel(nn.Module):
    def __init__(self, n_layers=6, hidden=16):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([DummyLayer(hidden=hidden) for _ in range(n_layers)])


class FakeTRM(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.thinking_modules = nn.ModuleList([nn.Linear(hidden, hidden)])


def run_smoke():
    device = torch.device('cpu')
    hidden = 16

    model = DummyModel(n_layers=6, hidden=hidden)

    # Discover layers
    layers, parent, attr = discover_transformer_layers(model)
    print('Discovered layers:', len(layers), 'attr:', attr)

    # Wrap layer 1 with TRMInjectionWrapper using a lightweight FakeTRM
    original = layers[1]
    fake_trm = FakeTRM(hidden=hidden)
    wrapper = TRMInjectionWrapper(original_layer=original, trm_template=fake_trm, layer_idx=1, alpha=0.2)
    model.model.layers[1] = wrapper

    # Forward pass
    x = torch.randn(1, 5, hidden, device=device, requires_grad=True)
    out = model.model.layers[1](x)
    print('Output shape:', out.shape)

    # Backprop test
    loss = out.sum()
    loss.backward()

    # Check gradients
    orig_has_grad = any(p.grad is not None for p in original.parameters())
    trm_has_grad = any(p.grad is not None for p in fake_trm.parameters())

    print('Original layer has grad:', orig_has_grad)
    print('Fake TRM has grad:', trm_has_grad)

    assert out.shape == x.shape
    assert orig_has_grad, 'Original layer parameters did not receive gradients'
    assert trm_has_grad, 'TRM parameters did not receive gradients'


if __name__ == '__main__':
    run_smoke()
