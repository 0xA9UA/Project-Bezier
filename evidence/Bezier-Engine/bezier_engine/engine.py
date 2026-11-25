"""A custom execution scheduler on PyTorch kernels, preserving model equations.

This is single-request/single-stream, not a thread-safe serving system. Pinned
buffers are reused only for inference; training uses differentiable tensor copies.
The public predict method always accepts and returns CPU tensors.
"""
from dataclasses import asdict
from collections import Counter
import copy
import torch
from torch import nn
import torch.nn.functional as F

from bezier_sphere.model import ModelConfig, SphereClassifier
from bezier_sphere.geometry import bezier, bernstein, inside_ball, on_sphere

MODES = ("cpu", "gpu", "cpu_activation", "gpu_routing")


class Bridge:
    def __init__(self, pinned=False):
        self.pinned = pinned
        self.buffers = {}
        self.audit = False
        self.counts = Counter()
        self.details = []

    def move(self, x, device, key):
        device = torch.device(device)
        if x.device.type == device.type:
            return x
        if self.audit:
            direction = f"{x.device.type}_to_{device.type}"
            self.counts[direction + "_copies"] += 1
            self.counts[direction + "_bytes"] += x.numel() * x.element_size()
            self.details.append({"key": key, "direction": direction,
                                 "shape": list(x.shape), "bytes": x.numel()*x.element_size()})
        if not self.pinned or torch.is_grad_enabled():
            return x.to(device)  # blocking D2H is deliberate; copies preserve autograd
        slot_key = (key, device.type, tuple(x.shape), x.dtype)
        if slot_key not in self.buffers:
            host = torch.empty(x.shape, dtype=x.dtype, pin_memory=True)
            target = torch.empty_like(x, device="cuda") if device.type == "cuda" else None
            self.buffers[slot_key] = (host, target, torch.cuda.Event())
        host, target, event = self.buffers[slot_key]
        # Protect both CPU reads after D2H and buffer reuse after asynchronous H2D.
        event.synchronize()
        if device.type == "cpu":
            host.copy_(x, non_blocking=True)
            event.record()
            event.synchronize()
            return host
        host.copy_(x)
        target.copy_(host, non_blocking=True)
        event.record()
        return target


class BezierEngine(nn.Module):
    def __init__(self, source: SphereClassifier, mode="gpu", pinned=False, cpu_dispatch="gather"):
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if cpu_dispatch not in ("gather", "grouped"):
            raise ValueError("CPU dispatch must be gather or grouped")
        if source.routing != "bezier":
            raise ValueError("This experiment requires a cubic Bézier route")
        self.mode, self.cpu_dispatch = mode, cpu_dispatch
        self.cfg = copy.deepcopy(source.cfg)
        self.model = copy.deepcopy(source).cpu()
        self.bridge = Bridge(pinned)
        if mode in ("gpu", "cpu_activation"):
            self.model.cuda()
            if mode == "cpu_activation" and self.cfg.activation == "bezier":
                self.model.bank.coefficients = nn.Parameter(self.model.bank.coefficients.detach().cpu())
        elif mode == "gpu_routing":
            self.model.route.cuda()
            self.model.raw_centers = nn.Parameter(self.model.raw_centers.detach().cuda())
            self.model.times = self.model.times.cuda()

    @property
    def compute_device(self):
        return "cuda" if self.mode in ("gpu", "cpu_activation") else "cpu"

    def activation(self, x, key):
        if self.mode != "cpu_activation":
            return F.silu(x)
        host = self.bridge.move(x, "cpu", key + "_out")
        return self.bridge.move(F.silu(host), "cuda", key + "_back")

    def controls(self, h):
        m = self.model
        r = self.bridge.move(h, m.route.network[0].weight.device, "router_input")
        activation_index = 0
        for layer in m.route.network:
            if isinstance(layer, nn.SiLU):
                key = "router_activation" if activation_index == 0 else f"router_activation_{activation_index}"
                r = self.activation(r, key)
                activation_index += 1
            else:
                r = layer(r)
        raw = r.reshape(h.shape[0], 4, 3)
        return torch.stack((on_sphere(raw[:, 0]), inside_ball(raw[:, 2]),
                            inside_ball(raw[:, 3]), on_sphere(raw[:, 1])), dim=1)

    def route(self, h):
        control = self.controls(h)
        positions = bezier(control, self.model.times)
        scores = -(positions[:, :, None] - self.model.centers[None, None]).square().sum(-1)
        scores = scores / self.cfg.bandwidth**2
        values, indices = scores.topk(self.cfg.top_k, dim=-1)
        weights = values.softmax(-1)
        return (self.bridge.move(indices, h.device, "route_indices"),
                self.bridge.move(weights, h.device, "route_weights"))

    def bank(self, h, indices, weights, step):
        m, c = self.model.bank, self.cfg
        if h.device.type == "cpu" and self.cpu_dispatch == "grouped":
            # Reuse each selected matrix across its inputs. No unused expert is run.
            selected = h.new_zeros((h.shape[0], c.top_k, c.width))
            for e in range(c.experts):
                rows, slots = torch.where(indices == e)
                if rows.numel() == 0:
                    continue
                z = h[rows] @ m.up[e] + m.up_bias[e]
                z = ((bernstein(torch.sigmoid(z))*m.coefficients[e]).sum(-1)
                     if c.activation == "bezier" else F.silu(z))
                selected[rows, slots] = z @ m.down[e] + m.down_bias[e]
        else:
            b, k = indices.shape
            flat = indices.reshape(-1)
            inp = h[:, None].expand(-1, k, -1).reshape(b*k, 1, c.width)
            z = torch.bmm(inp, m.up[flat]).squeeze(1) + m.up_bias[flat]
            if c.activation == "bezier":
                if self.mode == "cpu_activation":
                    host_z = self.bridge.move(z, "cpu", f"expert_{step}_out")
                    host_idx = self.bridge.move(flat, "cpu", f"expert_{step}_indices")
                    z = (bernstein(torch.sigmoid(host_z))*m.coefficients[host_idx]).sum(-1)
                    z = self.bridge.move(z, "cuda", f"expert_{step}_back")
                else:
                    z = (bernstein(torch.sigmoid(z))*m.coefficients[flat]).sum(-1)
            else:
                z = self.activation(z, f"expert_{step}")
            selected = (torch.bmm(z[:, None], m.down[flat]).squeeze(1)+m.down_bias[flat]).reshape(b,k,c.width)
        return (selected*weights[..., None]).sum(1)

    def forward(self, x):
        m = self.model
        x = self.bridge.move(x, m.encoder[0].weight.device, "input")
        h = self.activation(m.encoder[0](x), "encoder_activation")
        indices, weights = self.route(h)
        for step in range(self.cfg.steps):
            h = h + self.bank(m.norm(h), indices[:, step], weights[:, step], step)/self.cfg.steps
        return m.decoder(h)

    @torch.inference_mode()
    def predict(self, x):
        if x.device.type != "cpu":
            raise ValueError("predict expects CPU input for end-to-end measurement")
        result = self.bridge.move(self(x), "cpu", "output")
        # Do not expose a mutable pinned scratch buffer to callers.
        return result.clone() if self.bridge.pinned and self.compute_device == "cuda" else result

    def cpu_state(self):
        return {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}

    def parameter_bytes(self):
        counts = Counter()
        for p in self.parameters():
            counts[p.device.type] += p.numel()*p.element_size()
        return dict(counts)

    def describe_transfers(self, x):
        self.bridge.counts.clear()
        self.bridge.details.clear()
        self.bridge.audit = True
        try:
            self.predict(x)
            return {"totals": dict(self.bridge.counts), "transfers": list(self.bridge.details)}
        finally:
            self.bridge.audit = False


def source_model(config, state=None):
    model = SphereClassifier(ModelConfig(**config))
    if state is not None:
        model.load_state_dict(state)
    return model


def load_engine(path, mode="cpu", **kwargs):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    return BezierEngine(source_model(saved["config"], saved["state_dict"]), mode, **kwargs).eval()

