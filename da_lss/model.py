from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class ModelConfig:
    hidden: int = 64
    health_points: int = 7
    health_min: float = 0.7
    health_max: float = 1.05
    temperature: float = 0.05
    kappa: float = 0.5
    min_support: float = 8.0
    backbone: str = "lss"

    def __post_init__(self):
        if self.hidden < 2 or self.health_points < 2:
            raise ValueError("hidden and health_points must be at least two")
        if not 0 <= self.kappa < 1 or self.temperature <= 0:
            raise ValueError("Require 0 <= kappa < 1 and positive temperature")
        if self.health_min >= self.health_max or self.min_support < 1:
            raise ValueError("Invalid health reference range or support")
        if self.backbone not in ("lss", "lstm", "bigru", "transformer"):
            raise ValueError("backbone must be lss, lstm, bigru or transformer")
        if self.backbone == "transformer" and self.hidden % 4:
            raise ValueError("Transformer hidden dimension must be divisible by four")
        if self.backbone == "bigru" and self.hidden % 2:
            raise ValueError("Bidirectional GRU needs an even hidden dimension")


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, strength):
        ctx.strength = strength
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.strength * grad, None


class HealthGate(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("references", torch.linspace(
            cfg.health_min, cfg.health_max, cfg.health_points))
        self.register_buffer("discrepancy", torch.zeros(cfg.health_points, cfg.hidden))
        self.register_buffer("relevance", torch.zeros(cfg.hidden))
        self.register_buffer("supported", torch.zeros(cfg.health_points, dtype=torch.bool))

    def weights(self, soh):
        return torch.softmax(-0.5 * (
            (soh[:, None] - self.references) / self.cfg.temperature).square(), dim=-1)

    def forward(self, soh):
        d = self.discrepancy
        reduction = self.supported[:, None] * d / (1 + d + self.relevance)
        return 1 - self.cfg.kappa * (self.weights(soh) @ reduction)

    @torch.no_grad()
    def fit_moments(self, source, target):
        def moments(stats):
            mass, squared_mass, first, second = stats
            mean = first / mass.clamp_min(1e-12)[:, None]
            std = (second / mass.clamp_min(1e-12)[:, None] - mean.square()).clamp_min(0).sqrt()
            neff = mass.square() / squared_mass.clamp_min(1e-12)
            valid = (neff >= self.cfg.min_support) & (mass >= 1)
            return mean, std, valid

        sm, ss, sv = moments(source)
        tm, ts, tv = moments(target)

        mass, _, first, second = source
        mean = first.sum(0) / mass.sum().clamp_min(1e-12)
        scale = (second.sum(0) / mass.sum().clamp_min(1e-12) - mean.square()).clamp_min(1e-8).sqrt()
        self.discrepancy.copy_(((sm - tm).abs() + (ss - ts).abs()) / scale)
        adjacent = sv[1:] & sv[:-1]
        changes = (sm[1:] - sm[:-1]).abs() / scale
        self.relevance.copy_(changes[adjacent].mean(0) if adjacent.any() else torch.zeros_like(scale))
        self.supported.copy_(sv & tv)


def diagonal_zoh(a, delta, bx):
    ad = delta * a
    return ad.exp(), torch.expm1(ad) / a * bx


class DALSS(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        h = self.cfg.hidden
        branch = h // 2
        self.conv3 = nn.Conv1d(4, branch, 3, padding=1)
        self.conv7 = nn.Conv1d(4, branch, 7, padding=3)
        self.fusion = nn.Conv1d(2 * branch, h, 1)
        self.norm = nn.InstanceNorm1d(h, affine=True, track_running_stats=False)
        self.auxiliary = nn.Linear(h, 1)
        self.log_decay = nn.Parameter(torch.zeros(h))
        self.base_step = nn.Parameter(torch.full((h,), -2.0))
        self.step = nn.Linear(h, h, bias=False)
        self.input_projection = nn.Linear(h, h, bias=False)
        self.output_projection = nn.Linear(h, h, bias=False)
        self.gate = HealthGate(self.cfg)
        self.regression = nn.Linear(h, 1)
        self.domain = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        if self.cfg.backbone == "lstm":
            self.sequence = nn.LSTM(h, h, batch_first=True)
        elif self.cfg.backbone == "bigru":
            self.sequence = nn.GRU(h, h // 2, bidirectional=True, batch_first=True)
        elif self.cfg.backbone == "transformer":
            layer = nn.TransformerEncoderLayer(h, 4, dim_feedforward=2 * h,
                                              dropout=0.1, batch_first=True)
            self.sequence = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        if self.cfg.backbone != "lss":

            del self.log_decay, self.base_step, self.step
            del self.input_projection, self.output_projection
        nn.init.constant_(self.regression.bias, 0.9)
        nn.init.constant_(self.auxiliary.bias, 0.9)

    def extract(self, x):
        x = x.transpose(1, 2)
        fused = self.fusion(torch.cat((self.conv3(x), self.conv7(x)), dim=1))


        auxiliary = self.auxiliary(fused.mean(-1)).squeeze(-1)
        features = self.norm(fused).transpose(1, 2)
        return features, auxiliary

    def candidate(self, x):
        features, auxiliary = self.extract(x)
        delta = F.softplus(self.base_step + self.step(features))
        a = -self.log_decay.exp()
        transition, write = diagonal_zoh(a, delta, self.input_projection(features))
        return transition, write, auxiliary

    def encode(self, x, use_gate=True):
        if self.cfg.backbone != "lss":
            features, auxiliary = self.extract(x)
            if self.cfg.backbone == "transformer":

                position = torch.arange(x.shape[1], device=x.device, dtype=x.dtype)[:, None]
                frequency = torch.exp(torch.arange(0, self.cfg.hidden, 2, device=x.device,
                                                   dtype=x.dtype) * (-9.210340371976184 / self.cfg.hidden))
                encoding = torch.stack((torch.sin(position * frequency),
                                        torch.cos(position * frequency)), dim=-1).flatten(-2)
                hidden = self.sequence(features + encoding)
            else:
                hidden, _ = self.sequence(features)
            return hidden, auxiliary, torch.ones_like(hidden[:, 0])
        transition, write, auxiliary = self.candidate(x)


        gate = self.gate(auxiliary.detach()) if use_gate else torch.ones_like(write[:, 0])
        state = torch.zeros_like(write[:, 0])
        states = []
        for t in range(x.shape[1]):
            state = (1 - gate) * state + gate * (transition[:, t] * state + write[:, t])
            states.append(state)
        hidden = self.output_projection(torch.stack(states, dim=1))
        return hidden, auxiliary, gate

    def predict_soh(self, x, use_gate=True):
        hidden, _, _ = self.encode(x, use_gate=use_gate)
        return self.regression(hidden.mean(dim=1)).squeeze(-1)

    def forward(self, x, reversal=0.0, use_gate=True):
        hidden, auxiliary, gate = self.encode(x, use_gate=use_gate)
        pooled = hidden.mean(dim=1)
        return {
            "prediction": self.regression(pooled).squeeze(-1),
            "auxiliary": auxiliary,
            "domain": self.domain(GradientReversal.apply(pooled, reversal)).squeeze(-1),
            "hidden": hidden,
            "gate": gate,
        }


def temporal_monotonicity(current, following, valid, epsilon=0.005):
    if epsilon < 0:
        raise ValueError("epsilon must be nonnegative")
    penalties = F.relu(following - current - epsilon)
    return penalties[valid].mean() if valid.any() else current.sum() * 0
