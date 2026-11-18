"""Typed, validated decoder configuration (loaded from ``configs/decoder.yaml``)."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

EMISSION_MODES = ("normalised", "logit_normalised", "raw_log")


@dataclass(frozen=True)
class EmissionConfig:
    """Classifier-score -> log pseudo-emission transformation.

    mode:
      normalised        q_t(a) = clip(p_a) / sum_b clip(p_b); log_emission = log q_t(a)
      logit_normalised  q_t(a) ∝ (p_a / (1 - p_a)) ** (1 / temperature)  (softmax of logits)
      raw_log           log clip(p_a) (not normalised across candidates)
    """

    mode: str = "normalised"
    epsilon: float = 1e-6
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.mode not in EMISSION_MODES:
            raise ValueError(f"emission.mode must be one of {EMISSION_MODES}, got {self.mode!r}")
        if not 0.0 < self.epsilon < 0.5:
            raise ValueError("emission.epsilon must be in (0, 0.5)")
        if self.temperature <= 0:
            raise ValueError("emission.temperature must be > 0")


@dataclass(frozen=True)
class TransitionConfig:
    """Directed network transition scoring (all distances in metres)."""

    beta_m: float = 30.0                 # scale of |d_net - d_gps| penalty
    max_speed_mps: float = 45.0          # hard feasibility limit on implied speed
    soft_speed_mps: float = 30.0         # speeds above this are softly penalised
    speed_penalty_weight: float = 1.0
    speed_slack_m: float = 100.0         # extra search distance beyond max_speed * dt
    backward_tolerance_m: float = 5.0    # GPS-noise tolerance for same-arc "backward" projection
    max_search_distance_m: float = 5000.0  # hard cap on Dijkstra radius

    def __post_init__(self) -> None:
        for name in ("beta_m", "max_speed_mps", "soft_speed_mps", "max_search_distance_m"):
            if getattr(self, name) <= 0:
                raise ValueError(f"transition.{name} must be > 0")
        if self.soft_speed_mps > self.max_speed_mps:
            raise ValueError("transition.soft_speed_mps must be <= max_speed_mps")
        if self.speed_penalty_weight < 0 or self.speed_slack_m < 0 or self.backward_tolerance_m < 0:
            raise ValueError("transition penalties/slack/tolerance must be >= 0")


@dataclass(frozen=True)
class GapConfig:
    """Segment-boundary policy for large time gaps."""

    max_gap_seconds: float = 120.0
    # A skipped observation may either be bridged or force a route split.
    missing_candidate_policy: str = "bridge"   # "bridge" | "split"
    max_bridged_observations: int = 1000000

    def __post_init__(self) -> None:
        if self.max_gap_seconds <= 0:
            raise ValueError("gaps.max_gap_seconds must be > 0")
        if self.missing_candidate_policy not in ("bridge", "split"):
            raise ValueError("gaps.missing_candidate_policy must be 'bridge' or 'split'")
        if self.max_bridged_observations < 0:
            raise ValueError("gaps.max_bridged_observations must be >= 0")


@dataclass(frozen=True)
class ReconstructionConfig:
    insert_intermediate: bool = True
    max_intermediate_arcs: int = 500

    def __post_init__(self) -> None:
        if self.max_intermediate_arcs < 0:
            raise ValueError("reconstruction.max_intermediate_arcs must be >= 0")


@dataclass(frozen=True)
class DecoderConfig:
    emission: EmissionConfig = field(default_factory=EmissionConfig)
    transition: TransitionConfig = field(default_factory=TransitionConfig)
    gaps: GapConfig = field(default_factory=GapConfig)
    reconstruction: ReconstructionConfig = field(default_factory=ReconstructionConfig)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any] | None) -> "DecoderConfig":
        data = dict(data or {})
        sections = {"emission": EmissionConfig, "transition": TransitionConfig,
                    "gaps": GapConfig, "reconstruction": ReconstructionConfig}
        unknown = set(data) - set(sections)
        if unknown:
            raise ValueError(f"Unknown decoder config sections: {sorted(unknown)}")
        built = {name: _build(klass, data.get(name), name) for name, klass in sections.items()}
        return cls(**built)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DecoderConfig":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(yaml.safe_load(fh) or {})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _build(klass: type, data: Mapping[str, Any] | None, name: str):
    data = dict(data or {})
    valid = {f.name for f in dataclasses.fields(klass)}
    unknown = set(data) - valid
    if unknown:
        raise ValueError(f"Unknown keys in decoder config section '{name}': {sorted(unknown)}")
    return klass(**data)
