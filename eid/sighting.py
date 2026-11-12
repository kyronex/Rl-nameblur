# eid/sighting.py

"""Immutable public contract for a finalized EID sighting."""

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


def _freeze(value: Any) -> Any:
    """Freeze nested dicts and sequences without importing project modules."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    """Rebuild ordinary JSON-compatible containers from the frozen snapshot."""
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class Sighting:
    eid: int
    first_frame: int
    last_frame: int
    created_ts: float
    last_seen_ts: float
    n_obs: int
    identity: Mapping[str, Any]
    dispersion: Mapping[str, Any]
    signal: Mapping[str, Any]
    geometry: Mapping[str, Any]
    tracking: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("identity", "dispersion", "signal", "geometry", "tracking"):
            object.__setattr__(self, name, _freeze(getattr(self, name)))

    @property
    def span_frames(self) -> int:
        return max(1, self.last_frame - self.first_frame)

    @property
    def span_time_s(self) -> float:
        return round(max(0.001, self.last_seen_ts - self.created_ts), 3)

    def to_json(self) -> dict:
        """Return a fresh JSON-serializable dict with the stable wire shape."""
        return {
            "eid": self.eid,
            "identity": _thaw(self.identity),
            "dispersion": _thaw(self.dispersion),
            "signal": _thaw(self.signal),
            "geometry": _thaw(self.geometry),
            "tracking": _thaw(self.tracking),
            "lifecycle": {
                "first_frame": self.first_frame,
                "last_frame": self.last_frame,
                "created_ts": self.created_ts,
                "last_seen_ts": self.last_seen_ts,
                "n_obs": self.n_obs,
                "span_frames": self.span_frames,
                "span_time_s": self.span_time_s,
            },
        }
