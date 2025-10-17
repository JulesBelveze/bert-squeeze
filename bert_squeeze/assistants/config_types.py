from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TrainConfig:
    general: Dict[str, Any] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    logger: Optional[Dict[str, Any]] = None
    callbacks: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DistilConfig:
    general: Dict[str, Any] = field(default_factory=dict)
    train: Dict[str, Any] = field(default_factory=dict)
    model: Dict[str, Any] = field(default_factory=dict)
    data: Dict[str, Any] = field(default_factory=dict)
    logger: Optional[Dict[str, Any]] = None
    callbacks: List[Dict[str, Any]] = field(default_factory=list)
