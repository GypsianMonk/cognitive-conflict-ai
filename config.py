"""
Centralised, validated configuration.
Single source of truth — imported everywhere instead of raw dicts.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import yaml


@dataclass
class ModelConfig:
    provider: str = "ollama"
    name: str = "mistral"
    temperature: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0

    # Provider-specific
    ollama_base_url: str = "http://localhost:11434"
    openai_api_key: str = ""
    anthropic_api_key: str = ""


@dataclass
class AgentConfig:
    debate_rounds: int = 3
    enable_expert_injection: bool = True
    enable_reflection: bool = True
    reflection_threshold: float = 0.7   # Only reflect if confidence < this
    enable_tools: bool = True


@dataclass
class ScoringConfig:
    relevance_weight: float = 0.4
    coherence_weight: float = 0.4
    contradiction_penalty: float = 0.2
    hallucination_penalty: float = 0.15
    freshness_weight: float = 0.0


@dataclass
class MemoryConfig:
    enabled: bool = True
    max_history: int = 1000
    kg_enabled: bool = True


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    agents: AgentConfig = field(default_factory=AgentConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)

    @classmethod
    def from_dict(cls, data: dict) -> "AppConfig":
        return cls(
            model=ModelConfig(**{k: v for k, v in data.get("model", {}).items()
                                 if k in ModelConfig.__dataclass_fields__}),
            agents=AgentConfig(**{k: v for k, v in data.get("agents", {}).items()
                                  if k in AgentConfig.__dataclass_fields__}),
            scoring=ScoringConfig(**{k: v for k, v in data.get("scoring", {}).items()
                                     if k in ScoringConfig.__dataclass_fields__}),
            memory=MemoryConfig(**{k: v for k, v in data.get("memory", {}).items()
                                   if k in MemoryConfig.__dataclass_fields__}),
        )

    @classmethod
    def from_yaml(cls, path: str = "config/config.yaml") -> "AppConfig":
        p = Path(path)
        if not p.exists():
            p = Path("config/config.example.yaml")
        if p.exists():
            with open(p) as f:
                return cls.from_dict(yaml.safe_load(f) or {})
        return cls()

    def to_dict(self) -> dict:
        """Return legacy dict format for backward compat."""
        return {
            "model": {k: getattr(self.model, k) for k in ModelConfig.__dataclass_fields__},
            "agents": {k: getattr(self.agents, k) for k in AgentConfig.__dataclass_fields__},
            "scoring": {k: getattr(self.scoring, k) for k in ScoringConfig.__dataclass_fields__},
            "memory": {k: getattr(self.memory, k) for k in MemoryConfig.__dataclass_fields__},
        }
