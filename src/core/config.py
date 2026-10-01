"""Environment-driven configuration for the MAGI Consensus Engine."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    base_url: str = "http://localhost:11434/v1"
    api_key: str = "ollama"
    model: str = "qwen2.5:3b"
    timeout_s: float = 120.0
    max_retries: int = 1
    max_tokens: int = 400
    model_overrides: dict[str, str] = field(default_factory=dict)

    def model_for(self, agent_key: str) -> str:
        return self.model_overrides.get(agent_key) or self.model


def load_settings() -> Settings:
    """Load settings from the environment (and a local .env file, if present)."""
    load_dotenv()

    overrides: dict[str, str] = {}
    for key in ("melchior", "balthasar", "casper"):
        value = os.getenv(f"MAGI_{key.upper()}_MODEL", "").strip()
        if value:
            overrides[key] = value

    return Settings(
        base_url=os.getenv("MAGI_BASE_URL", Settings.base_url),
        api_key=os.getenv("MAGI_API_KEY", Settings.api_key),
        model=os.getenv("MAGI_MODEL", Settings.model),
        timeout_s=float(os.getenv("MAGI_TIMEOUT_S", Settings.timeout_s)),
        max_retries=int(os.getenv("MAGI_MAX_RETRIES", Settings.max_retries)),
        max_tokens=int(os.getenv("MAGI_MAX_TOKENS", Settings.max_tokens)),
        model_overrides=overrides,
    )
