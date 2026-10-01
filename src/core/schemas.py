"""Pydantic models for structured agent output and consensus results."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

MAX_SENTENCES = 3

_VOTE_ALIASES = {
    "APPROVE": "APPROVE", "APPROVED": "APPROVE", "YES": "APPROVE",
    "ACCEPT": "APPROVE", "GRANT": "APPROVE", "GRANTED": "APPROVE",
    "REJECT": "REJECT", "REJECTED": "REJECT", "NO": "REJECT",
    "DENY": "REJECT", "DENIED": "REJECT",
}
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class AgentVote(BaseModel):
    """The strict JSON contract every MAGI node must satisfy.

    Validators are deliberately forgiving about *formatting* slips common in
    small local models (lowercase votes, 0-100 confidence, verbose reasoning)
    while still rejecting genuinely unusable output.
    """

    model_config = ConfigDict(extra="ignore")

    agent_name: str = Field(min_length=1)
    vote: Literal["APPROVE", "REJECT"]
    confidence_score: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)

    @field_validator("vote", mode="before")
    @classmethod
    def _normalise_vote(cls, v: object) -> object:
        if isinstance(v, str):
            return _VOTE_ALIASES.get(v.strip().upper(), v)
        return v

    @field_validator("confidence_score", mode="before")
    @classmethod
    def _coerce_confidence(cls, v: object) -> object:
        if isinstance(v, str):
            v = float(v.strip().rstrip("%"))  # ValueError -> ValidationError
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = float(v)
            if 1.0 < v <= 100.0:  # model answered on a 0-100 scale
                v /= 100.0
            return min(max(v, 0.0), 1.0)
        return v

    @field_validator("reasoning")
    @classmethod
    def _limit_sentences(cls, v: str) -> str:
        v = " ".join(v.split())
        if not v:
            raise ValueError("reasoning is empty")
        return " ".join(_SENTENCE_SPLIT.split(v)[:MAX_SENTENCES])


class AgentResult(BaseModel):
    """Outcome of one agent's deliberation: a valid vote, or an error (abstain)."""

    agent_name: str
    vote: AgentVote | None = None
    error: str | None = None
    latency_s: float = 0.0
    attempts: int = 1

    @property
    def ok(self) -> bool:
        return self.vote is not None


Decision = Literal["APPROVE", "REJECT", "NO_CONSENSUS"]


class ConsensusResult(BaseModel):
    """Final arbiter output after the 2/3 majority calculation."""

    decision: Decision
    approve_count: int
    reject_count: int
    abstain_count: int
    avg_confidence: float
    threshold: int
    results: list[AgentResult]
