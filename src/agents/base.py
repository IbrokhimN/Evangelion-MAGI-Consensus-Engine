"""Base async agent and the consensus Arbiter."""
from __future__ import annotations

import asyncio
import inspect
import json
import logging
import math
import re
import time
from typing import Awaitable, Callable, Protocol

from pydantic import ValidationError

from src.core.llm_client import LLMError
from src.core.schemas import AgentResult, AgentVote, ConsensusResult

log = logging.getLogger(__name__)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_CODE_FENCE = re.compile(r"```(?:json)?", re.IGNORECASE)


class SupportsComplete(Protocol):
    async def complete(
        self, *, system: str, user: str, temperature: float, model: str | None = None
    ) -> str: ...


def extract_json_object(raw: str) -> dict:
    """Pull the first JSON object out of noisy model output.

    Handles <think> blocks, markdown fences and leading/trailing prose.
    """
    text = _CODE_FENCE.sub("", _THINK_BLOCK.sub("", raw)).strip()
    start = text.find("{")
    if start == -1:
        raise ValueError("no JSON object found in output")
    obj, _ = json.JSONDecoder().raw_decode(text[start:])
    if not isinstance(obj, dict):
        raise ValueError("JSON root is not an object")
    return obj


class BaseAgent:
    """One MAGI node: a persona + generation params + defensive JSON parsing."""

    def __init__(
        self,
        *,
        key: str,
        name: str,
        role: str,
        system_prompt: str,
        temperature: float,
        model: str | None = None,
        max_attempts: int = 2,
    ) -> None:
        self.key = key
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model = model
        self.max_attempts = max_attempts

    def _user_message(self, prompt: str, attempt: int) -> str:
        msg = f"PROPOSAL:\n{prompt}\n\nReply with ONE valid JSON object and nothing else."
        if attempt > 1:
            msg += (
                "\nYour previous reply was not valid. Use exactly the keys "
                "agent_name, vote (APPROVE|REJECT), confidence_score (0-1), reasoning."
            )
        return msg

    def parse(self, raw: str) -> AgentVote:
        data = extract_json_object(raw)
        data["agent_name"] = self.name  # never trust a hallucinated identity
        return AgentVote.model_validate(data)

    async def deliberate(self, prompt: str, llm: SupportsComplete) -> AgentResult:
        """Run the node. Never raises: failures become an abstaining AgentResult."""
        started = time.perf_counter()
        error = "UNKNOWN FAULT"
        attempts = 0
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            try:
                raw = await llm.complete(
                    system=self.system_prompt,
                    user=self._user_message(prompt, attempt),
                    temperature=self.temperature,
                    model=self.model,
                )
                vote = self.parse(raw)
                return AgentResult(
                    agent_name=self.name,
                    vote=vote,
                    latency_s=time.perf_counter() - started,
                    attempts=attempt,
                )
            except LLMError as exc:  # transport failure: retrying won't help
                error = str(exc)
                break
            except (ValueError, ValidationError) as exc:  # hallucinated / malformed
                error = f"MALFORMED OUTPUT: {type(exc).__name__}"
                log.warning("%s attempt %d parse failure: %s", self.name, attempt, exc)
        return AgentResult(
            agent_name=self.name,
            error=error,
            latency_s=time.perf_counter() - started,
            attempts=attempts,
        )


AgentCallback = Callable[[AgentResult], Awaitable[None] | None]


class ConsensusArbiter:
    """Runs all agents in parallel and computes the 2/3 majority."""

    def __init__(self, agents: list[BaseAgent], llm: SupportsComplete) -> None:
        self.agents = agents
        self._llm = llm

    @property
    def threshold(self) -> int:
        return math.ceil(len(self.agents) * 2 / 3)

    async def deliberate(
        self, prompt: str, on_agent_result: AgentCallback | None = None
    ) -> ConsensusResult:
        async def run(agent: BaseAgent) -> AgentResult:
            try:
                result = await agent.deliberate(prompt, self._llm)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - last line of defence
                result = AgentResult(agent_name=agent.name, error=f"UNHANDLED: {type(exc).__name__}")
            if on_agent_result is not None:
                try:
                    maybe = on_agent_result(result)
                    if inspect.isawaitable(maybe):
                        await maybe
                except Exception:  # noqa: BLE001 - UI errors must not kill the vote
                    log.exception("on_agent_result callback failed")
            return result

        results = await asyncio.gather(*(run(a) for a in self.agents))
        return self.tally(list(results))

    def tally(self, results: list[AgentResult]) -> ConsensusResult:
        votes = [r.vote for r in results if r.vote is not None]
        approve = [v for v in votes if v.vote == "APPROVE"]
        reject = [v for v in votes if v.vote == "REJECT"]

        if len(approve) >= self.threshold:
            decision, winners = "APPROVE", approve
        elif len(reject) >= self.threshold:
            decision, winners = "REJECT", reject
        else:
            decision, winners = "NO_CONSENSUS", votes

        avg = sum(v.confidence_score for v in winners) / len(winners) if winners else 0.0
        return ConsensusResult(
            decision=decision,
            approve_count=len(approve),
            reject_count=len(reject),
            abstain_count=len(results) - len(votes),
            avg_confidence=avg,
            threshold=self.threshold,
            results=results,
        )
