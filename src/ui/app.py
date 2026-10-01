"""Textual application: wires the widgets to the async consensus arbiter."""
from __future__ import annotations

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.widgets import Footer, Input

from src.agents.base import ConsensusArbiter
from src.agents.personalities import build_magi_agents
from src.core.config import Settings, load_settings
from src.core.llm_client import LLMClient
from src.core.schemas import AgentResult
from src.ui.widgets import AgentPanel, ConsensusPanel, NervHeader


class MagiApp(App[None]):
    """MAGI Consensus Engine terminal interface."""

    CSS_PATH = "styles.tcss"
    TITLE = "MAGI CONSENSUS ENGINE"

    BINDINGS = [
        Binding("escape", "focus_input", "Focus input"),
        Binding("ctrl+l", "reset", "Reset"),
        Binding("ctrl+q", "quit", "Quit"),
    ]

    def __init__(
        self,
        settings: Settings | None = None,
        arbiter: ConsensusArbiter | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        super().__init__()
        if arbiter is None:
            settings = settings or load_settings()
            arbiter = ConsensusArbiter(build_magi_agents(settings), LLMClient(settings))
        self.arbiter = arbiter
        self._initial_prompt = initial_prompt
        self._panels: dict[str, AgentPanel] = {}
        self._busy = False

    # ── layout ──────────────────────────────────────────────────────────
    def compose(self) -> ComposeResult:
        yield NervHeader(id="topbar")
        yield Input(placeholder="ENTER PROPOSAL FOR MAGI DELIBERATION >", id="prompt-input")
        with Horizontal(id="council"):
            for agent in self.arbiter.agents:
                yield AgentPanel(agent.name, agent.role, agent.key)
        yield ConsensusPanel(id="consensus")
        yield Footer()

    def on_mount(self) -> None:
        self._panels = {p.agent_name: p for p in self.query(AgentPanel)}
        self.query_one("#prompt-input", Input).focus()
        if self._initial_prompt:
            self.query_one("#prompt-input", Input).value = self._initial_prompt
            self._start(self._initial_prompt)

    # ── events / actions ────────────────────────────────────────────────
    def on_input_submitted(self, event: Input.Submitted) -> None:
        prompt = event.value.strip()
        if prompt and not self._busy:
            self._start(prompt)

    def action_focus_input(self) -> None:
        self.query_one("#prompt-input", Input).focus()

    def action_reset(self) -> None:
        if self._busy:
            return
        for panel in self._panels.values():
            panel.reset()
        self.query_one(ConsensusPanel).show_idle()
        inp = self.query_one("#prompt-input", Input)
        inp.value = ""
        inp.focus()

    # ── deliberation ────────────────────────────────────────────────────
    def _start(self, prompt: str) -> None:
        self._busy = True
        self.query_one("#prompt-input", Input).disabled = True
        for panel in self._panels.values():
            panel.set_analyzing()
        self.query_one(ConsensusPanel).show_pending()
        self.run_deliberation(prompt)

    async def _on_agent_result(self, result: AgentResult) -> None:
        panel = self._panels.get(result.agent_name)
        if panel is not None:
            panel.set_result(result)

    @work(exclusive=True, group="deliberation")
    async def run_deliberation(self, prompt: str) -> None:
        consensus = self.query_one(ConsensusPanel)
        try:
            outcome = await self.arbiter.deliberate(prompt, on_agent_result=self._on_agent_result)
            consensus.show_result(outcome)
        except Exception as exc:  # noqa: BLE001 - surface any failure in the UI
            consensus.show_error(type(exc).__name__)
        finally:
            self._busy = False
            inp = self.query_one("#prompt-input", Input)
            inp.disabled = False
            inp.focus()
