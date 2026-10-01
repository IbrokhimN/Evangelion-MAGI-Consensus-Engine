"""Custom Textual widgets: header, agent columns and the consensus arbiter panel."""
from __future__ import annotations

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.timer import Timer
from textual.widgets import Static

from src.core.schemas import AgentResult, ConsensusResult

GREEN = "#00ff66"
RED = "#ff2a2a"
AMBER = "#ffcc00"
ORANGE = "#ff9d00"

_GLYPHS: dict[str, list[str]] = {
    "A": [" █████╗ ", "██╔══██╗", "███████║", "██╔══██║", "██║  ██║", "╚═╝  ╚═╝"],
    "C": [" ██████╗", "██╔════╝", "██║     ", "██║     ", "╚██████╗", " ╚═════╝"],
    "D": ["██████╗ ", "██╔══██╗", "██║  ██║", "██║  ██║", "██████╔╝", "╚═════╝ "],
    "E": ["███████╗", "██╔════╝", "█████╗  ", "██╔══╝  ", "███████╗", "╚══════╝"],
    "G": [" ██████╗ ", "██╔════╝ ", "██║  ███╗", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "I": ["██╗", "██║", "██║", "██║", "██║", "╚═╝"],
    "N": ["███╗   ██╗", "████╗  ██║", "██╔██╗ ██║", "██║╚██╗██║", "██║ ╚████║", "╚═╝  ╚═══╝"],
    "O": [" ██████╗ ", "██╔═══██╗", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
    "R": ["██████╗ ", "██╔══██╗", "██████╔╝", "██╔══██╗", "██║  ██║", "╚═╝  ╚═╝"],
    "S": ["███████╗", "██╔════╝", "███████╗", "╚════██║", "███████║", "╚══════╝"],
    "T": ["████████╗", "╚══██╔══╝", "   ██║   ", "   ██║   ", "   ██║   ", "   ╚═╝   "],
    "U": ["██╗   ██╗", "██║   ██║", "██║   ██║", "██║   ██║", "╚██████╔╝", " ╚═════╝ "],
}
_ROWS = 6


def _word_rows(word: str) -> list[str]:
    rows = [""] * _ROWS
    for ch in word:
        glyph = _GLYPHS[ch]
        width = max(len(g) for g in glyph)
        rows = [r + glyph[i].ljust(width) for i, r in enumerate(rows)]
    return rows


def render_banner(words: list[str], max_width: int) -> list[str]:
    """Render words as ASCII art; stack them vertically if too wide for one line."""
    per_word = [_word_rows(w) for w in words]
    one_line = ["   ".join(w[i] for w in per_word) for i in range(_ROWS)]
    if max(len(r) for r in one_line) <= max_width:
        return one_line
    stacked: list[str] = []
    for idx, rows in enumerate(per_word):
        if idx:
            stacked.append("")
        stacked.extend(rows)
    return stacked


class NervHeader(Static):
    """Blinking system status bar."""

    def on_mount(self) -> None:
        self.update(Text("SYSTEM: MAGI // STATUS: SECURE"))
        self.border_title = "╣ NERV ╠"
        self.set_interval(0.7, lambda: self.toggle_class("blink-off"))


class AgentPanel(Vertical):
    """One MAGI node column with live status, reasoning and vote."""

    _FRAMES = ("[ANALYZING.  ]", "[ANALYZING.. ]", "[ANALYZING...]", "[ANALYZING   ]")

    def __init__(self, agent_name: str, role: str, key: str) -> None:
        super().__init__(classes=key, id=f"panel-{key}")
        self.agent_name = agent_name
        self.border_title = f"╣ {agent_name} ╠"
        self.border_subtitle = f"╚ {role} ╝"
        self._timer: Timer | None = None
        self._frame = 0

    def compose(self) -> ComposeResult:
        yield Static(Text("[STANDBY]"), classes="agent-status")
        yield Static(Text(""), classes="agent-body")
        yield Static(Text(""), classes="agent-vote")

    def _status(self) -> Static:
        return self.query_one(".agent-status", Static)

    def _body(self) -> Static:
        return self.query_one(".agent-body", Static)

    def _vote(self) -> Static:
        return self.query_one(".agent-vote", Static)

    def _stop_timer(self) -> None:
        if self._timer is not None:
            self._timer.stop()
            self._timer = None

    def _tick(self) -> None:
        self._frame = (self._frame + 1) % len(self._FRAMES)
        self._status().update(Text(self._FRAMES[self._frame]))

    def reset(self) -> None:
        self._stop_timer()
        self._status().update(Text("[STANDBY]"))
        self._body().update(Text(""))
        self._vote().update(Text(""))

    def set_analyzing(self) -> None:
        self._stop_timer()
        self._frame = 0
        self._status().update(Text(self._FRAMES[0]))
        self._body().update(Text(""))
        self._vote().update(Text(""))
        self._timer = self.set_interval(0.35, self._tick)

    def set_result(self, result: AgentResult) -> None:
        self._stop_timer()
        if result.vote is None:
            self._status().update(Text("[FAULT // NO VOTE]", style=f"bold {AMBER}"))
            self._body().update(Text(result.error or "UNKNOWN FAULT", style=AMBER))
            self._vote().update(Text("VOTE: ABSTAIN", style=f"bold {AMBER}"))
            return

        v = result.vote
        colour = GREEN if v.vote == "APPROVE" else RED
        self._status().update(Text(f"[RESPONSE // {result.latency_s:.1f}s]"))
        self._body().update(Text(v.reasoning))
        pct = round(v.confidence_score * 100)
        filled = round(v.confidence_score * 10)
        bar = "█" * filled + "░" * (10 - filled)
        text = Text()
        text.append(f"╠═ VOTE: {v.vote} ═╣\n", style=f"bold {colour}")
        text.append(f"CONF {bar} {pct}%", style=colour)
        self._vote().update(text)


class ConsensusPanel(Static):
    """Bottom arbiter panel that reveals the final decision."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._result: ConsensusResult | None = None
        self.border_title = "╣ MAGI CONSENSUS ╠"

    def _set_state(self, css: str | None) -> None:
        self.remove_class("approve", "reject", "nocon")
        if css:
            self.add_class(css)

    def on_mount(self) -> None:
        self.show_idle()

    def show_idle(self) -> None:
        self._result = None
        self._set_state(None)
        self.update(Text("AWAITING PROPOSAL ...", style=f"bold {ORANGE}", justify="center"))

    def show_pending(self) -> None:
        self._result = None
        self._set_state(None)
        self.update(
            Text("╠═══ DELIBERATING // 2/3 MAJORITY REQUIRED ═══╣", style=f"bold {ORANGE}", justify="center")
        )

    def show_error(self, message: str) -> None:
        self._result = None
        self._set_state("nocon")
        self.update(Text(f"SYSTEM FAULT: {message}", style=f"bold {AMBER}", justify="center"))

    def show_result(self, result: ConsensusResult) -> None:
        self._result = result
        self._render_result()

    def on_resize(self) -> None:
        if self._result is not None:
            self._render_result()

    def _render_result(self) -> None:
        r = self._result
        assert r is not None
        width = self.size.width or 80
        if r.decision == "APPROVE":
            words, colour, css = ["ACCESS", "GRANTED"], GREEN, "approve"
            note = "PROPOSAL APPROVED BY MAGI MAJORITY"
        elif r.decision == "REJECT":
            words, colour, css = ["ACCESS", "DENIED"], RED, "reject"
            note = "▲▲ WARNING // PROPOSAL REJECTED BY MAGI MAJORITY ▲▲"
        else:
            words, colour, css = ["NO", "CONSENSUS"], AMBER, "nocon"
            note = f"INSUFFICIENT VALID VOTES // {r.threshold}/3 MAJORITY NOT REACHED"

        self._set_state(css)
        text = Text(justify="center")
        for line in render_banner(words, width - 2):
            text.append(line + "\n", style=f"bold {colour}")
        text.append("\n")
        text.append(note + "\n", style=f"bold {colour}")
        text.append(
            f"╠═══ APPROVE {r.approve_count} ║ REJECT {r.reject_count} "
            f"║ ABSTAIN {r.abstain_count} ═══╣\n",
            style=colour,
        )
        if r.decision != "NO_CONSENSUS":
            text.append(f"MEAN CONFIDENCE {r.avg_confidence * 100:.0f}%", style=colour)
        self.update(text)
