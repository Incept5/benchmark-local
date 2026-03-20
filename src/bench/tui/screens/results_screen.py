"""Results screen: DataTable comparisons, export."""

from __future__ import annotations

import json
from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Vertical
from textual.screen import Screen
from textual.widgets import (
    Button,
    DataTable,
    Footer,
    Header,
    Label,
    TabbedContent,
    TabPane,
)

from bench.store import SessionResult


class ResultsScreen(Screen):
    """Screen for viewing benchmark results."""

    BINDINGS = [
        ("escape", "app.pop_screen", "Back"),
        ("e", "export", "Export JSON"),
    ]

    DEFAULT_CSS = """
    ResultsScreen {
        layout: vertical;
    }
    #results-tabs {
        height: 1fr;
        margin: 1 2;
    }
    DataTable {
        height: 1fr;
    }
    #export-btn {
        dock: bottom;
        width: 100%;
        margin: 1 2;
    }
    """

    def __init__(self, session: SessionResult | None = None) -> None:
        super().__init__()
        self._session = session

    @property
    def session(self) -> SessionResult | None:
        if self._session is not None:
            return self._session
        return getattr(self.app, "session_result", None)

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)

        with TabbedContent(id="results-tabs"):
            with TabPane("Summary", id="tab-summary"):
                yield DataTable(id="summary-table")
            with TabPane("By Family", id="tab-family"):
                yield DataTable(id="family-table")
            with TabPane("By Prompt", id="tab-prompt"):
                yield DataTable(id="prompt-table")
            with TabPane("Export", id="tab-export"):
                yield Label("Press 'e' or click Export to save results as JSON.")
                yield Button("Export JSON", id="export-btn", variant="primary")

        yield Footer()

    def on_mount(self) -> None:
        self._populate_summary()
        self._populate_family()
        self._populate_prompt()

    def _populate_summary(self) -> None:
        """Populate the summary DataTable."""
        session = self.session
        if not session:
            return

        table = self.query_one("#summary-table", DataTable)
        table.add_columns(
            "Model", "Quant", "TTFT (ms)", "tok/s", "tok/W",
            "Memory (MB)", "Perplexity", "MMLU %",
        )

        for key, metrics in session.aggregated.items():
            repo, quant = key.split("|", 1)
            short_name = repo.split("/")[-1]

            ttft = metrics.get("ttft_ms", {})
            tps = metrics.get("tokens_per_sec", {})
            tpw = metrics.get("tokens_per_watt", {})
            mem = metrics.get("peak_memory_bytes", {})

            q = session.quality.get(key, {})
            ppl = q.get("perplexity")
            mmlu = q.get("mmlu_accuracy")

            table.add_row(
                short_name,
                quant,
                f"{ttft.get('median', 0):.1f}" if ttft else "—",
                f"{tps.get('median', 0):.1f}" if tps else "—",
                f"{tpw.get('median', 0):.2f}" if tpw else "—",
                f"{mem.get('median', 0) / 1024**2:.0f}" if mem else "—",
                f"{ppl:.2f}" if ppl else "—",
                f"{mmlu * 100:.1f}" if mmlu is not None else "—",
            )

    def _populate_family(self) -> None:
        """Populate the family comparison DataTable."""
        session = self.session
        if not session:
            return

        table = self.query_one("#family-table", DataTable)
        table.add_columns(
            "Family", "Quant", "TTFT (ms)", "tok/s", "Memory (MB)",
            "Perplexity", "MMLU %", "Avg Similarity",
        )

        # Group by family from config snapshot
        families = session.config_snapshot.get("model_families", [])
        for fam in families:
            for var in fam.get("variants", []):
                key = f"{var['repo']}|{var['quant']}"
                metrics = session.aggregated.get(key, {})
                q = session.quality.get(key, {})

                ttft = metrics.get("ttft_ms", {})
                tps = metrics.get("tokens_per_sec", {})
                mem = metrics.get("peak_memory_bytes", {})
                ppl = q.get("perplexity")
                mmlu = q.get("mmlu_accuracy")

                sim = q.get("output_similarity", {})
                avg_sim = (
                    sum(sim.values()) / len(sim) if sim else None
                )

                table.add_row(
                    fam["name"],
                    var["quant"],
                    f"{ttft.get('median', 0):.1f}" if ttft else "—",
                    f"{tps.get('median', 0):.1f}" if tps else "—",
                    f"{mem.get('median', 0) / 1024**2:.0f}" if mem else "—",
                    f"{ppl:.2f}" if ppl else "—",
                    f"{mmlu * 100:.1f}" if mmlu is not None else "—",
                    f"{avg_sim:.3f}" if avg_sim is not None else "—",
                )

    def _populate_prompt(self) -> None:
        """Populate the per-prompt DataTable."""
        session = self.session
        if not session:
            return

        table = self.query_one("#prompt-table", DataTable)
        table.add_columns(
            "Model", "Quant", "Prompt", "TTFT (ms)", "tok/s",
            "Tokens", "CI", "CV%",
        )

        # Group runs by variant + prompt
        from collections import defaultdict
        from bench.stats import aggregate

        grouped: dict[tuple[str, str, str], list] = defaultdict(list)
        for run in session.runs:
            if not run.get("is_warmup", False):
                key = (run["variant_repo"], run["variant_quant"], run["prompt_id"])
                grouped[key].append(run)

        for (repo, quant, prompt_id), runs in grouped.items():
            short_name = repo.split("/")[-1]
            tps_values = [r["tokens_per_sec"] for r in runs]
            agg = aggregate(tps_values)

            table.add_row(
                short_name,
                quant,
                prompt_id,
                f"{runs[0]['ttft_ms']:.1f}",
                f"{agg.median:.1f}",
                str(runs[0].get("tokens_generated", 0)),
                agg.ci_str,
                f"{agg.cv_percent:.1f}%" + (" !" if agg.unreliable else ""),
            )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "export-btn":
            self.action_export()

    def action_export(self) -> None:
        """Export results to JSON and HTML."""
        session = self.session
        if not session:
            return

        from bench.report import generate_report
        from bench.store import save_session

        output_dir = session.config_snapshot.get("output_dir", "results")
        path = save_session(session, output_dir)
        html_path = generate_report(session, path.with_suffix(".html"))
        self.notify(f"Exported to {path}\nHTML report: {html_path}")
