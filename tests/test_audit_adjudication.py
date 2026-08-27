import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from stockpilot.adjudication import adjudicate_future_test
from stockpilot.audit import (
    bootstrap_audit_chain,
    create_protocol_addendum,
    sha256_file,
    verify_audit_chain,
    verify_protocol_addendum,
)
from stockpilot.config import Settings
from stockpilot.data import save_panel


class AuditAndAdjudicationTests(unittest.TestCase):
    def _protocol(self, root: Path) -> tuple[Path, Path, Path, Path]:
        market = save_panel(
            pd.DataFrame(
                {
                    "date": ["2026-01-01"],
                    "symbol": ["000001"],
                    "open": [10.0],
                    "high": [10.0],
                    "low": [10.0],
                    "close": [10.0],
                    "volume": [1000],
                    "amount": [10000],
                }
            ),
            root / "market.csv",
        )
        membership = root / "membership.csv"
        exposure = root / "exposure.csv"
        selected = root / "selected.json"
        membership.write_text("membership", encoding="utf-8")
        exposure.write_text("exposure", encoding="utf-8")
        selected.write_text('{"model_name":"ridge"}', encoding="utf-8")
        manifest = root / "manifest.json"
        manifest.write_text(
            json.dumps(
                {
                    "mode": "shadow_observation",
                    "evaluation_start": "2026-01-02",
                    "minimum_trading_days": 2,
                    "selected_config": {"model_name": "ridge"},
                    "frozen_inputs": {
                        "market": {"path": str(market), "sha256": sha256_file(market)},
                        "membership": {
                            "path": str(membership),
                            "sha256": sha256_file(membership),
                        },
                        "exposure": {
                            "path": str(exposure),
                            "sha256": sha256_file(exposure),
                        },
                        "selected_config": {
                            "path": str(selected),
                            "sha256": sha256_file(selected),
                        },
                    },
                }
            ),
            encoding="utf-8",
        )
        addendum = root / "addendum.json"
        create_protocol_addendum(
            manifest,
            addendum,
            Settings(horizon=1, rebalance_every=1),
        )
        return manifest, addendum, market, selected

    def test_hash_chain_detects_tracked_file_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, addendum, _, tracked = self._protocol(root)
            chain = root / "chain.jsonl"
            bootstrap_audit_chain(
                chain,
                [(manifest, "manifest"), (addendum, "addendum"), (tracked, "signal")],
            )
            self.assertTrue(verify_audit_chain(chain)["intact"])
            self.assertTrue(all(verify_protocol_addendum(addendum).values()))
            tracked.write_text("changed", encoding="utf-8")
            self.assertFalse(verify_audit_chain(chain, raise_on_error=False)["intact"])

    def test_ready_window_writes_non_executable_decision_lock(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest, addendum, market, _ = self._protocol(root)
            bars = root / "bars"
            signals = root / "signals"
            bars.mkdir()
            signals.mkdir()
            for date in ["2026-01-02", "2026-01-03"]:
                pd.DataFrame({"date": [date], "symbol": ["000001"]}).to_csv(
                    bars / f"{date}.csv", index=False
                )
            signal = signals / "2026-01-02.csv"
            pd.DataFrame({"symbol": ["000001"]}).to_csv(signal, index=False)
            ledger = root / "ledger.csv"
            pd.DataFrame(
                {
                    "signal_date": ["2026-01-02"],
                    "exit_date": ["2026-01-03"],
                    "net_return": [0.02],
                    "benchmark_return": [0.01],
                    "rank_ic": [0.1],
                    "exposure_coverage": [1.0],
                }
            ).to_csv(ledger, index=False)
            chain = root / "chain.jsonl"
            files = [
                (manifest, "manifest"),
                (addendum, "addendum"),
                (signal, "signal"),
            ]
            files.extend((path, "bar") for path in bars.glob("*.csv"))
            bootstrap_audit_chain(chain, files)
            decision = root / "decision.json"
            report = adjudicate_future_test(
                manifest,
                addendum,
                market,
                bars,
                signals,
                ledger,
                root / "status.json",
                decision,
                chain,
            )
            self.assertTrue(report["ready_for_adjudication"])
            self.assertTrue(report["passed"])
            self.assertFalse(report["execution_authorized"])
            self.assertTrue(decision.exists())


if __name__ == "__main__":
    unittest.main()
