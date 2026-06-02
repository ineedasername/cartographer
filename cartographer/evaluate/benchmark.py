"""
Lightweight benchmark database wrapper for storing evaluation runs and results.

Model-agnostic SQLite backend. Schema lifted from medgemma/benchmark_db.py
and generalized to work with any model/benchmark combination.

Source: medgemma/benchmark_db.py (init_db, ensure_benchmark, ensure_model, get_db)
"""

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional, Union


class BenchmarkDB:
    """SQLite wrapper for benchmark evaluation storage."""

    SCHEMA_VERSION = "1.0"

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT
    );

    CREATE TABLE IF NOT EXISTS benchmarks (
        benchmark_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        source TEXT,
        split TEXT,
        description TEXT,
        n_examples INTEGER,
        n_options INTEGER DEFAULT 4,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS models (
        model_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        variant TEXT,
        architecture TEXT,
        params TEXT,
        layers INTEGER,
        heads INTEGER,
        hidden_dim INTEGER,
        description TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS runs (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER REFERENCES models(model_id),
        benchmark_id INTEGER REFERENCES benchmarks(benchmark_id),
        label TEXT NOT NULL,
        intervention_type TEXT,
        intervention_params TEXT,
        n_examples INTEGER,
        n_correct INTEGER,
        accuracy REAL,
        elapsed_seconds REAL,
        notes TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER REFERENCES runs(run_id),
        example_id TEXT,
        predicted TEXT,
        gold TEXT,
        correct INTEGER,
        confidence REAL,
        probabilities TEXT,
        metadata TEXT,
        UNIQUE(run_id, example_id)
    );
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        self.conn = sqlite3.connect(str(self.path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript(self.SCHEMA)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", self.SCHEMA_VERSION),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- Ensure helpers (idempotent upsert) --

    def ensure_benchmark(
        self,
        name: str,
        source: Optional[str] = None,
        split: Optional[str] = None,
        n_examples: Optional[int] = None,
        description: Optional[str] = None,
    ) -> int:
        """Return benchmark_id, creating the row if needed."""
        row = self.conn.execute(
            "SELECT benchmark_id FROM benchmarks WHERE name=? AND split=?",
            (name, split),
        ).fetchone()
        if row:
            return row["benchmark_id"]
        cur = self.conn.execute(
            "INSERT INTO benchmarks (name, source, split, n_examples, description) "
            "VALUES (?, ?, ?, ?, ?)",
            (name, source, split, n_examples, description),
        )
        self.conn.commit()
        return cur.lastrowid

    def ensure_model(
        self,
        name: str,
        variant: Optional[str] = None,
        architecture: Optional[str] = None,
        params: Optional[str] = None,
        layers: Optional[int] = None,
        heads: Optional[int] = None,
        hidden_dim: Optional[int] = None,
    ) -> int:
        """Return model_id, creating the row if needed."""
        row = self.conn.execute(
            "SELECT model_id FROM models WHERE name=?", (name,)
        ).fetchone()
        if row:
            return row["model_id"]
        cur = self.conn.execute(
            "INSERT INTO models (name, variant, architecture, params, layers, heads, hidden_dim) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (name, variant, architecture, params, layers, heads, hidden_dim),
        )
        self.conn.commit()
        return cur.lastrowid

    # -- Run management --

    def create_run(
        self,
        model_id: int,
        benchmark_id: int,
        label: str,
        intervention_type: Optional[str] = None,
        intervention_params: Optional[dict] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Create a new evaluation run, return run_id."""
        cur = self.conn.execute(
            "INSERT INTO runs (model_id, benchmark_id, label, intervention_type, "
            "intervention_params, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (
                model_id,
                benchmark_id,
                label,
                intervention_type,
                json.dumps(intervention_params) if intervention_params else None,
                notes,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def add_result(
        self,
        run_id: int,
        example_id: str,
        predicted: str,
        gold: str,
        confidence: Optional[float] = None,
        probabilities: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ):
        """Record a single prediction result."""
        correct = int(predicted == gold)
        self.conn.execute(
            "INSERT OR IGNORE INTO results "
            "(run_id, example_id, predicted, gold, correct, confidence, probabilities, metadata) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                str(example_id),
                predicted,
                gold,
                correct,
                confidence,
                json.dumps(probabilities) if probabilities else None,
                json.dumps(metadata) if metadata else None,
            ),
        )

    def finalize_run(self, run_id: int, elapsed_seconds: Optional[float] = None):
        """Compute and store aggregate stats for a run."""
        row = self.conn.execute(
            "SELECT COUNT(*) as n, SUM(correct) as nc FROM results WHERE run_id=?",
            (run_id,),
        ).fetchone()
        n = row["n"]
        nc = row["nc"] or 0
        acc = (nc / n * 100) if n > 0 else 0.0
        self.conn.execute(
            "UPDATE runs SET n_examples=?, n_correct=?, accuracy=?, elapsed_seconds=? "
            "WHERE run_id=?",
            (n, nc, acc, elapsed_seconds, run_id),
        )
        self.conn.commit()
        return {"n_examples": n, "n_correct": nc, "accuracy": acc}

    # -- Queries --

    def get_run(self, run_id: int) -> Optional[dict]:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return dict(row) if row else None

    def list_runs(self, benchmark_id: Optional[int] = None) -> list:
        if benchmark_id is not None:
            rows = self.conn.execute(
                "SELECT * FROM runs WHERE benchmark_id=? ORDER BY run_id", (benchmark_id,)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM runs ORDER BY run_id").fetchall()
        return [dict(r) for r in rows]

    def get_results(self, run_id: int) -> list:
        rows = self.conn.execute(
            "SELECT * FROM results WHERE run_id=? ORDER BY example_id", (run_id,)
        ).fetchall()
        return [dict(r) for r in rows]
