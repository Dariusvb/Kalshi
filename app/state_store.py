from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class StateStore:
    path: str

    def __post_init__(self) -> None:
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            direction TEXT NOT NULL,
            base_conviction REAL NOT NULL,
            social_bonus REAL NOT NULL,
            final_conviction REAL NOT NULL,
            stake_dollars REAL NOT NULL,
            action TEXT NOT NULL,
            reasons TEXT,
            dry_run INTEGER NOT NULL DEFAULT 1
        )
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS fills (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            ticker TEXT NOT NULL,
            pnl_dollars REAL NOT NULL DEFAULT 0,
            result TEXT
        )
        """)
        self.conn.commit()

    def insert_decision(self, row: Dict[str, Any]) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO decisions (
                ts, ticker, direction, base_conviction, social_bonus, final_conviction,
                stake_dollars, action, reasons, dry_run
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["ts"], row["ticker"], row["direction"],
            row["base_conviction"], row["social_bonus"], row["final_conviction"],
            row["stake_dollars"], row["action"], row.get("reasons", ""),
            1 if row.get("dry_run", True) else 0
        ))
        self.conn.commit()

    def recent_decisions(self, limit: int = 50) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in cur.fetchall()]

    def close(self) -> None:
        self.conn.close()
