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

    def _table_columns(self, table_name: str) -> set[str]:
        cur = self.conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        rows = cur.fetchall()
        # row format: cid, name, type, notnull, dflt_value, pk
        return {str(r[1]) for r in rows}

    def _ensure_column(
        self,
        table_name: str,
        column_name: str,
        column_sql_type: str,
        default_sql: Optional[str] = None,
    ) -> None:
        cols = self._table_columns(table_name)
        if column_name in cols:
            return

        cur = self.conn.cursor()
        sql = f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql_type}"
        if default_sql is not None:
            sql += f" DEFAULT {default_sql}"
        cur.execute(sql)
        self.conn.commit()

    def _init_db(self) -> None:
        cur = self.conn.cursor()

        # Core decisions table (original base schema)
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

        # Existing fills table (kept for compatibility / optional usage)
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

        # ---- Backward-compatible schema migrations for outcome/learning fields ----
        self._ensure_column("decisions", "realized_pnl", "REAL", None)
        self._ensure_column("decisions", "won", "INTEGER", None)          # 1 / 0 / NULL
        self._ensure_column("decisions", "resolved_ts", "TEXT", None)
        self._ensure_column("decisions", "market_category", "TEXT", None)

        # ---- New fields for news/trend scoring and setup diagnostics ----
        self._ensure_column("decisions", "news_score", "REAL", None)
        self._ensure_column("decisions", "news_confidence", "REAL", None)
        self._ensure_column("decisions", "news_regime", "TEXT", None)
        self._ensure_column("decisions", "spread_cents", "INTEGER", None)

        # Helpful indexes
        cur = self.conn.cursor()
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ts ON decisions(ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_ticker ON decisions(ticker)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_action ON decisions(action)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_dry_run ON decisions(dry_run)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_resolved_ts ON decisions(resolved_ts)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_market_category ON decisions(market_category)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_decisions_news_regime ON decisions(news_regime)")
        self.conn.commit()

    def insert_decision(self, row: Dict[str, Any]) -> int:
        """
        Inserts a decision row.
        Supports original fields plus outcome/news fields.
        Returns inserted decision id.
        """
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO decisions (
                ts, ticker, direction, base_conviction, social_bonus, final_conviction,
                stake_dollars, action, reasons, dry_run,
                realized_pnl, won, resolved_ts, market_category,
                news_score, news_confidence, news_regime, spread_cents
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            row["ts"],
            row["ticker"],
            row["direction"],
            float(row["base_conviction"]),
            float(row["social_bonus"]),
            float(row["final_conviction"]),
            float(row["stake_dollars"]),
            row["action"],
            row.get("reasons", ""),
            1 if row.get("dry_run", True) else 0,
            self._maybe_float(row.get("realized_pnl")),
            self._normalize_won(row.get("won")),
            row.get("resolved_ts"),
            row.get("market_category"),
            self._maybe_float(row.get("news_score")),
            self._maybe_float(row.get("news_confidence")),
            row.get("news_regime"),
            self._maybe_int(row.get("spread_cents")),
        ))
        self.conn.commit()
        return int(cur.lastrowid)

    def update_decision_outcome(
        self,
        *,
        decision_id: Optional[int] = None,
        ticker: Optional[str] = None,
        ts: Optional[str] = None,
        realized_pnl: Optional[float] = None,
        won: Optional[bool | int] = None,
        resolved_ts: Optional[str] = None,
        market_category: Optional[str] = None,
    ) -> int:
        """
        Update a decision after outcome is known (paper trade or live trade result).

        Preferred:
            decision_id=...

        Fallback targeting (less precise):
            ticker + ts
        """
        if decision_id is None and not (ticker and ts):
            raise ValueError("Provide decision_id OR (ticker and ts) to update outcome")

        updates = []
        params: list[Any] = []

        if realized_pnl is not None:
            updates.append("realized_pnl = ?")
            params.append(float(realized_pnl))

        if won is not None:
            updates.append("won = ?")
            params.append(self._normalize_won(won))

        if resolved_ts is not None:
            updates.append("resolved_ts = ?")
            params.append(resolved_ts)

        if market_category is not None:
            updates.append("market_category = ?")
            params.append(market_category)

        if not updates:
            return 0

        sql = f"UPDATE decisions SET {', '.join(updates)} "

        if decision_id is not None:
            sql += "WHERE id = ?"
            params.append(int(decision_id))
        else:
            sql += "WHERE ticker = ? AND ts = ?"
            params.extend([ticker, ts])

        cur = self.conn.cursor()
        cur.execute(sql, params)
        self.conn.commit()
        return int(cur.rowcount)

    def recent_decisions(self, limit: int = 50) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM decisions ORDER BY id DESC LIMIT ?", (int(limit),))
        return [dict(r) for r in cur.fetchall()]

    def recent_resolved_trades(self, limit: int = 50, dry_run: Optional[bool] = None) -> List[dict]:
        """
        Returns only resolved trade actions (TRADE_*) with realized outcome fields populated (if available).
        Useful for learning.py rolling win rate / expectancy.
        """
        cur = self.conn.cursor()

        if dry_run is None:
            cur.execute("""
                SELECT *
                FROM decisions
                WHERE action LIKE 'TRADE_%'
                  AND resolved_ts IS NOT NULL
                ORDER BY id DESC
                LIMIT ?
            """, (int(limit),))
        else:
            cur.execute("""
                SELECT *
                FROM decisions
                WHERE action LIKE 'TRADE_%'
                  AND resolved_ts IS NOT NULL
                  AND dry_run = ?
                ORDER BY id DESC
                LIMIT ?
            """, (1 if dry_run else 0, int(limit)))

        return [dict(r) for r in cur.fetchall()]

    def unresolved_trade_decisions(self, limit: int = 100, dry_run: Optional[bool] = None) -> List[dict]:
        """
        Trade decisions that have not yet been resolved. Useful for paper-trade resolver.
        """
        cur = self.conn.cursor()

        if dry_run is None:
            cur.execute("""
                SELECT *
                FROM decisions
                WHERE action LIKE 'TRADE_%'
                  AND resolved_ts IS NULL
                ORDER BY id ASC
                LIMIT ?
            """, (int(limit),))
        else:
            cur.execute("""
                SELECT *
                FROM decisions
                WHERE action LIKE 'TRADE_%'
                  AND resolved_ts IS NULL
                  AND dry_run = ?
                ORDER BY id ASC
                LIMIT ?
            """, (1 if dry_run else 0, int(limit)))

        return [dict(r) for r in cur.fetchall()]

    def aggregate_mode_stats(self, lookback_limit: int = 200) -> Dict[str, Dict[str, Any]]:
        """
        Quick summary split by simulated vs live decisions.
        Includes wins/losses/PnL if outcomes have been written.
        """
        rows = self.recent_decisions(lookback_limit)

        def _summarize(sub: List[dict]) -> Dict[str, Any]:
            total = len(sub)
            trades = [r for r in sub if str(r.get("action", "")).startswith("TRADE")]
            skips = [r for r in sub if r.get("action") == "SKIP"]
            errors = [r for r in sub if r.get("action") == "ERROR"]

            avg_conv = round(
                sum(float(r.get("final_conviction", 0.0)) for r in sub) / total, 2
            ) if total else 0.0

            avg_stake = round(
                sum(float(r.get("stake_dollars", 0.0)) for r in trades) / len(trades), 2
            ) if trades else 0.0

            resolved = [r for r in trades if r.get("resolved_ts") is not None]
            wins = [r for r in resolved if r.get("won") == 1]
            losses = [r for r in resolved if r.get("won") == 0]

            pnl_values = [
                float(r["realized_pnl"])
                for r in resolved
                if r.get("realized_pnl") is not None
            ]
            pnl_total = round(sum(pnl_values), 2) if pnl_values else None

            win_rate = None
            if (len(wins) + len(losses)) > 0:
                win_rate = round((len(wins) / (len(wins) + len(losses))) * 100.0, 1)

            expectancy = None
            if pnl_values:
                expectancy = round(sum(pnl_values) / len(pnl_values), 4)

            return {
                "count": total,
                "trades": len(trades),
                "skips": len(skips),
                "errors": len(errors),
                "resolved_trades": len(resolved),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": win_rate,
                "pnl_total": pnl_total,
                "expectancy": expectancy,
                "avg_conviction": avg_conv,
                "avg_stake": avg_stake,
            }

        sim = [r for r in rows if bool(r.get("dry_run", 1))]
        live = [r for r in rows if not bool(r.get("dry_run", 1))]

        return {"sim": _summarize(sim), "live": _summarize(live)}

    def insert_fill(self, row: Dict[str, Any]) -> int:
        """
        Optional legacy/simple fills logger.
        """
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO fills (ts, ticker, pnl_dollars, result)
            VALUES (?, ?, ?, ?)
        """, (
            row["ts"],
            row["ticker"],
            float(row.get("pnl_dollars", 0.0)),
            row.get("result"),
        ))
        self.conn.commit()
        return int(cur.lastrowid)

    def recent_fills(self, limit: int = 50) -> List[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM fills ORDER BY id DESC LIMIT ?", (int(limit),))
        return [dict(r) for r in cur.fetchall()]

    def _normalize_won(self, value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            return 1 if value else 0
        if isinstance(value, int):
            if value in (0, 1):
                return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"1", "true", "win", "won", "yes"}:
                return 1
            if v in {"0", "false", "loss", "lost", "no"}:
                return 0
        raise ValueError(f"Invalid won value: {value!r}")

    def _maybe_float(self, value: Any) -> Optional[float]:
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _maybe_int(self, value: Any) -> Optional[int]:
        if value is None or value == "":
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def close(self) -> None:
        self.conn.close()
