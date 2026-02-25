from __future__ import annotations

import argparse
import os
import re
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


CLEAR = "\033[2J\033[H"
DIM = "\033[2m"
BOLD = "\033[1m"
RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"


def _safe_float(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    try:
        return float(v)
    except Exception:
        return None


def _safe_int(v: Any) -> Optional[int]:
    if v is None or v == "":
        return None
    try:
        return int(v)
    except Exception:
        return None


def _won_value(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return 1 if v else 0
    if isinstance(v, int):
        if v in (0, 1):
            return v
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"1", "true", "won", "win", "yes"}:
            return 1
        if s in {"0", "false", "lost", "loss", "no"}:
            return 0
    return None


def _fmt_money(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"${v:.2f}"


def _fmt_pct(v: Optional[float]) -> str:
    if v is None:
        return "n/a"
    return f"{v:.1f}%"


def _ts_age(ts: Optional[str]) -> str:
    if not ts:
        return "n/a"
    try:
        s = ts.replace("Z", "+00:00")
        dt_obj = datetime.fromisoformat(s)
        if dt_obj.tzinfo is None:
            dt_obj = dt_obj.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        secs = int((now - dt_obj).total_seconds())
        if secs < 0:
            secs = 0
        if secs < 60:
            return f"{secs}s"
        mins = secs // 60
        if mins < 60:
            return f"{mins}m"
        hrs = mins // 60
        mins_rem = mins % 60
        return f"{hrs}h{mins_rem}m"
    except Exception:
        return "?"


def _truncate(s: Any, n: int) -> str:
    txt = "" if s is None else str(s)
    if len(txt) <= n:
        return txt
    return txt[: max(0, n - 1)] + "…"


def _extract_reason_value(reasons: str, key: str) -> Optional[str]:
    # Matches "key:value" until semicolon
    m = re.search(rf"(?:^|;){re.escape(key)}:([^;]+)", reasons or "")
    if not m:
        return None
    return m.group(1)


@dataclass
class ModeStats:
    count: int = 0
    trades: int = 0
    skips: int = 0
    errors: int = 0
    resolved_trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl_total: float = 0.0
    pnl_count: int = 0
    avg_stake_sum: float = 0.0
    avg_conv_sum: float = 0.0

    @property
    def avg_stake(self) -> float:
        return round(self.avg_stake_sum / self.trades, 2) if self.trades else 0.0

    @property
    def avg_conv(self) -> float:
        return round(self.avg_conv_sum / self.count, 2) if self.count else 0.0

    @property
    def win_rate(self) -> Optional[float]:
        d = self.wins + self.losses
        if d == 0:
            return None
        return round((self.wins / d) * 100.0, 1)

    @property
    def expectancy(self) -> Optional[float]:
        if self.pnl_count == 0:
            return None
        return round(self.pnl_total / self.pnl_count, 4)

    @property
    def pnl(self) -> Optional[float]:
        if self.pnl_count == 0:
            return None
        return round(self.pnl_total, 2)


class DashboardDB:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _table_columns(self, conn: sqlite3.Connection, table_name: str) -> set[str]:
        cur = conn.cursor()
        cur.execute(f"PRAGMA table_info({table_name})")
        return {str(r[1]) for r in cur.fetchall()}

    def get_snapshot(self, recent_limit: int = 120, latest_limit: int = 12) -> Dict[str, Any]:
        conn = self._connect()
        try:
            cols = self._table_columns(conn, "decisions")
            cur = conn.cursor()

            select_cols = [
                "id", "ts", "ticker", "direction",
                "base_conviction", "social_bonus", "final_conviction",
                "stake_dollars", "action", "reasons", "dry_run",
            ]
            optional = [
                "realized_pnl", "won", "resolved_ts", "market_category",
                "news_score", "news_confidence", "news_regime",
                "news_effective_score", "spread_cents",
            ]
            for c in optional:
                if c in cols:
                    select_cols.append(c)

            cur.execute(
                f"SELECT {', '.join(select_cols)} FROM decisions ORDER BY id DESC LIMIT ?",
                (int(recent_limit),),
            )
            rows = [dict(r) for r in cur.fetchall()]

            latest_rows = rows[: int(latest_limit)]

            summary = self._build_summary(rows)
            quick = self._build_quick_signals(latest_rows, rows)
            buckets = self._build_buckets(rows)

            return {
                "ok": True,
                "rows": rows,
                "latest_rows": latest_rows,
                "summary": summary,
                "quick": quick,
                "buckets": buckets,
                "columns": sorted(cols),
                "db_mtime": os.path.getmtime(self.db_path) if os.path.exists(self.db_path) else None,
            }
        finally:
            conn.close()

    def _build_summary(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        overall = ModeStats()
        sim = ModeStats()
        live = ModeStats()

        for r in rows:
            target = sim if bool(r.get("dry_run", 1)) else live
            for bucket in (overall, target):
                bucket.count += 1
                bucket.avg_conv_sum += (_safe_float(r.get("final_conviction")) or 0.0)

                action = str(r.get("action", "") or "")
                if action.startswith("TRADE"):
                    bucket.trades += 1
                    bucket.avg_stake_sum += (_safe_float(r.get("stake_dollars")) or 0.0)
                    if r.get("resolved_ts") is not None:
                        bucket.resolved_trades += 1

                    w = _won_value(r.get("won"))
                    if w == 1:
                        bucket.wins += 1
                    elif w == 0:
                        bucket.losses += 1

                    pnl = _safe_float(r.get("realized_pnl"))
                    if pnl is not None:
                        bucket.pnl_total += pnl
                        bucket.pnl_count += 1
                elif action == "SKIP":
                    bucket.skips += 1
                elif action == "ERROR":
                    bucket.errors += 1

        return {
            "overall": overall,
            "sim": sim,
            "live": live,
        }

    def _build_quick_signals(self, latest_rows: List[Dict[str, Any]], rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        latest10 = latest_rows[:10]
        skip_count_10 = sum(1 for r in latest10 if str(r.get("action", "")).upper() == "SKIP")
        trade_count_50 = sum(1 for r in rows[:50] if str(r.get("action", "")).startswith("TRADE"))
        error_count_50 = sum(1 for r in rows[:50] if str(r.get("action", "")).upper() == "ERROR")

        # candidate weirdness heuristics from stored row info
        low_spread_extreme_like = 0
        news_seen = 0
        news_eff_seen = 0

        for r in rows[:50]:
            conv = _safe_float(r.get("final_conviction"))
            spread = _safe_int(r.get("spread_cents"))
            if spread is not None and spread <= 2 and conv is not None and conv >= 60:
                low_spread_extreme_like += 1

            if r.get("news_regime") is not None or r.get("news_score") is not None:
                news_seen += 1
            if r.get("news_effective_score") is not None:
                news_eff_seen += 1

        newest_ts = latest_rows[0].get("ts") if latest_rows else None

        return {
            "skip_count_latest10": skip_count_10,
            "trade_count_latest50": trade_count_50,
            "error_count_latest50": error_count_50,
            "newest_row_ts": newest_ts,
            "newest_row_age": _ts_age(newest_ts),
            "news_fields_present_rows50": news_seen,
            "news_effective_rows50": news_eff_seen,
            "low_spread_highconv_rows50": low_spread_extreme_like,
            "all_latest10_skips": (len(latest10) > 0 and skip_count_10 == len(latest10)),
        }

    def _build_buckets(self, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
        cats: Dict[str, int] = {}
        regimes: Dict[str, int] = {}
        actions: Dict[str, int] = {}

        for r in rows[:200]:
            actions[str(r.get("action") or "unknown")] = actions.get(str(r.get("action") or "unknown"), 0) + 1
            cat = str(r.get("market_category") or "unknown")
            cats[cat] = cats.get(cat, 0) + 1
            reg = str(r.get("news_regime") or "unknown")
            regimes[reg] = regimes.get(reg, 0) + 1

        def topn(d: Dict[str, int], n: int = 6) -> List[tuple[str, int]]:
            return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:n]

        return {
            "top_categories": topn(cats),
            "top_news_regimes": topn(regimes),
            "top_actions": topn(actions),
        }


def _render_header(title: str, db_path: str, refresh_sec: float) -> None:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{BOLD}{title}{RESET}  {DIM}{now}{RESET}")
    print(f"DB: {db_path} | refresh={refresh_sec:.1f}s")
    print("-" * 120)


def _render_quick(quick: Dict[str, Any]) -> None:
    flags = []
    if quick.get("all_latest10_skips"):
        flags.append(f"{YELLOW}Latest10 all SKIP{RESET}")
    if (quick.get("error_count_latest50") or 0) > 0:
        flags.append(f"{RED}Errors in latest50={quick['error_count_latest50']}{RESET}")
    if (quick.get("trade_count_latest50") or 0) == 0:
        flags.append(f"{YELLOW}No trades in latest50{RESET}")
    if (quick.get("newest_row_age") not in (None, "n/a")):
        # if age string ends with h+ maybe stale
        pass

    print(f"{BOLD}Quick Signals{RESET}")
    print(
        f" newest_row_age={quick.get('newest_row_age')} | "
        f"latest10_skips={quick.get('skip_count_latest10')} | "
        f"latest50_trades={quick.get('trade_count_latest50')} | "
        f"latest50_errors={quick.get('error_count_latest50')} | "
        f"news_rows50={quick.get('news_fields_present_rows50')} | "
        f"news_eff_rows50={quick.get('news_effective_rows50')}"
    )
    if flags:
        print(" flags:", " | ".join(flags))
    print()


def _render_mode_stats(summary: Dict[str, ModeStats]) -> None:
    print(f"{BOLD}Mode Stats (recent window){RESET}")
    header = (
        f"{'Mode':<8} {'Rows':>6} {'Trades':>7} {'Resolved':>9} {'W':>4} {'L':>4} "
        f"{'WR%':>7} {'PnL':>10} {'Exp':>10} {'AvgStake':>10} {'AvgConv':>9}"
    )
    print(header)
    print("-" * len(header))

    for key in ("overall", "sim", "live"):
        st = summary[key]
        wr = _fmt_pct(st.win_rate)
        pnl = _fmt_money(st.pnl)
        exp = f"{st.expectancy:.4f}" if st.expectancy is not None else "n/a"
        print(
            f"{key:<8} {st.count:>6} {st.trades:>7} {st.resolved_trades:>9} {st.wins:>4} {st.losses:>4} "
            f"{wr:>7} {pnl:>10} {exp:>10} {st.avg_stake:>10.2f} {st.avg_conv:>9.2f}"
        )
    print()


def _render_buckets(buckets: Dict[str, Any]) -> None:
    print(f"{BOLD}Top Buckets (recent window){RESET}")
    print(" Actions:", " | ".join([f"{k}={v}" for k, v in buckets.get("top_actions", [])]) or "n/a")
    print(" Cats:   ", " | ".join([f"{k}={v}" for k, v in buckets.get("top_categories", [])]) or "n/a")
    print(" News:   ", " | ".join([f"{k}={v}" for k, v in buckets.get("top_news_regimes", [])]) or "n/a")
    print()


def _row_color(action: str) -> str:
    a = (action or "").upper()
    if a.startswith("TRADE"):
        return GREEN
    if a == "ERROR":
        return RED
    if a == "SKIP":
        return YELLOW
    return RESET


def _render_latest_rows(rows: List[Dict[str, Any]]) -> None:
    print(f"{BOLD}Latest Decisions{RESET}")
    header = (
        f"{'ID':>6} {'Age':>6} {'Mode':<4} {'Action':<10} {'Dir':<5} "
        f"{'Conv':>6} {'Stake':>8} {'NewsRaw':>8} {'NewsEff':>8} {'NewsReg':<14} "
        f"{'Spread':>6} {'Ticker':<34} {'Reason Snips':<36}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        rid = r.get("id")
        age = _ts_age(r.get("ts"))
        mode = "SIM" if bool(r.get("dry_run", 1)) else "LIVE"
        action = str(r.get("action", "") or "")
        direction = str(r.get("direction", "") or "")
        conv = _safe_float(r.get("final_conviction"))
        stake = _safe_float(r.get("stake_dollars"))
        news_raw = _safe_float(r.get("news_score"))
        news_eff = _safe_float(r.get("news_effective_score"))
        news_reg = str(r.get("news_regime") or "n/a")
        spread = _safe_int(r.get("spread_cents"))
        ticker = _truncate(r.get("ticker"), 34)
        reasons = str(r.get("reasons", "") or "")

        # snip only most useful reason fields
        snips = []
        for k in (
            "risk",
            "learning_mode",
            "news",
            "news_score_eff",
            "news_conf",
            "risk_gate_wr",
            "entry_price_cents",
        ):
            v = _extract_reason_value(reasons, k)
            if v is not None:
                snips.append(f"{k}={v}")
        reason_snips = _truncate(" | ".join(snips), 36)

        color = _row_color(action)
        print(
            f"{color}"
            f"{str(rid):>6} {age:>6} {mode:<4} {_truncate(action,10):<10} {_truncate(direction,5):<5} "
            f"{(f'{conv:.1f}' if conv is not None else 'n/a'):>6} "
            f"{(f'{stake:.2f}' if stake is not None else 'n/a'):>8} "
            f"{(f'{news_raw:.2f}' if news_raw is not None else 'n/a'):>8} "
            f"{(f'{news_eff:.2f}' if news_eff is not None else 'n/a'):>8} "
            f"{_truncate(news_reg,14):<14} "
            f"{(str(spread) if spread is not None else 'n/a'):>6} "
            f"{ticker:<34} {reason_snips:<36}"
            f"{RESET}"
        )
    print()


def run_dashboard(
    db_path: str,
    refresh_sec: float = 2.0,
    recent_limit: int = 120,
    latest_limit: int = 12,
    once: bool = False,
) -> None:
    db = DashboardDB(db_path)

    while True:
        try:
            snap = db.get_snapshot(recent_limit=recent_limit, latest_limit=latest_limit)

            sys.stdout.write(CLEAR)
            _render_header("Kalshi Bot Terminal Dashboard", db_path=db_path, refresh_sec=refresh_sec)
            _render_quick(snap["quick"])
            _render_mode_stats(snap["summary"])
            _render_buckets(snap["buckets"])
            _render_latest_rows(snap["latest_rows"])

            print(f"{DIM}Press Ctrl+C to exit.{RESET}")
            sys.stdout.flush()

        except KeyboardInterrupt:
            print("\nExiting dashboard.")
            return
        except sqlite3.OperationalError as e:
            sys.stdout.write(CLEAR)
            print(f"{RED}SQLite error:{RESET} {e}")
            print(f"DB path: {db_path}")
            print("Waiting and retrying...")
            sys.stdout.flush()
        except Exception as e:
            sys.stdout.write(CLEAR)
            print(f"{RED}Dashboard error:{RESET} {e}")
            print("Waiting and retrying...")
            sys.stdout.flush()

        if once:
            return

        time.sleep(max(0.25, float(refresh_sec)))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live terminal dashboard for Kalshi bot SQLite decisions.")
    p.add_argument("--db", dest="db_path", required=True, help="Path to SQLite DB (e.g. /data/kalshi_bot.db)")
    p.add_argument("--refresh", type=float, default=2.0, help="Refresh interval seconds (default: 2.0)")
    p.add_argument("--recent", type=int, default=120, help="Recent rows window for summary stats (default: 120)")
    p.add_argument("--latest", type=int, default=12, help="Latest rows shown in table (default: 12)")
    p.add_argument("--once", action="store_true", help="Render once and exit")
    return p


def main() -> None:
    args = build_arg_parser().parse_args()
    run_dashboard(
        db_path=args.db_path,
        refresh_sec=args.refresh,
        recent_limit=args.recent,
        latest_limit=args.latest,
        once=bool(args.once),
    )


if __name__ == "__main__":
    main()
