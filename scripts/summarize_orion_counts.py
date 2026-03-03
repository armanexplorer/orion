#!/usr/bin/env python3

import argparse
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional


KCOUNT_RE = re.compile(
    r"KCOUNT\s+client=(?P<client>\d+)\s+"
    r"iter=(?P<iter>\d+)\s+"
    r"seen=(?P<seen>\d+)\s+"
    r"configured_num_kernels=(?P<configured>\d+)\s+"
    r"captured_records=(?P<captured>\d+)\s+"
    r"opinfo_rows=(?P<opinfo_rows>\d+)\s+"
    r"queue_size=(?P<queue_size>\d+)"
)

AUTO_RE = re.compile(
    r"AUTO_NUM_KERNELS\s+client=(?P<client>\d+)\s+"
    r"infer_iter_end\s+seen=(?P<seen>\d+)\s+"
    r"idle_s=(?P<idle_s>[0-9.]+)"
)


@dataclass
class ClientMeta:
    arch: Optional[str] = None
    model_name: Optional[str] = None
    config_num_kernels: Optional[int] = None


@dataclass
class ClientSummary:
    k_seen: List[int]
    k_captured: List[int]
    k_configured: List[int]
    auto_seen: List[int]


def _read_lines(path: Optional[str]) -> Iterable[str]:
    if path is None or path == "-":
        yield from sys.stdin
        return
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        yield from f


def _fmt_meta(meta: ClientMeta) -> str:
    parts = []
    if meta.arch:
        parts.append(meta.arch)
    if meta.model_name and meta.model_name != meta.arch:
        parts.append(meta.model_name)
    if not parts:
        return "(unknown)"
    return "/".join(parts)


def _percentile(values: List[int], p: float) -> int:
    """Nearest-rank percentile, p in [0,100]."""
    if not values:
        raise ValueError("empty")
    if p <= 0:
        return min(values)
    if p >= 100:
        return max(values)
    xs = sorted(values)
    k = int(round((p / 100.0) * (len(xs) - 1)))
    return xs[k]


def _summ_stats(values: List[int]) -> str:
    if not values:
        return "n=0"
    med = int(statistics.median(values))
    return (
        f"n={len(values)} min={min(values)} p10={_percentile(values, 10)} "
        f"med={med} p90={_percentile(values, 90)} max={max(values)}"
    )


def _mode(values: List[int], max_items: int = 3) -> str:
    if not values:
        return ""
    c = Counter(values)
    items = c.most_common(max_items)
    return ", ".join([f"{v}({n})" for v, n in items])


def _load_config_meta(config_file: Optional[str]) -> Dict[int, ClientMeta]:
    if not config_file:
        return {}
    with open(config_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if not isinstance(cfg, list):
        raise ValueError("Config JSON must be a list of client dicts")

    meta: Dict[int, ClientMeta] = {}
    for i, entry in enumerate(cfg):
        if not isinstance(entry, dict):
            continue
        args = entry.get("args")
        model_name = None
        if isinstance(args, dict):
            model_name = args.get("model_name")
        meta[i] = ClientMeta(
            arch=entry.get("arch"),
            model_name=model_name,
            config_num_kernels=entry.get("num_kernels"),
        )
    return meta


def parse_log(lines: Iterable[str]) -> Dict[int, ClientSummary]:
    k_seen: Dict[int, List[int]] = defaultdict(list)
    k_captured: Dict[int, List[int]] = defaultdict(list)
    k_configured: Dict[int, List[int]] = defaultdict(list)
    auto_seen: Dict[int, List[int]] = defaultdict(list)

    for line in lines:
        m = KCOUNT_RE.search(line)
        if m:
            client = int(m.group("client"))
            k_seen[client].append(int(m.group("seen")))
            k_captured[client].append(int(m.group("captured")))
            k_configured[client].append(int(m.group("configured")))
            continue

        m = AUTO_RE.search(line)
        if m:
            client = int(m.group("client"))
            auto_seen[client].append(int(m.group("seen")))
            continue

    out: Dict[int, ClientSummary] = {}
    for client in sorted(set(k_seen) | set(auto_seen) | set(k_captured) | set(k_configured)):
        out[client] = ClientSummary(
            k_seen=k_seen.get(client, []),
            k_captured=k_captured.get(client, []),
            k_configured=k_configured.get(client, []),
            auto_seen=auto_seen.get(client, []),
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Summarize Orion scheduler stdout logs by extracting KCOUNT and AUTO_NUM_KERNELS lines. "
            "Reads from --logfile or stdin."
        )
    )
    ap.add_argument(
        "--logfile",
        type=str,
        default="-",
        help="Path to a captured stdout/stderr log (use '-' for stdin).",
    )
    ap.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Optional Orion config JSON to map client_id -> arch/model_name and configured num_kernels.",
    )
    args = ap.parse_args()

    try:
        meta = _load_config_meta(args.config_file)
    except Exception as e:
        print(f"ERROR: failed to read --config_file: {e}", file=sys.stderr)
        return 2

    summaries = parse_log(_read_lines(args.logfile))

    if not summaries:
        print("No KCOUNT/AUTO_NUM_KERNELS lines found.")
        return 1

    for client_id in sorted(summaries.keys()):
        s = summaries[client_id]
        m = meta.get(client_id, ClientMeta())

        title = f"client {client_id}: {_fmt_meta(m)}"
        if m.config_num_kernels is not None:
            title += f" (config_num_kernels={m.config_num_kernels})"
        print(title)

        if s.k_seen:
            print(f"  KCOUNT.seen:      {_summ_stats(s.k_seen)} | mode: {_mode(s.k_seen)}")
            print(f"  KCOUNT.captured:  {_summ_stats(s.k_captured)} | mode: {_mode(s.k_captured)}")
            print(f"  KCOUNT.configured {_summ_stats(s.k_configured)} | mode: {_mode(s.k_configured)}")

            mism = sum(1 for a, b in zip(s.k_seen, s.k_captured) if a != b)
            n = min(len(s.k_seen), len(s.k_captured))
            if n:
                print(f"  seen!=captured:   {mism}/{n} ({(mism/n)*100.0:.1f}%)")

            if m.config_num_kernels is not None:
                med_seen = int(statistics.median(s.k_seen))
                if med_seen != m.config_num_kernels:
                    print(f"  NOTE: median(seen)={med_seen} differs from config_num_kernels={m.config_num_kernels}")
        else:
            print("  KCOUNT:           n=0")

        if s.auto_seen:
            print(f"  AUTO.seen:        {_summ_stats(s.auto_seen)} | mode: {_mode(s.auto_seen)}")
        else:
            print("  AUTO.seen:        n=0")

        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
