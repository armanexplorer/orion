#!/usr/bin/env python3
"""Compute GPU params used by Orion profiling postprocessing.

Prints:
- Per-SM resource limits for get_num_blocks.py
- Roofline ridge-point arithmetic intensity (ai_threshold) estimate for roofline_analysis.py

The ridge-point estimate uses NVML (via pynvml) to read max clocks + memory bus width.
"""

from __future__ import annotations

import argparse
import sys


def _fmt_bytes(n: float) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(n) < 1024.0:
            return f"{n:.3f} {unit}"
        n /= 1024.0
    return f"{n:.3f} PiB"


def _turing_fp32_cores_per_sm(cc_major: int, cc_minor: int) -> int | None:
    # This script only hardcodes what we need for RTX 6000 (Turing, sm75).
    # If you run on another GPU, we still print clocks/bw and will refuse to guess FP32 cores/SM.
    if (cc_major, cc_minor) == (7, 5):
        return 64
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--assume_turing_fp32",
        action="store_true",
        help="Assume 64 FP32 cores/SM (only valid for Turing sm75) when computing ai_threshold.",
    )
    args = ap.parse_args()

    try:
        import torch
    except Exception as e:
        print(f"ERROR: failed to import torch: {e}", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available (torch.cuda.is_available() is False)", file=sys.stderr)
        return 2

    props = torch.cuda.get_device_properties(args.device)
    cc = (int(props.major), int(props.minor))

    print("=== Device ===")
    print(f"name: {props.name}")
    print(f"cc: {cc[0]}.{cc[1]}")
    print(f"SMs: {props.multi_processor_count}")
    print(f"total_memory: {_fmt_bytes(float(props.total_memory))}")

    print("\n=== get_num_blocks.py inputs (per-SM maxima) ===")
    max_threads_sm = getattr(props, "max_threads_per_multi_processor", None)
    max_shmem_sm = getattr(props, "shared_memory_per_multiprocessor", None)
    max_regs_sm = getattr(props, "regs_per_multiprocessor", None)

    print(f"max_threads_sm: {max_threads_sm}")
    print(f"max_shmem_sm_bytes: {max_shmem_sm}")
    print(f"max_regs_sm: {max_regs_sm}")
    if None not in (max_threads_sm, max_shmem_sm, max_regs_sm):
        print("suggested flags:")
        print(
            "  --max_threads_sm "
            + str(int(max_threads_sm))
            + " --max_shmem_sm "
            + str(int(max_shmem_sm))
            + " --max_regs_sm "
            + str(int(max_regs_sm))
        )
    else:
        print(
            "WARNING: torch device properties missing one or more per-SM maxima; "
            "use Nsight Compute GUI or CUDA deviceQuery to obtain them."
        )

    print("\n=== roofline ai_threshold estimate (ridge point) ===")
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(args.device)
        bus_bits = int(pynvml.nvmlDeviceGetMemoryBusWidth(h))
        mem_mhz = int(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM))
        sm_mhz = int(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM))
        pynvml.nvmlShutdown()

        print(f"mem_bus_width_bits: {bus_bits}")
        print(f"mem_max_clock_MHz: {mem_mhz}")
        print(f"sm_max_clock_MHz: {sm_mhz}")

        # Peak DRAM bandwidth (bytes/s): DDR * bus_bytes * mem_clock
        peak_dram_Bps = (mem_mhz * 1e6) * 2.0 * (bus_bits / 8.0)
        print(f"peak_dram_GB_per_s: {peak_dram_Bps / 1e9:.3f}")

        fp32_cores_per_sm = _turing_fp32_cores_per_sm(*cc)
        if fp32_cores_per_sm is None and not args.assume_turing_fp32:
            print(
                "NOTE: This script only knows FP32 cores/SM for sm75 (Turing).\n"
                "      Re-run with --assume_turing_fp32 if you are on sm75, or extend the mapping for your GPU."
            )
            return 0

        if fp32_cores_per_sm is None:
            fp32_cores_per_sm = 64

        # Peak FP32 (FLOP/s): SMs * (fp32_cores/SM) * (2 FLOP/cycle for FMA) * SM_clock
        peak_fp32_flops = (
            float(props.multi_processor_count)
            * float(fp32_cores_per_sm)
            * 2.0
            * (sm_mhz * 1e6)
        )
        print(f"peak_fp32_TFLOP_per_s: {peak_fp32_flops / 1e12:.4f}")

        ai = peak_fp32_flops / peak_dram_Bps
        print(f"ai_threshold_flop_per_byte: {ai:.6f}")
        print(f"ai_threshold_suggested: {ai:.1f}")
        print(f"suggested flag: --ai_threshold {ai:.1f}")

    except Exception as e:
        print(
            "WARNING: failed to compute ai_threshold via NVML (pynvml).\n"
            f"  {type(e).__name__}: {e}\n"
            "You can still set ai_threshold by reading the ridge point in Nsight Compute Roofline/SpeedOfLight."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
