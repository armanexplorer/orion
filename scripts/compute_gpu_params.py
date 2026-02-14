#!/usr/bin/env python3
"""Compute GPU params used by Orion profiling postprocessing.

Prints:
- Per-SM resource limits for get_num_blocks.py
- Roofline ridge-point arithmetic intensity (ai_threshold) estimate for roofline_analysis.py

This script prints two ridge-point estimates because different tools use different
"peak" definitions:

1) CUDA device attributes (via libcudart `cudaDeviceGetAttribute`)
    - Matches what Nsight Compute typically reports as device peak clocks/bandwidth
      for Roofline/SpeedOfLight (e.g., `device__attribute_clock_rate`).

2) NVML max clocks (via pynvml / nvidia-smi "Max Clocks")
    - Reflects the maximum supported clocks the GPU may reach.

For reproducibility with Nsight Compute Roofline, prefer the CUDA-device-attribute
ridge point.
"""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import os
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


def _load_cudart() -> ctypes.CDLL:
    """Best-effort load of libcudart for cudaDeviceGetAttribute.

    We avoid third-party Python deps (cupy/numba/pycuda) and instead call the CUDA
    runtime API directly.
    """

    lib = ctypes.util.find_library("cudart")
    if lib:
        return ctypes.CDLL(lib)

    candidates = [
        "libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so",
        "/usr/local/cuda/lib64/libcudart.so.12",
        "/usr/local/cuda/lib64/libcudart.so.12.0",
        "/usr/local/cuda-12/lib64/libcudart.so",
        "/usr/local/cuda-12.6/lib64/libcudart.so",
    ]
    for path in candidates:
        try:
            if os.path.isabs(path) and not os.path.exists(path):
                continue
            return ctypes.CDLL(path)
        except OSError:
            continue

    raise OSError("Failed to locate libcudart.so")


def _cuda_device_get_attribute(cudart: ctypes.CDLL, *, attr: int, device: int) -> int:
    """Call cudaDeviceGetAttribute(int* value, cudaDeviceAttr attr, int device)."""

    fn = cudart.cudaDeviceGetAttribute
    fn.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    fn.restype = ctypes.c_int

    out = ctypes.c_int()
    rc = int(fn(ctypes.byref(out), int(attr), int(device)))
    if rc != 0:
        raise RuntimeError(f"cudaDeviceGetAttribute(attr={attr}, device={device}) failed rc={rc}")
    return int(out.value)


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

    fp32_cores_per_sm = _turing_fp32_cores_per_sm(*cc)
    if fp32_cores_per_sm is None:
        if args.assume_turing_fp32:
            if cc != (7, 5):
                print(
                    "ERROR: --assume_turing_fp32 is only valid for sm75 (Turing, cc 7.5).",
                    file=sys.stderr,
                )
                return 2
            fp32_cores_per_sm = 64
        else:
            print(
                "\nNOTE: This script only knows FP32 cores/SM for sm75 (Turing).\n"
                "      Extend _turing_fp32_cores_per_sm() for your GPU if you want an ai_threshold estimate.\n"
                "      (We will still print memory bandwidth and clocks.)"
            )

    print("\n=== roofline ai_threshold estimate (ridge point) ===")

    # (1) CUDA device attributes: matches Nsight Compute exports (e.g., device__attribute_clock_rate).
    try:
        # enum cudaDeviceAttr values from /usr/local/cuda/include/driver_types.h
        CUDA_DEV_ATTR_CLOCK_RATE = 13  # kHz
        CUDA_DEV_ATTR_MEMORY_CLOCK_RATE = 36  # kHz
        CUDA_DEV_ATTR_GLOBAL_MEMORY_BUS_WIDTH = 37  # bits
        CUDA_DEV_ATTR_MULTI_PROCESSOR_COUNT = 16

        cudart = _load_cudart()
        cuda_sm_khz = _cuda_device_get_attribute(cudart, attr=CUDA_DEV_ATTR_CLOCK_RATE, device=args.device)
        cuda_mem_khz = _cuda_device_get_attribute(
            cudart, attr=CUDA_DEV_ATTR_MEMORY_CLOCK_RATE, device=args.device
        )
        cuda_bus_bits = _cuda_device_get_attribute(
            cudart, attr=CUDA_DEV_ATTR_GLOBAL_MEMORY_BUS_WIDTH, device=args.device
        )
        cuda_sms = _cuda_device_get_attribute(
            cudart, attr=CUDA_DEV_ATTR_MULTI_PROCESSOR_COUNT, device=args.device
        )

        print("[CUDA device attributes] (matches Nsight Compute Roofline peaks)")
        print(f"sm_clock_kHz: {cuda_sm_khz}")
        print(f"mem_clock_kHz: {cuda_mem_khz}")
        print(f"mem_bus_width_bits: {cuda_bus_bits}")
        print(f"SMs: {cuda_sms}")

        peak_dram_Bps_cuda = (cuda_mem_khz * 1e3) * 2.0 * (cuda_bus_bits / 8.0)
        print(f"peak_dram_GB_per_s: {peak_dram_Bps_cuda / 1e9:.3f}")

        if fp32_cores_per_sm is not None:
            peak_fp32_flops_cuda = (
                float(cuda_sms) * float(fp32_cores_per_sm) * 2.0 * (cuda_sm_khz * 1e3)
            )
            ai_cuda = peak_fp32_flops_cuda / peak_dram_Bps_cuda
            print(f"peak_fp32_TFLOP_per_s: {peak_fp32_flops_cuda / 1e12:.4f}")
            print(f"ai_threshold_flop_per_byte: {ai_cuda:.6f}")
            print(f"ai_threshold_suggested: {ai_cuda:.1f}")
            print(f"suggested flag: --ai_threshold {ai_cuda:.1f}")
        else:
            print("NOTE: skipping ai_threshold computation (unknown FP32 cores/SM for this GPU).")

    except Exception as e:
        print(
            "[CUDA device attributes] WARNING: failed to query libcudart device attributes.\n"
            f"  {type(e).__name__}: {e}"
        )

    # (2) NVML max clocks: maximum supported clocks as reported by nvidia-smi.
    try:
        import pynvml

        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(args.device)
        bus_bits = int(pynvml.nvmlDeviceGetMemoryBusWidth(h))
        mem_mhz = int(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_MEM))
        sm_mhz = int(pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM))
        pynvml.nvmlShutdown()

        print("\n[NVML max clocks] (nvidia-smi 'Max Clocks'; may differ from NCU)")
        print(f"mem_bus_width_bits: {bus_bits}")
        print(f"mem_max_clock_MHz: {mem_mhz}")
        print(f"sm_max_clock_MHz: {sm_mhz}")

        peak_dram_Bps = (mem_mhz * 1e6) * 2.0 * (bus_bits / 8.0)
        print(f"peak_dram_GB_per_s: {peak_dram_Bps / 1e9:.3f}")

        if fp32_cores_per_sm is not None:
            peak_fp32_flops = (
                float(props.multi_processor_count)
                * float(fp32_cores_per_sm)
                * 2.0
                * (sm_mhz * 1e6)
            )
            ai = peak_fp32_flops / peak_dram_Bps
            print(f"peak_fp32_TFLOP_per_s: {peak_fp32_flops / 1e12:.4f}")
            print(f"ai_threshold_flop_per_byte: {ai:.6f}")
            print(f"ai_threshold_suggested: {ai:.1f}")
            print(f"suggested flag: --ai_threshold {ai:.1f}")
        else:
            print("NOTE: skipping ai_threshold computation (unknown FP32 cores/SM for this GPU).")

    except Exception as e:
        print(
            "\n[NVML max clocks] WARNING: failed to compute ai_threshold via NVML (pynvml).\n"
            f"  {type(e).__name__}: {e}\n"
            "You can still set ai_threshold by reading the ridge point in Nsight Compute Roofline/SpeedOfLight."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
