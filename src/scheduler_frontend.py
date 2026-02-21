import ctypes
from ctypes import *
import torch
import numpy as np
import os
import time


def _env_flag_enabled(name: str, default: str = "0") -> bool:
    v = os.environ.get(name, default)
    return v not in ("", "0", "false", "False", "no", "No")


def _sched_debug_enabled() -> bool:
    return _env_flag_enabled("ORION_SCHED_DEBUG", "0")


def _dprint(msg: str) -> None:
    if _sched_debug_enabled():
        print(msg, flush=True)


class PyScheduler:

    def __init__(self, sched_lib, num_clients):

        torch.cuda.set_device(0)
        # Set ctypes signatures to avoid argument/return truncation.
        # Without these, ctypes assumes int return types and int arguments,
        # which can corrupt pointers/booleans on 64-bit systems.
        sched_lib.sched_init.restype = c_void_p
        sched_lib.sched_init.argtypes = []

        # Core entrypoints used by the Python frontend.
        sched_lib.setup.restype = None
        sched_lib.setup.argtypes = [
            c_void_p,
            c_int,
            POINTER(c_int),
            POINTER(c_char_p),
            POINTER(c_char_p),
            POINTER(c_int),
            POINTER(c_int),
            POINTER(c_bool),
            c_bool,
            c_bool,
        ]

        sched_lib.setup_change.restype = None
        sched_lib.setup_change.argtypes = [c_void_p, c_int, c_char_p, c_int]

        # void* schedule(Scheduler*, int, bool, int, bool, int, bool, bool, int, int, int)
        sched_lib.schedule.restype = c_void_p
        sched_lib.schedule.argtypes = [
            c_void_p,
            c_int,
            c_bool,
            c_int,
            c_bool,
            c_int,
            c_bool,
            c_bool,
            c_int,
            c_int,
            c_int,
        ]

        # Optional reset hooks.
        if hasattr(sched_lib, "reset"):
            sched_lib.reset.restype = c_void_p
            sched_lib.reset.argtypes = [c_void_p, c_int]
        if hasattr(sched_lib, "reset_iters"):
            sched_lib.reset_iters.restype = c_void_p
            sched_lib.reset_iters.argtypes = [c_void_p, c_int]

        if hasattr(sched_lib, "schedule_one"):
            sched_lib.schedule_one.restype = None
            sched_lib.schedule_one.argtypes = [c_void_p, c_int]

        self._scheduler = sched_lib.sched_init()
        self._sched_lib = sched_lib
        self._num_clients = num_clients

    def run_scheduler(
        self,
        barriers,
        tids,
        model_names,
        kernel_files,
        additional_kernel_files,
        num_kernels,
        additional_num_kernels,
        num_iters,
        profile,
        run_eval,
        reef,
        sequential,
        reef_depth,
        hp_limit,
        update_start,
        train
    ):

        _dprint(f"REEF IS {reef}, SEQUENTIAL IS {sequential}")

        model_names_ctypes = [x.encode('utf-8') for x in model_names]
        lib_names = [x.encode('utf-8') for x in kernel_files]

        # convert
        IntAr = c_int * self._num_clients
        tids_ar = IntAr(*tids)
        num_kernels_ar = IntAr(*num_kernels)
        num_iters_ar = IntAr(*num_iters)

        CharAr = c_char_p * self._num_clients
        model_names_ctypes_ar = CharAr(*model_names_ctypes)
        lib_names_ar = CharAr(*lib_names)

        BoolAr = c_bool * self._num_clients
        train_ar = BoolAr(*train)

        _dprint(str(train))
        _dprint(f"{model_names} {lib_names} {tids}")

        self._sched_lib.setup(self._scheduler, self._num_clients, tids_ar, model_names_ctypes_ar, lib_names_ar, num_kernels_ar, num_iters_ar, train_ar, reef, sequential)

        num_clients = len(tids)
        _dprint(f"Num clients is {num_clients}")

        _dprint(f"before starting, profile is {profile}")
        timings=[]

        if run_eval:
            if profile:
                _dprint("SCHED DEBUG: Starting first barrier wait (warmup setup)")
                barriers[0].wait()
                _dprint("SCHED DEBUG: Passed first barrier, calling schedule for warmup setup")

                # Some workloads (notably YOLOv5 via torch.hub) can spend a long time
                # after the first barrier loading weights / initializing CUDA graphs.
                # If we call into schedule() before the first client request is
                # submitted, scheduler_eval may immediately exit its loop, and the
                # client will later deadlock inside backend_lib.block().
                #
                # Mitigation: optionally add a short, configurable delay for YOLO workloads.
                try:
                    has_yolo = any(("yolov5" in (name or "").lower()) for name in model_names)
                except Exception:
                    has_yolo = False

                # YOLOv5 can have slight per-iteration variability in the number of
                # intercepted records (e.g., allocator/caching effects). Enable the
                # scheduler's per-iteration auto boundary inference unless the user
                # explicitly configured it.
                if has_yolo and os.environ.get("ORION_AUTO_NUM_KERNELS_SEC") is None:
                    os.environ["ORION_AUTO_NUM_KERNELS_SEC"] = "0.05"
                    _dprint("SCHED DEBUG: YOLO workload: enabling ORION_AUTO_NUM_KERNELS_SEC=0.05")

                if has_yolo:
                    delay_s = float(os.environ.get("ORION_SCHED_PREWARM_DELAY_SEC", "0"))
                    if delay_s > 0:
                        _dprint(f"SCHED DEBUG: Detected YOLO workload, sleeping {delay_s}s before warmup-setup schedule")
                        time.sleep(delay_s)

                # run once to warm-up and setup
                self._sched_lib.schedule(self._scheduler, num_clients, True, 0, True, 1, reef, sequential, reef_depth, hp_limit, update_start)
                torch.cuda.synchronize()
                _dprint("SCHED DEBUG: First schedule done, synchronize complete")

                # YOLOv5 can issue CUDA work during model loading that consumes
                # one logical iteration in the scheduler (num_client_cur_iters).
                # Reset iteration counters after the warmup-setup pass so the
                # following warmup(10) aligns with the client loop.
                if has_yolo:
                    try:
                        self._sched_lib.reset_iters(self._scheduler, num_clients)
                        _dprint("SCHED DEBUG: YOLO workload: reset scheduler iteration counters after warmup-setup")
                    except Exception as e:
                        print(f"SCHED WARN: reset_iters failed: {e}")

                for j in range(num_clients):
                    if (additional_kernel_files[j] is not None):
                        new_kernel_file = additional_kernel_files[j].encode('utf-8')
                        self._sched_lib.setup_change(self._scheduler, j, new_kernel_file, additional_num_kernels[j])

                if has_yolo:
                    _dprint("SCHED DEBUG: YOLO workload: skipping barrier after warmup setup")
                else:
                    _dprint("SCHED DEBUG: wait here (waiting for barrier after warmup setup)")
                    barriers[0].wait() #FIXME
                    _dprint("SCHED DEBUG: done! (passed barrier after warmup setup)")

                # warmup
                _dprint("SCHED DEBUG: Starting warmup with 10 iterations")
                self._sched_lib.schedule(self._scheduler, num_clients, True, 0, True, 10, reef, sequential, reef_depth, hp_limit, update_start)
                torch.cuda.synchronize()
                _dprint("SCHED DEBUG: Warmup schedule done, waiting at barrier")
                barriers[0].wait()
                _dprint("Warmup done, starting eval")

                start = time.time()
                _dprint("call schedule")
                self._sched_lib.schedule(self._scheduler, num_clients, True, 0, False, 0, reef, sequential, reef_depth, hp_limit, update_start)
                _dprint("SCHED DEBUG: Main schedule call returned, waiting at barrier")
                barriers[0].wait()
                torch.cuda.synchronize()
                _dprint(f"Total time is {time.time()-start}")

        else:
            for i in range(num_iters[0]):

                print(f"Start {i} iteration")
                if profile:
                    barriers[0].wait()
                    # needed for backward
                    if (i==1):
                        for j in range(num_clients):
                            if (additional_kernel_files[j] is not None):
                                new_kernel_file = additional_kernel_files[j].encode('utf-8')
                                self._sched_lib.setup_change(self._scheduler, j, new_kernel_file, additional_num_kernels[j])
                        barriers[0].wait() #FIXME

                    start = time.time()
                    print("call schedule")
                    self._sched_lib.schedule(self._scheduler, num_clients, True, i)
                    torch.cuda.synchronize()

                # or this
                else:
                    start = time.time()
                    for j in range(num_clients):
                        barriers[j].wait()
                        self._sched_lib.schedule_one(self._scheduler, j)
                        torch.cuda.synchronize()

                total_time = time.time()-start
                print(f"Iteration {i} took {total_time} sec")
                timings.append(total_time)
            timings = timings[3:]
            print(f"Avg is {np.median(np.asarray(timings))}, Min is {min(timings)} sec")
