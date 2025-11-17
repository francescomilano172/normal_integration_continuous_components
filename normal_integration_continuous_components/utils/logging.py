import time
import torch
import os


class TimingLogger:
    _start_time = None
    _log_path = "timing.log"
    _initialized = False

    @classmethod
    def init(cls, log_path="timing.log", overwrite=True, also_log_to_screen=False):
        cls._start_time = time.perf_counter()
        cls._log_path = log_path
        cls._also_log_to_screen = also_log_to_screen
        if overwrite and os.path.exists(log_path):
            open(log_path, "w").close()
        cls._initialized = True
        cls.log("Timing started")

    @classmethod
    def log(cls, message, sync_cuda=True):
        if not cls._initialized:
            cls.init()

        if sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()

        now = time.perf_counter()
        elapsed = now - cls._start_time
        log_entry = f"[{elapsed:.3f}s] {message}"

        if cls._also_log_to_screen:
            # Optional: also print to console.
            print(log_entry)
        with open(cls._log_path, "a") as f:
            f.write(log_entry + "\n")


class MADELogger:
    def __init__(self, made_fixed_message, log_path, also_log_to_screen=False):
        self._made_fixed_message = made_fixed_message
        self._log_path = log_path
        self._also_log_to_screen = also_log_to_screen

    def log(self, iter_idx, effective_iter_idx, made, num_connected_components):
        log_entry = (
            f"[Iteration {iter_idx} (effective iteration {effective_iter_idx})] "
            f"{self._made_fixed_message} = {made} (#components = "
            f"{num_connected_components})"
        )

        if self._also_log_to_screen:
            # Optional: also print to console.
            print(log_entry)
        with open(self._log_path, "a") as f:
            f.write(log_entry + "\n")
