"""Concurrency-safe append to the shared best_results.csv summary.

Every optimizer appends one row per dataset to the same file. Run one at a time
that is harmless, but under a process pool the naive

    write_header = not os.path.exists(path)
    row.to_csv(path, mode='a', header=write_header, index=False)

races two ways: several processes can each observe "no file" and each write a
header, and concurrent appends can interleave mid-row. This module funnels the
check-and-append through an exclusive lock so both stay correct.
"""

import errno
import os
import time

LOCK_SUFFIX = ".lock"
LOCK_POLL_SECONDS = 0.05
LOCK_STALE_SECONDS = 120.0
LOCK_WAIT_SECONDS = 300.0


class _FileLock:
    """Exclusive lock via atomic O_CREAT|O_EXCL, with stale-lock recovery.

    Portable across platforms and process-crash tolerant: a lock left behind by
    a killed process is reclaimed once it is older than LOCK_STALE_SECONDS.
    """

    def __init__(self, target_path):
        self.lock_path = target_path + LOCK_SUFFIX
        self._fd = None

    def __enter__(self):
        deadline = time.monotonic() + LOCK_WAIT_SECONDS

        while True:
            try:
                self._fd = os.open(
                    self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR
                )
                os.write(self._fd, str(os.getpid()).encode("ascii"))
                return self
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise

            self._break_if_stale()

            if time.monotonic() > deadline:
                raise TimeoutError(
                    f"Timed out after {LOCK_WAIT_SECONDS:.0f}s waiting for "
                    f"{self.lock_path}. If no optimizer is running, delete it."
                )
            time.sleep(LOCK_POLL_SECONDS)

    def _break_if_stale(self):
        try:
            age = time.time() - os.path.getmtime(self.lock_path)
        except OSError:
            return  # Vanished between the failed create and here -- retry.

        if age > LOCK_STALE_SECONDS:
            print(
                f"[results_io] Reclaiming stale lock {self.lock_path} "
                f"(age {age:.0f}s)"
            )
            try:
                os.unlink(self.lock_path)
            except OSError:
                pass  # Another process won the race to reclaim it.

    def __exit__(self, exc_type, exc_value, traceback):
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None
        try:
            os.unlink(self.lock_path)
        except OSError:
            pass
        return False


def append_best_row(summary_path, row):
    """Append one row DataFrame to summary_path, writing a header only once.

    Safe to call concurrently from multiple processes.
    """
    os.makedirs(os.path.dirname(os.path.abspath(summary_path)), exist_ok=True)

    with _FileLock(summary_path):
        # Inside the lock, so this observation cannot go stale before the write.
        needs_header = (
            not os.path.exists(summary_path)
            or os.path.getsize(summary_path) == 0
        )
        row.to_csv(summary_path, mode="a", header=needs_header, index=False)
