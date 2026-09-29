"""Byte-bounded single-worker persistence of owned CPU arrays."""

from concurrent.futures import ThreadPoolExecutor
from contextvars import copy_context
from threading import Condition

from .profiling import measure, record_count, save_array


class ArrayWriter:
    def __init__(self, max_pending_bytes):
        self.limit = max_pending_bytes
        self.pending = 0
        self.peak_pending = 0
        self.condition = Condition()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="anima-save")
        self.error = None
        self.closed = False

    def _raise_error(self):
        if self.error is not None:
            raise RuntimeError(f"attention 파일 저장에 실패했습니다: {self.error}") from self.error

    def _write(self, context, path, array, metadata):
        size = array.nbytes
        try:
            context.run(save_array, path, array, **metadata)
        except BaseException as error:
            with self.condition:
                self.error = self.error or error
        finally:
            del array
            with self.condition:
                self.pending -= size
                self.condition.notify_all()

    def submit(self, path, array, **metadata):
        size = array.nbytes
        with measure("write_queue_wait"):
            with self.condition:
                if self.closed:
                    raise RuntimeError("종료된 저장 작업입니다")
                while self.pending and self.pending + size > self.limit and self.error is None:
                    self.condition.wait()
                self._raise_error()
                if size <= self.limit:
                    self.pending += size
                    self.peak_pending = max(self.peak_pending, self.pending)
                    self.executor.submit(self._write, copy_context(), path, array, metadata)
                    record_count("async_write")
                    return
        # An individual record larger than the queue budget is saved directly.
        with measure("write_oversize_sync"):
            record_count("oversize_write")
            save_array(path, array, **metadata)

    def finish(self):
        self.closed = True
        with measure("write_drain"):
            self.executor.shutdown(wait=True)
        self._raise_error()
