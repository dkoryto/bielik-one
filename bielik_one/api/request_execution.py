"""Optional request bounds shared by decision frontends and embedded middleware."""
import asyncio
import math
import threading

import httpx
from fastapi.responses import Response
from bielik_one.api.vllm_reads import BackendReadError, BackendBusyError, _cancel_event


class RequestExecutor:
    def __init__(self,max_requests,timeout):
        if not 1<=max_requests<=256:raise ValueError('max_requests must be between 1 and 256')
        if not math.isfinite(timeout) or timeout<=0:raise ValueError('request_timeout must be finite and positive')
        self.gate=asyncio.Semaphore(max_requests)
        self.timeout=timeout

    async def run(self,evaluate,body,request):
        loop=asyncio.get_running_loop()
        deadline=loop.time()+self.timeout
        try:
            await asyncio.wait_for(self.gate.acquire(),timeout=min(1.,self.timeout))
        except TimeoutError as exc:
            if loop.time()>=deadline:
                raise BackendReadError('Decision request deadline exceeded') from httpx.ReadTimeout('Decision request deadline exceeded')
            raise BackendBusyError('Decision request queue is full; retry later') from exc
        try:
            if await request.is_disconnected():
                self.gate.release()
                return Response(status_code=499)
            if loop.time()>=deadline:
                raise BackendReadError('Decision request deadline exceeded') from httpx.ReadTimeout('Decision request deadline exceeded')
        except BaseException:
            self.gate.release();raise
        cancel=threading.Event();token=_cancel_event.set(cancel)
        try:
            work=asyncio.create_task(asyncio.to_thread(evaluate,body))
        except BaseException:
            self.gate.release();raise
        finally:
            _cancel_event.reset(token)
        def finished(task):
            self.gate.release()
            if not task.cancelled():task.exception()
        work.add_done_callback(finished)
        async def disconnected():
            while not await request.is_disconnected():await asyncio.sleep(.1)
        watcher=asyncio.create_task(disconnected())
        try:
            done,_=await asyncio.wait((work,watcher),timeout=max(0.,deadline-loop.time()),return_when=asyncio.FIRST_COMPLETED)
            if not done:
                raise BackendReadError('Decision request deadline exceeded') from httpx.ReadTimeout('Decision request deadline exceeded')
            if work in done:return work.result()
            watcher.result()
            return Response(status_code=499)
        finally:
            cancel.set();watcher.cancel();await asyncio.gather(watcher,return_exceptions=True)
