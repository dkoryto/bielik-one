"""Shared vLLM completion reads, anchored chunk merging and optional bounded scheduling."""
from __future__ import annotations

import contextvars
import math
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import httpx
from bielik_one.api.http_guard import backend_request_id

MAX_LOGPROB_TOKEN_IDS = 128
_cancel_event = contextvars.ContextVar('decision_cancel_event', default=None)


class BackendReadError(RuntimeError):
    pass


class BackendBusyError(BackendReadError):
    pass


def check_cancelled():
    event = _cancel_event.get()
    if event is not None and event.is_set():
        raise BackendReadError('Decision request was cancelled')


def conditional_logprobs(values, wanted):
    logits = [float(values[i]) for i in wanted]
    if not logits or not all(math.isfinite(v) for v in logits):
        raise ValueError('Invalid answer logits')
    peak = max(logits)
    log_z = peak + math.log(math.fsum(math.exp(v-peak) for v in logits))
    return {i:v-log_z for i,v in zip(wanted,logits,strict=True)}


class VllmReadTransport:
    def __init__(self, url, model, *, workers=8, client=None, queue_size=64,
                 admission_timeout=1., timeout=180):
        if not 1 <= workers <= 256 or (queue_size is not None and queue_size < 0) or admission_timeout <= 0 or timeout <= 0:
            raise ValueError('Invalid backend concurrency or timeout configuration')
        self.url, self.model, self.workers = url.rstrip('/'), model, workers
        self.client = client or httpx.Client(timeout=httpx.Timeout(timeout,connect=10),trust_env=False,
            limits=httpx.Limits(max_connections=workers,max_keepalive_connections=workers))
        self.pool = ThreadPoolExecutor(max_workers=workers,thread_name_prefix='decision-read')
        # None preserves the existing adapter scheduling/admission behavior.
        self.slots = None if queue_size is None else threading.BoundedSemaphore(workers+queue_size)
        self.admission_timeout = admission_timeout
        self.lifecycle = threading.Lock()
        self.closed = False

    @staticmethod
    def read_count(output_ids, anchor=None, canary=None):
        anchor = output_ids[0] if anchor is None else anchor
        count = sum(i != anchor for i in output_ids)
        return max(1,math.ceil(count/(MAX_LOGPROB_TOKEN_IDS-1-(canary is not None))))

    def read(self, prompt, output_ids, *, anchor=None, canary=None, url=None, model=None,
             extra=None, reject_clamped=True, logprobs=1, anchor_first=False):
        """Every chunk repeats an anchor, cancelling per-read vocabulary normalizers."""
        check_cancelled()
        if not prompt or not output_ids or len(set(output_ids)) != len(output_ids):
            raise BackendReadError('Expected nonempty prompt and unique answer token IDs')
        anchor = output_ids[0] if anchor is None else anchor
        remaining = [i for i in output_ids if i != anchor]
        width = MAX_LOGPROB_TOKEN_IDS-1-(canary is not None)
        logits = {anchor:0.} if anchor in output_ids else {}
        for start in range(0,max(1,len(remaining)),width):
            check_cancelled()
            chunk = remaining[start:start+width]
            requested = ([anchor]+chunk) if anchor_first else (chunk+[anchor])
            payload = dict(model=model or self.model,prompt=prompt,max_tokens=1,temperature=1,
                top_p=1,top_k=-1,min_p=0,repetition_penalty=1,logprobs=logprobs,
                logprob_token_ids=requested+([] if canary is None else [canary]),
                return_tokens_as_token_ids=True,seed=0)
            if extra:payload.update(extra)
            rid = backend_request_id()
            try:
                response=self.client.post((url or self.url)+'/v1/completions',json=payload,
                    headers={'X-Request-Id':rid} if rid else None)
                check_cancelled();response.raise_for_status();body=response.json()
                table={int(k.removeprefix('token_id:')):float(v)
                    for k,v in body['choices'][0]['logprobs']['top_logprobs'][0].items()}
                if body['usage']['prompt_tokens'] != len(prompt):raise ValueError('backend changed prompt length')
                if any(i not in table or not math.isfinite(table[i]) for i in requested):
                    raise ValueError('missing or non-finite requested logprobs')
                if canary is not None and (canary not in table or table[canary]>-9990):
                    raise ValueError('backend did not apply decision heads to a flagged base row (logits processor missing?)')
                if reject_clamped and any(table[i]<=-9990 for i in requested):
                    raise ValueError('invalid or clamped requested logprob')
            except BackendReadError:
                raise
            except (httpx.HTTPError,ValueError,KeyError,IndexError,TypeError) as exc:
                raise BackendReadError('vLLM did not return a complete decision read') from exc
            logits.update({i:table[i]-table[anchor] for i in chunk})
        return logits

    def _read(self, task):
        prompt,wanted=task
        return self.read(prompt,wanted,reject_clamped=False,logprobs=0,anchor_first=True)

    def _submit(self, function, task):
        check_cancelled()
        if self.slots is not None and not self.slots.acquire(timeout=self.admission_timeout):
            raise BackendBusyError('Decision read queue is full; retry later')
        try:
            with self.lifecycle:
                if self.closed:raise BackendReadError('Decision reader is closed')
                future=self.pool.submit(contextvars.copy_context().run,function,*task)
            if self.slots is not None:future.add_done_callback(lambda _:self.slots.release())
            return future
        except BaseException:
            if self.slots is not None:self.slots.release()
            raise

    def map(self, function, tasks):
        if self.slots is None:
            futures=[self._submit(function,task) for task in tasks]
            return [future.result() for future in futures]
        tasks=iter(enumerate(tasks));pending={};results={}
        try:
            for _ in range(self.workers):
                item=next(tasks,None)
                if item is None:break
                i,task=item;pending[self._submit(function,task)]=i
            while pending:
                check_cancelled()
                done,_=wait(pending,timeout=.1,return_when=FIRST_COMPLETED)
                for future in done:
                    i=pending.pop(future);results[i]=future.result()
                    item=next(tasks,None)
                    if item is not None:
                        j,task=item;pending[self._submit(function,task)]=j
            return [results[i] for i in range(len(results))]
        finally:
            for future in pending:future.cancel()

    def __call__(self,tasks):
        return self.map(self._read,[(task,) for task in tasks])

    def close(self):
        with self.lifecycle:
            if self.closed:return
            self.closed=True
        self.pool.shutdown(wait=True,cancel_futures=True)
        self.client.close()
