import math
import httpx,pytest
from bielik_one.api.vllm_reads import VllmReadTransport as VllmTokenReadout, conditional_logprobs, check_cancelled
from bielik_one.api.remote_vllm import build_remote_app
from bielik_one.api.server import SystemOneIn,QuestionIn

def test_remote_uses_original_token_ids_no_plugin():
    calls=[]
    def respond(request):
        import json
        d=json.loads(request.content);calls.append(d)
        return httpx.Response(200,json={'usage':{'prompt_tokens':2},'choices':[{'logprobs':{'top_logprobs':[{'token_id:7':-4,'token_id:8':-3}]}}]})
    c=httpx.Client(transport=httpx.MockTransport(respond))
    r=VllmTokenReadout('http://backend','base',client=c)
    result=r([([1,2],[7,8])]);r.close()
    assert result==[{7:0,8:1}]
    assert calls[0]['model']=='base' and calls[0]['logprob_token_ids']==[7,8]
    assert 'vllm_xargs' not in calls[0] and 'logit_bias' not in calls[0]


def test_selected_softmax_stable_and_shift_invariant():
    a=conditional_logprobs({0:1000,1:1001},[0,1]);b=conditional_logprobs({0:-1000,1:-999},[0,1])
    assert a==pytest.approx(b) and sum(math.exp(x) for x in a.values())==pytest.approx(1)


def test_shared_read_pool_bounds_concurrent_requests():
    import threading,time
    from concurrent.futures import ThreadPoolExecutor
    active=0;peak=0;lock=threading.Lock()
    def respond(request):
        nonlocal active,peak
        with lock:active+=1;peak=max(peak,active)
        time.sleep(.01)
        with lock:active-=1
        return httpx.Response(200,json={'usage':{'prompt_tokens':2},'choices':[{'logprobs':{'top_logprobs':[{'token_id:7':-4}]}}]})
    reader=VllmTokenReadout('http://backend','base',workers=2,client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        with ThreadPoolExecutor(max_workers=4) as callers:
            batches=list(callers.map(lambda _:reader([([1,2],[7])]*10),range(4)))
        assert peak==2 and all(x==[{7:0}]*10 for x in batches)
    finally:reader.close()


def test_backend_timeout_is_typed():
    from bielik_one.api.remote_vllm import BackendReadError
    def timeout(request):raise httpx.ReadTimeout('test',request=request)
    reader=VllmTokenReadout('http://backend','base',client=httpx.Client(transport=httpx.MockTransport(timeout)))
    try:
        with pytest.raises(BackendReadError) as error:reader([([1],[7])])
        assert isinstance(error.value.__cause__,httpx.ReadTimeout)
    finally:reader.close()


def test_queue_exhaustion_and_close_during_work():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from bielik_one.api.remote_vllm import BackendReadError
    entered=threading.Event();release=threading.Event()
    def respond(request):
        entered.set();assert release.wait(3)
        return httpx.Response(200,json={'usage':{'prompt_tokens':1},'choices':[{'logprobs':{'top_logprobs':[{'token_id:7':-4}]}}]})
    reader=VllmTokenReadout('http://backend','base',workers=1,queue_size=0,admission_timeout=.02,client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            active=pool.submit(reader,[([1],[7])]);assert entered.wait(2)
            try:
                with pytest.raises(BackendReadError,match='queue is full'):reader([([1],[7])])
                closing=pool.submit(reader.close)
                # close marks the lifecycle closed before waiting for the active read.
                import time
                deadline=time.monotonic()+2
                while not reader.closed and time.monotonic()<deadline:time.sleep(.001)
                assert reader.closed and not closing.done()
            finally:release.set()
            assert active.result(timeout=2)==[{7:0}];closing.result(timeout=2)
        reader.close()
        with pytest.raises(BackendReadError,match='closed'):reader([([1],[7])])
    finally:release.set();reader.close()


def test_overload_http_returns_retryable_503():
    from bielik_one.api.remote_vllm import BackendBusyError,build_remote_app
    from fastapi.testclient import TestClient
    class BusyEngine:
        aliases={'decision'};temperatures={};calibration_source=None;backend_url='http://backend'
        def evaluate(self,body):raise BackendBusyError('queue is full; retry later')
        def close(self):pass
    with TestClient(build_remote_app(BusyEngine(),chat_model='base')) as client:
        r=client.post('/v1/systemone',json={'model':'decision','state':'s','questions':{'q':{'type':'noul','instructions':'Is this correct?'}}})
        assert r.status_code==503 and r.headers['retry-after']=='1'
        assert r.json()['code']=='BACKEND_UNAVAILABLE'


def test_native_disconnect_stops_cooperative_worker():
    import asyncio,threading,time
    from bielik_one.api.vllm_reads import check_cancelled
    started=threading.Event();exited=threading.Event();stop=threading.Event()
    class Engine:
        aliases={'decision'};temperatures={};calibration_source=None;backend_url='http://backend'
        def evaluate(self,body):
            started.set()
            try:
                for _ in range(1000):check_cancelled();time.sleep(.002)
                pytest.fail('worker did not cancel')
            finally:exited.set()
        def close(self):pass
    class Disconnected:
        async def is_disconnected(self):return started.is_set()
    async def run():
        app=build_remote_app(Engine(),max_requests=1)
        async with app.router.lifespan_context(app):
            endpoint=next(r.endpoint for r in app.routes if getattr(r,'path',None)=='/v1/systemone')
            body=SystemOneIn(state='s',model='decision',questions={'q':QuestionIn(type='noul')})
            response=await endpoint(body,Disconnected())
            assert response.status_code==499
            for _ in range(100):
                if exited.is_set():break
                await asyncio.sleep(.01)
            assert exited.is_set()
    asyncio.run(run())


def test_endpoint_task_cancellation_recovers_http_permit():
    import asyncio,threading,time
    from bielik_one.api.vllm_reads import check_cancelled
    started=threading.Event();exited=threading.Event();stop=threading.Event()
    class Engine:
        aliases={'decision'};temperatures={};calibration_source=None;backend_url='http://backend'
        def evaluate(self,body):
            if body.state=='second':return {'ok':True}
            started.set()
            try:
                while not stop.is_set():check_cancelled();time.sleep(.002)
            finally:exited.set()
        def close(self):pass
    class Connected:
        async def is_disconnected(self):return False
    async def run():
        try:
            app=build_remote_app(Engine(),max_requests=1)
            async with app.router.lifespan_context(app):
                endpoint=next(r.endpoint for r in app.routes if getattr(r,'path',None)=='/v1/systemone')
                body=SystemOneIn(state='first',model='decision',questions={'q':QuestionIn(type='noul')})
                task=asyncio.create_task(endpoint(body,Connected()))
                for _ in range(100):
                    if started.is_set():break
                    await asyncio.sleep(.01)
                assert started.is_set();task.cancel()
                with pytest.raises(asyncio.CancelledError):await task
                assert await endpoint(body.model_copy(update={'state':'second'}),Connected())=={'ok':True}
                assert exited.is_set()
        finally:stop.set()
    asyncio.run(run())


def test_request_deadline_cancels_worker_and_reports_timeout():
    import asyncio,threading,time
    from bielik_one.api.vllm_reads import check_cancelled
    from bielik_one.api.remote_vllm import BackendReadError
    exited=threading.Event();stop=threading.Event()
    class Engine:
        aliases={'decision'};temperatures={};calibration_source=None;backend_url='http://backend'
        def evaluate(self,body):
            try:
                while not stop.is_set():check_cancelled();time.sleep(.002)
            finally:exited.set()
        def close(self):pass
    class Connected:
        async def is_disconnected(self):return False
    async def run():
        try:
            app=build_remote_app(Engine(),max_requests=32,request_timeout=.02)
            async with app.router.lifespan_context(app):
                endpoint=next(r.endpoint for r in app.routes if getattr(r,'path',None)=='/v1/systemone')
                body=SystemOneIn(state='s',model='decision',questions={'q':QuestionIn(type='noul')})
                with pytest.raises(BackendReadError) as error:await endpoint(body,Connected())
                assert isinstance(error.value.__cause__,httpx.ReadTimeout)
                for _ in range(100):
                    if exited.is_set():break
                    await asyncio.sleep(.01)
                assert exited.is_set()
        finally:stop.set()
    asyncio.run(run())


@pytest.mark.parametrize('count', [129, 255, 300])
def test_remote_chunks_original_tokens_and_normalizes_after_merge(count):
    import json
    calls=[]
    def respond(request):
        d=json.loads(request.content);calls.append(d)
        values={f'token_id:{i}': -10-i/100 for i in d['logprob_token_ids']}
        return httpx.Response(200,json={'usage':{'prompt_tokens':2},
            'choices':[{'logprobs':{'top_logprobs':[values]}}]})
    reader=VllmTokenReadout('http://backend','base',client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        wanted=list(range(count))
        values=reader([([5,6],wanted)])[0]
        assert len(calls)==math.ceil(count/128)
        assert all(c['prompt']==[5,6] and len(c['logprob_token_ids'])<=128 for c in calls)
        assert sorted({i for c in calls for i in c['logprob_token_ids']})==wanted
        assert all(wanted[0] in c['logprob_token_ids'] for c in calls)
        probabilities=conditional_logprobs(values,wanted)
        assert sum(math.exp(v) for v in probabilities.values())==pytest.approx(1)
        assert math.exp(probabilities[0]-probabilities[count-1])==pytest.approx(math.exp((count-1)/100))
    finally:
        reader.close()


def test_chunk_timeout_preserves_original_timeout_cause():
    import json
    from bielik_one.api.remote_vllm import BackendReadError
    calls=[]
    def respond(request):
        d=json.loads(request.content);calls.append(d)
        if len(calls)==2:
            raise httpx.ReadTimeout('second chunk timed out',request=request)
        return httpx.Response(200,json={'usage':{'prompt_tokens':2},
            'choices':[{'logprobs':{'top_logprobs':[{f'token_id:{i}':-10 for i in d['logprob_token_ids']}]}}]})
    reader=VllmTokenReadout('http://backend','base',client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(BackendReadError) as caught:
            reader([([5,6],list(range(129)))])
        assert isinstance(caught.value.__cause__,httpx.ReadTimeout)
        assert len(calls)==2
    finally:
        reader.close()


def test_original_token_chunks_cancel_different_vocabulary_normalizers():
    import json
    calls=[]
    def respond(request):
        body=json.loads(request.content);calls.append(body)
        shift=1000*len(calls)
        table={f'token_id:{i}':i/100-shift for i in body['logprob_token_ids']}
        return httpx.Response(200,json={'usage':{'prompt_tokens':1},'choices':[{'logprobs':{'top_logprobs':[table]}}]})
    reader=VllmTokenReadout('http://backend','base',client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        values=reader([([1],list(range(255)))])[0]
        assert values==pytest.approx({i:i/100 for i in range(255)},abs=1e-10)
    finally:reader.close()


def test_extreme_equal_scores_still_normalize():
    values=conditional_logprobs({0:-1e20,1:-1e20},[0,1])
    assert [math.exp(v) for v in values.values()]==pytest.approx([.5,.5])


@pytest.mark.parametrize('table', [[], {'token_id:7':10**400}, {'token_id:7':0,'token_id:9':'NaN'},
    {'token_id:7':0,'token_id:9':'-Infinity'}])
def test_malformed_table_and_nonfinite_canary_are_typed(table):
    from bielik_one.api.vllm_reads import BackendReadError
    client=httpx.Client(transport=httpx.MockTransport(lambda _:httpx.Response(200,json={
        'usage':{'prompt_tokens':1},'choices':[{'logprobs':{'top_logprobs':[table]}}]})))
    reader=VllmTokenReadout('http://backend','base',client=client)
    try:
        with pytest.raises(BackendReadError):reader.read([1],[7],canary=9)
    finally:reader.close()


def test_queued_disconnect_skips_dispatch_and_recovers_permit():
    import asyncio
    from bielik_one.api.request_execution import RequestExecutor
    class Disconnected:
        async def is_disconnected(self):return True
    async def run():
        executor=RequestExecutor(1,1)
        await executor.gate.acquire()
        calls=[]
        task=asyncio.create_task(executor.run(lambda _:calls.append(1),None,Disconnected()))
        await asyncio.sleep(.01)
        executor.gate.release()
        assert (await task).status_code==499
        assert calls==[] and executor.gate._value==1
    asyncio.run(run())


def test_request_deadline_includes_admission():
    import asyncio
    from bielik_one.api.request_execution import RequestExecutor
    from bielik_one.api.vllm_reads import BackendReadError
    class Connected:
        async def is_disconnected(self):return False
    async def run():
        executor=RequestExecutor(1,.02)
        await executor.gate.acquire()
        calls=[]
        try:
            with pytest.raises(BackendReadError) as caught:
                await asyncio.wait_for(executor.run(lambda _:calls.append(1),None,Connected()),.5)
            assert isinstance(caught.value.__cause__,httpx.ReadTimeout)
            assert not calls and executor.gate._value==0
        finally:executor.gate.release()
    asyncio.run(run())


def test_disconnect_watcher_failure_is_not_reported_as_disconnect():
    import asyncio,threading
    from bielik_one.api.request_execution import RequestExecutor
    stop=threading.Event()
    class Broken:
        calls=0
        async def is_disconnected(self):
            self.calls+=1
            if self.calls>1:raise RuntimeError('watch failed')
            return False
    async def run():
        executor=RequestExecutor(1,1)
        try:
            with pytest.raises(RuntimeError,match='watch failed'):
                await executor.run(lambda _:stop.wait(2),None,Broken())
        finally:stop.set()
    asyncio.run(run())


def test_bounded_direct_app_retains_schema_and_typed_deadline():
    import time
    from fastapi.testclient import TestClient
    from bielik_one.api.server import build_app
    class Engine:
        def evaluate(self,body):time.sleep(.04);return {'ok':True}
    plain=build_app(Engine()).openapi()['paths']['/v1/systemone']['post']['responses']['200']
    bounded=build_app(Engine(),max_requests=1,request_timeout=.01)
    schema=bounded.openapi()['paths']['/v1/systemone']['post']['responses']['200']['content']['application/json']['schema']
    expected=plain['content']['application/json']['schema']
    assert {k:v for k,v in schema.items() if k!='title'}=={k:v for k,v in expected.items() if k!='title'}
    with TestClient(bounded) as client:
        response=client.post('/v1/systemone',json={'model':'decision','state':'s','questions':{'q':{'type':'noul'}}})
        assert response.status_code==504 and response.json()['code']=='TIMEOUT'


@pytest.mark.parametrize('queue_size',[None,4])
def test_failed_row_drains_active_siblings_before_releasing_http_permit(queue_size):
    import asyncio,threading
    from bielik_one.api.request_execution import RequestExecutor
    from bielik_one.api.vllm_reads import BackendReadError
    entered=threading.Event();release=threading.Event()
    reader=VllmTokenReadout('http://backend','base',workers=2,queue_size=queue_size)
    def row(i):
        if i==0:
            assert entered.wait(2)
            raise BackendReadError('row failed')
        entered.set();assert release.wait(3)
    class Connected:
        async def is_disconnected(self):return False
    async def run():
        executor=RequestExecutor(1,2)
        first=asyncio.create_task(executor.run(lambda _:reader.map(row,[(0,),(1,)]),None,Connected()))
        try:
            assert await asyncio.to_thread(entered.wait,2)
            second=asyncio.create_task(executor.run(lambda _:'second',None,Connected()))
            await asyncio.sleep(.1)
            assert not first.done() and not second.done()
            release.set()
            with pytest.raises(BackendReadError,match='row failed'):await first
            assert await second=='second'
        finally:
            release.set()
            await asyncio.gather(first,return_exceptions=True)
    try:asyncio.run(run())
    finally:release.set();reader.close()


def test_cancelled_queued_adapter_read_does_not_wait_for_other_request():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from bielik_one.api.vllm_reads import _cancel_event,BackendReadError
    reader=VllmTokenReadout('http://backend','base',workers=1,queue_size=None)
    entered=threading.Event();release=threading.Event();queued=threading.Event();cancel=threading.Event()
    original_submit=reader._submit
    def submit(fn,task):
        future=original_submit(fn,task)
        if task==('queued',):queued.set()
        return future
    reader._submit=submit
    def row(value):
        if value=='active':entered.set();assert release.wait(3)
        else:pytest.fail('cancelled queued read dispatched')
    def queued_request():
        token=_cancel_event.set(cancel)
        try:return reader.map(row,[('queued',)])
        finally:_cancel_event.reset(token)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            active=pool.submit(reader.map,row,[('active',)])
            try:
                assert entered.wait(2)
                waiting=pool.submit(queued_request);assert queued.wait(2)
                cancel.set()
                with pytest.raises(BackendReadError,match='cancelled'):waiting.result(timeout=1)
                assert not active.done()
            finally:release.set()
            active.result(timeout=2)
    finally:release.set();reader.close()


def test_shutdown_cancelled_future_exits_bounded_map():
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from bielik_one.api.vllm_reads import BackendReadError,_cancel_event
    reader=VllmTokenReadout('http://backend','base',workers=1,queue_size=2)
    entered=threading.Event();release=threading.Event();queued=threading.Event();cancel=threading.Event()
    original_submit=reader._submit
    def submit(fn,task):
        future=original_submit(fn,task)
        if task==('queued',):queued.set()
        return future
    reader._submit=submit
    def row(value):
        if value=='active':entered.set();assert release.wait(3)
        else:pytest.fail('shutdown did not cancel queued read')
    def queued_request():
        token=_cancel_event.set(cancel)
        try:return reader.map(row,[('queued',)])
        finally:_cancel_event.reset(token)
    try:
        with ThreadPoolExecutor(max_workers=3) as pool:
            active=pool.submit(reader.map,row,[('active',)])
            try:
                assert entered.wait(2)
                waiting=pool.submit(queued_request)
                assert queued.wait(2)
                closing=pool.submit(reader.close)
                with pytest.raises(BackendReadError,match='cancelled'):waiting.result(timeout=1)
            finally:cancel.set();release.set()
            active.result(timeout=2);closing.result(timeout=2)
    finally:cancel.set();release.set();reader.close()


def test_bounded_direct_app_maps_admission_overload():
    from fastapi.testclient import TestClient
    from bielik_one.api.server import build_app
    from bielik_one.api.vllm_reads import BackendBusyError
    class Engine:
        def evaluate(self,body):raise BackendBusyError('queue is full; retry later')
    with TestClient(build_app(Engine(),max_requests=1)) as client:
        response=client.post('/v1/systemone',json={'model':'decision','state':'s','questions':{'q':{'type':'noul'}}})
        assert response.status_code==503 and response.headers['retry-after']=='1'
        assert response.json()['code']=='BACKEND_UNAVAILABLE'


def test_bounded_admission_cancellation_releases_http_permit_promptly():
    import asyncio,threading
    from concurrent.futures import ThreadPoolExecutor
    from bielik_one.api.request_execution import RequestExecutor
    entered=threading.Event();release=threading.Event();waiting=threading.Event()
    reader=VllmTokenReadout('http://backend','base',workers=1,queue_size=0,admission_timeout=2)
    def row():entered.set();assert release.wait(4)
    def queued_work(_):
        waiting.set()
        return reader.map(lambda:pytest.fail('cancelled request dispatched'),[()])
    class Connected:
        async def is_disconnected(self):return False
    async def run():
        executor=RequestExecutor(1,3)
        task=asyncio.create_task(executor.run(queued_work,None,Connected()))
        try:
            assert await asyncio.to_thread(waiting.wait,2)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):await task
            # No active reads for this request: its permit must recover without
            # waiting for the unrelated read or the two-second admission cap.
            assert await asyncio.wait_for(executor.run(lambda _:'recovered',None,Connected()),.5)=='recovered'
            assert not release.is_set()
        finally:
            release.set()
            task.cancel();await asyncio.gather(task,return_exceptions=True)
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            active=pool.submit(reader.map,row,[()])
            try:
                assert entered.wait(2)
                asyncio.run(run())
            finally:release.set()
            active.result(timeout=2)
    finally:release.set();reader.close()


def test_cancel_after_slot_acquisition_returns_slot():
    import threading
    from bielik_one.api.vllm_reads import BackendReadError,_cancel_event
    reader=VllmTokenReadout('http://backend','base',workers=1,queue_size=0)
    cancel=threading.Event()
    class CancellingSlot:
        released=0
        def acquire(self,timeout):cancel.set();return True
        def release(self):self.released+=1
    slot=CancellingSlot();reader.slots=slot
    token=_cancel_event.set(cancel)
    try:
        with pytest.raises(BackendReadError,match='cancelled'):reader.map(lambda:None,[()])
        assert slot.released==1
    finally:_cancel_event.reset(token);reader.close()


@pytest.mark.parametrize('name',['admission_timeout','timeout'])
@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_transport_rejects_nonfinite_deadlines(name,value):
    with pytest.raises(ValueError):VllmTokenReadout('http://backend','base',**{name:value})
