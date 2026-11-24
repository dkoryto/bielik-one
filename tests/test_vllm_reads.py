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
    started=threading.Event();exited=threading.Event()
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
    started=threading.Event();exited=threading.Event()
    class Engine:
        aliases={'decision'};temperatures={};calibration_source=None;backend_url='http://backend'
        def evaluate(self,body):
            if body.state=='second':return {'ok':True}
            started.set()
            try:
                while True:check_cancelled();time.sleep(.002)
            finally:exited.set()
        def close(self):pass
    class Connected:
        async def is_disconnected(self):return False
    async def run():
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
    asyncio.run(run())


def test_request_deadline_cancels_worker_and_reports_timeout():
    import asyncio,threading,time
    from bielik_one.api.vllm_reads import check_cancelled
    from bielik_one.api.remote_vllm import BackendReadError
    exited=threading.Event()
    class Engine:
        aliases={'decision'};temperatures={};calibration_source=None;backend_url='http://backend'
        def evaluate(self,body):
            try:
                while True:check_cancelled();time.sleep(.002)
            finally:exited.set()
        def close(self):pass
    class Connected:
        async def is_disconnected(self):return False
    async def run():
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
