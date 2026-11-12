"""Local real-model serving checks; workload/queue effects are not classifier accuracy."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import httpx
import mlflow
import numpy as np
from src.config import ROOT, MODELS, REPORTS
from src.tracking import tracking
from src.development import DEV


def measure(name,path,port,queue_timeout):
    env={**os.environ,'SENTIMENT_ARTIFACT':str(path),'SENTIMENT_QUEUE_TIMEOUT':str(queue_timeout),
         'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    base=f'http://127.0.0.1:{port}'
    log=(ROOT/f'service_{name}.log').open('w')
    process=subprocess.Popen([sys.executable,'-m','uvicorn','app.main:app','--host','127.0.0.1','--port',str(port)],cwd=ROOT,env=env,stdout=log,stderr=log)
    try:
        ready=False
        for _ in range(160):
            if process.poll() is not None: raise RuntimeError(f'{name} process stopped; inspect service log')
            try:
                res=httpx.get(base+'/health',timeout=1,trust_env=False)
                if res.status_code==200 and res.json()['model_loaded']:
                    ready=True
                    break
            except httpx.HTTPError: pass
            time.sleep(.25)
        if not ready: raise RuntimeError(f'{name} API not ready')
        short='The company reported record profits and raised its revenue forecast.'
        with httpx.Client(timeout=60,trust_env=False) as client:
            for _ in range(20):
                assert client.post(base+'/predict',json={'text':short}).status_code==200
            steady=[]
            for _ in range(30):
                start=time.perf_counter()
                res=client.post(base+'/predict',json={'text':short})
                steady.append(1000*(time.perf_counter()-start))
                assert res.status_code==200
            long=(short+' ')*200
            assert len(long)<20000
            def send(_):
                start=time.perf_counter()
                res=client.post(base+'/predict',json={'text':long})
                return res.status_code,1000*(time.perf_counter()-start)
            begin=time.perf_counter()
            with ThreadPoolExecutor(max_workers=12) as pool:
                burst=list(pool.map(send,range(40)))
            elapsed=time.perf_counter()-begin
            assert all(status in (200,503) for status,_ in burst)
            recovery=client.post(base+'/predict',json={'text':short})
            assert recovery.status_code==200
            assert client.get(base+'/health').json()['model_loaded']
        return {'profile':name,'artifact':str(path.relative_to(ROOT)),'queue_timeout_seconds':queue_timeout,
                'steady_requests':30,'steady_p50_ms':float(np.median(steady)),
                'steady_p95_ms':float(np.percentile(steady,95)),
                'burst_requests':40,'burst_concurrency':12,'burst_text_characters':len(long),
                'burst_success_200':sum(status==200 for status,_ in burst),
                'burst_busy_503':sum(status==503 for status,_ in burst),
                'burst_p95_all_ms':float(np.percentile([v for _,v in burst],95)),
                'burst_elapsed_seconds':elapsed,'recovered':True}
    finally:
        process.terminate()
        try: process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        log.close()


def main():
    tracking()
    target=DEV/'service.json'
    if target.exists(): raise SystemExit('Service benchmark already recorded')
    results=[]
    for name,stage,port,timeout in [('reference','R0_reference',8010,60.),('bounded_bf16','R3_deploy',8011,.01)]:
        with mlflow.start_run(run_name=f'dev2-service-{name}',tags={'phase':'development-v2-service'}) as run:
            result=measure(name,MODELS/'development'/'fold0'/stage,port,timeout)
            result['run_id']=run.info.run_id
            mlflow.log_params({k:v for k,v in result.items() if isinstance(v,str)})
            mlflow.log_params({'purpose':'systems benchmark, not new accuracy measurement','isolation':'same fold0 training weights, bf16 representation differs'})
            mlflow.log_metrics({k:float(v) for k,v in result.items() if isinstance(v,(int,float,bool))})
            mlflow.log_dict(result,'service.json')
            results.append(result)
            print(result,flush=True)
    target.write_text(json.dumps({'note':'Deliberately low 10ms bounded queue exposes backpressure; production default is 2s. Synthetic local workload; no URL/network latency or real traffic claim.', 'results':results},indent=2))

if __name__=='__main__': main()
