"""Read-only public gateway to the existing local MLflow server."""
from urllib.parse import parse_qs, unquote

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response


def read_routes():
    from mlflow.server.handlers import get_endpoints
    allowed=set()
    prefixes=('/api/2.0/mlflow/','/ajax-api/2.0/mlflow/','/api/3.0/mlflow/','/ajax-api/3.0/mlflow/')
    for path,handler,methods in get_endpoints():
        name=getattr(handler,'__name__','')
        if path.startswith(prefixes) and '<' not in path and '/gateway/' not in path:
            if name.startswith(('_get_','_search_','_list_')) or name=='get_metric_history_bulk_interval_handler':
                allowed.update((method,path) for method in methods if method in ('GET','POST'))
    return frozenset(allowed)


READ_ROUTES=read_routes()


def allowed(method,path,query=''):
    decoded=unquote(path)
    if '%' in decoded or '..' in decoded.split('/') or '\\' in decoded:return False
    for key,values in parse_qs(query).items():
        if key in ('path','artifact_path'):
            if any(v.startswith(('/','\\')) or '..' in v.split('/') or '\\' in v or '%' in v for v in values):return False
    if (method,decoded) in READ_ROUTES:return True
    if method not in ('GET','HEAD'):return False
    if decoded in ('/','/health','/version','/favicon.ico','/manifest.json'):return True
    if decoded.startswith(('/static-files/','/static/')):return True
    if decoded=='/get-artifact':
        return bool(parse_qs(query).get('path',[''])[0])
    return False


def allowed_graphql(body):
    import json
    from graphql import parse
    from graphql.language.ast import OperationDefinitionNode, FieldNode
    roots={'mlflowGetExperiment','mlflowGetMetricHistoryBulkInterval','mlflowGetRun',
           'mlflowListArtifacts','mlflowSearchModelVersions','mlflowSearchRuns','mlflowSearchDatasets'}
    try:
        payload=json.loads(body)
        if not isinstance(payload,dict) or not isinstance(payload.get('query'),str):return False
        document=parse(payload['query'])
        operations=[d for d in document.definitions if isinstance(d,OperationDefinitionNode)]
        return bool(operations) and all(all(isinstance(field,FieldNode) and field.name.value in roots
                   for field in op.selection_set.selections) for op in operations)
    except Exception:return False


app=FastAPI(docs_url=None,redoc_url=None,openapi_url=None)


@app.api_route('/{path:path}',methods=['GET','HEAD','POST','PUT','PATCH','DELETE','OPTIONS'])
async def proxy(request:Request,path:str):
    query=request.url.query
    body=await request.body()
    if len(body)>1024*1024:return Response(status_code=413)
    permitted = allowed_graphql(body) if path=='graphql' and request.method=='POST' else allowed(request.method,'/'+path,query)
    if not permitted:
        return JSONResponse({'error_code':'PERMISSION_DENIED','message':'Read-only MLflow access: modifying data is disabled.'},status_code=403)
    try:
        async with httpx.AsyncClient(timeout=30,follow_redirects=False,trust_env=False) as client:
            upstream=await client.request(request.method,'http://127.0.0.1:5000/'+path,
                params=list(request.query_params.multi_items()),content=body,
                headers={'Host':'127.0.0.1:5000','Content-Type':request.headers.get('content-type','application/json')})
        headers={k:v for k,v in upstream.headers.items() if k.lower() in ('content-type','cache-control','content-disposition','etag','last-modified')}
        headers['X-Presentation-Mode']='read-only-local-mlflow-live'
        headers['X-Content-Type-Options']='nosniff'
        headers['X-Frame-Options']='DENY'
        content=upstream.content
        if path=='' and 'text/html' in headers.get('content-type',''):
            setup=b'<script>try{localStorage.setItem("mlflow.workflowType_v1",JSON.stringify("machine_learning"))}catch(e){}</script>'
            content=content.replace(b'<head>',b'<head>'+setup,1)
        return Response(content,status_code=upstream.status_code,headers=headers)
    except httpx.HTTPError:
        return JSONResponse({'message':'Local MLflow service temporarily unavailable'},status_code=503)
