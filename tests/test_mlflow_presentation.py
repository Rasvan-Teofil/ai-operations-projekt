"""Public presentation permits reading evidence, never modifying tracking data."""
import json

from src.studies.mlflow_presentation.proxy import allowed,allowed_graphql


def test_rest_reads_and_searches_work_but_mutation_endpoints_are_blocked():
    assert allowed('POST','/api/2.0/mlflow/runs/search')
    assert allowed('GET','/ajax-api/2.0/mlflow/experiments/get')
    for action in ['create','update','delete','restore','log-batch','set-tag']:
        assert not allowed('POST','/api/2.0/mlflow/runs/'+action)
    assert not allowed('DELETE','/api/2.0/mlflow/logged-models/model')


def test_graphql_read_operations_allow_aliases_without_allowing_writes_or_batches():
    def payload(query):return json.dumps({'query':query}).encode()
    assert allowed_graphql(payload('query { x: mlflowGetRun(input: {runId:"public"}) { run { info { runId } } } }'))
    assert allowed_graphql(payload('mutation { mlflowSearchRuns(input: {experimentIds:["2"]}) { runs { info { runId } } } }'))
    assert not allowed_graphql(payload('mutation { mlflowCreateRun { runId } }'))
    assert not allowed_graphql(payload('query { __schema { types { name } } }'))
    assert not allowed_graphql(b'[{"query":"query { mlflowGetRun { run { info { runId } } } }"}]')
    assert not allowed_graphql(b'not json')


def test_artifact_and_static_paths_cannot_escape_run_artifacts():
    assert allowed('GET','/get-artifact','path=result.json')
    assert allowed('GET','/get-artifact','path=training%2Ftelemetry-v1.json')
    assert allowed('GET','/static-files/static/js/main.js')
    assert not allowed('GET','/get-artifact','path=%2Fetc%2Fpasswd')
    assert not allowed('GET','/get-artifact','path=..%2Fprivate.json')
    assert not allowed('GET','/static-files/%252e%252e/private')
    assert not allowed('GET','/graphql')
    assert not allowed('POST','/get-artifact','path=public-result.json')
