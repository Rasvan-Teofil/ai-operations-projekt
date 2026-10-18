"""Live API verification with real local model. External URL behavior is date dependent."""
import argparse
import json
from datetime import datetime, timezone
import httpx
from src.config import REPORTS, LABELS


def main(base, missing_base):
    records = []
    with httpx.Client(timeout=60, trust_env=False) as client:
        health = client.get(base+'/health')
        health.raise_for_status()
        assert health.json()['model_loaded'], health.text
        records.append({'case':'health','status':health.status_code,'response':health.json()})
        cases = [
            ('text_de', {'text':'Die Firma meldet einen Gewinneinbruch und muss Beschäftigte entlassen.'},200),
            ('text_en', {'text':'The company reported record earnings and raised its outlook.'},200),
            ('text_neutral', {'text':'Die Hauptversammlung findet am Dienstag in Berlin statt.'},200),
            ('valid_url', {'url':'https://www.bbc.com/news/articles/c687z8127302o'},200),
            ('missing_fields', {},422),
            ('both_fields', {'text':'hi','url':'https://example.com'},422),
            ('empty_text', {'text':'  '},422),
            ('wrong_type', {'text':42},422),
            ('private_url', {'url':'http://127.0.0.1/'},422),
            ('no_article', {'url':'https://www.bbc.com/robots.txt'},422),
            ('robots_denied', {'url':'https://www.google.com/search?q=sentiment'},403),
            ('dns_failure', {'url':'https://second-path-no-such-host.invalid/news'},502),
        ]
        for name, payload, expected in cases:
            res = client.post(base+'/predict',json=payload)
            records.append({'case':name,'request':payload,'status':res.status_code,'expected':expected,'response':res.json()})
            assert res.status_code == expected, records[-1]
            if expected == 200:
                body = res.json()
                assert body['label'] in LABELS
                assert set(body['scores']) == set(LABELS)
                assert abs(sum(body['scores'].values())-1) < 1e-5
                assert body['model_version']
            print(name, res.status_code, flush=True)
        missing = client.get(missing_base+'/health')
        assert not missing.json()['model_loaded']
        res = client.post(missing_base+'/predict',json={'text':'This is a valid request.'})
        records.append({'case':'missing_model','status':res.status_code,'expected':503,'response':res.json()})
        assert res.status_code == 503
    result = {'checked_at_utc':datetime.now(timezone.utc).isoformat(),'api':base,'cases':records}
    (REPORTS/'smoke.json').write_text(json.dumps(result,indent=2,ensure_ascii=False))
    print('Live checks passed; reports/smoke.json written')

if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--api',default='http://127.0.0.1:8000')
    p.add_argument('--missing-api',default='http://127.0.0.1:8001')
    args=p.parse_args()
    main(args.api,args.missing_api)
