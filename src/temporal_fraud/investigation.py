import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .data import digest


def load_bundle(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if digest(directory / 'cases.json') != manifest['cases_sha256']:
        raise ValueError('Investigation bundle digest mismatch')
    bundle = json.loads((directory / 'cases.json').read_text())
    ids = [case['event_id'] for case in bundle['cases']]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate investigation case')
    for case in bundle['cases']:
        if any(edge['timestamp'] >= case['timestamp'] for edge in case['context']):
            raise ValueError('Investigation context contains current or future transactions')
    return bundle, manifest


def create_app(directory=None):
    bundle, manifest = load_bundle(directory or os.environ['FRAUD_CASES'])
    cases = {case['event_id']: case for case in bundle['cases']}
    assets = Path(__file__).parent / 'web'
    app = FastAPI(title='Transaction Atlas', version='1.0.0')
    app.mount('/assets', StaticFiles(directory=assets), name='assets')

    @app.get('/')
    def home():
        return FileResponse(assets / 'index.html')

    @app.get('/api/overview')
    def overview():
        return {k: v for k, v in bundle.items() if k != 'cases'} | {'case_count': len(cases), 'bundle_sha256': manifest['cases_sha256']}

    @app.get('/api/cases')
    def ranked(model: str = 'temporal_attention_32', minute: int = Query(30, ge=1, le=30),
               search: str = Query('', max_length=128), cold: bool = False):
        if model not in ['temporal_attention_32', 'history_xgboost']:
            raise HTTPException(422, 'Unknown model')
        selected = [c for c in cases.values() if c['minute_index'] < minute and
                    (not cold or c['cold']) and
                    search.lower() in (c['source'] + ' ' + c['target'] + ' ' + str(c['event_id'])).lower()]
        selected.sort(key=lambda c: (-c[model + '_calibrated'], c['event_id']))
        return [{k: v for k, v in c.items() if k not in ['context', 'label']} for c in selected]

    @app.get('/api/cases/{event_id}')
    def detail(event_id: int, outcomes: bool = False):
        if event_id not in cases:
            raise HTTPException(404, 'Case outside the saved review sample')
        case = dict(cases[event_id])
        if not outcomes:
            case.pop('label', None)
        return case

    @app.get('/health')
    def health():
        return {'status': 'ready', 'mode': 'saved_model_replay', 'cases': len(cases),
                'test_data_used': False, 'bundle_sha256': manifest['cases_sha256']}

    return app
