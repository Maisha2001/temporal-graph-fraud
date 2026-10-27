import os

from fastapi import FastAPI, HTTPException, Request

from .store import Conflict, Store
from .streaming import MinuteBatch, Models


def create_app(store=None):
    if store is None:
        store = Store(os.environ['FRAUD_DSN'], Models(os.environ['FRAUD_MODELS']))
    app = FastAPI(title='Temporal fraud replay', version='0.4.0')

    @app.middleware('http')
    async def body_limit(request: Request, call_next):
        if request.method == 'POST':
            body = bytearray()
            async for chunk in request.stream():
                body.extend(chunk)
                if len(body) > 8_000_000:
                    from starlette.responses import JSONResponse
                    return JSONResponse({'detail': 'Batch exceeds 8 MB'}, status_code=413)
            request._body = bytes(body)
        return await call_next(request)

    @app.get('/health')
    def health():
        import psycopg
        try:
            store.ready()
        except psycopg.Error as exc:
            raise HTTPException(503, "Database unavailable") from exc
        return {'status': 'ready', 'pid': os.getpid(), 'model_fingerprint': store.models.fingerprint,
                'scope': 'bounded synthetic development replay; no production fraud claim'}

    @app.get('/streams/{stream}')
    def status(stream: str):
        try:
            return store.status(stream)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.get('/streams/{stream}/batches/{sequence}')
    def batch(stream: str, sequence: int):
        try:
            return store.batch(stream, sequence)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post('/streams/{stream}/batches')
    def score(stream: str, batch: MinuteBatch):
        try:
            return store.apply(stream, batch)
        except Conflict as exc:
            raise HTTPException(409, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    return app
