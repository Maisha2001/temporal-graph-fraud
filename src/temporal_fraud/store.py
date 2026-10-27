import hashlib
import json
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from .streaming import advance_accounts


class Conflict(ValueError):
    pass


class Store:
    def __init__(self, dsn, models):
        self.dsn, self.models = dsn, models

    def ready(self):
        with psycopg.connect(self.dsn, connect_timeout=3) as conn:
            return conn.execute("SELECT 1").fetchone()[0] == 1

    def initialize(self):
        sql = (Path(__file__).parent / 'schema.sql').read_text()
        with psycopg.connect(self.dsn) as conn:
            conn.execute(sql)

    def create(self, stream, snapshot):
        if snapshot['model_fingerprint'] != self.models.fingerprint:
            raise Conflict('Snapshot uses different model artifacts')
        if len(snapshot['accounts']) > 25000:
            raise ValueError('Bounded replay supports at most 25000 accounts')
        fingerprint = hashlib.sha256(json.dumps(snapshot, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        with psycopg.connect(self.dsn) as conn:
            conn.execute('INSERT INTO fraud_streams(stream, last_minute, model_fingerprint, snapshot_fingerprint) VALUES (%s,%s,%s,%s)',
                         (stream, snapshot['last_minute'], self.models.fingerprint, fingerprint))
            with conn.cursor() as cur:
                cur.executemany('INSERT INTO fraud_accounts(stream, owner, state) VALUES (%s,%s,%s)',
                                [(stream, owner, Jsonb(state)) for owner, state in snapshot['accounts'].items()])

    def apply(self, stream, batch, fail_before_commit=False):
        fingerprint = batch.fingerprint()
        with psycopg.connect(self.dsn) as conn:
            conn.execute("SET LOCAL lock_timeout = '10s'")
            row = conn.execute('SELECT sequence, last_minute, model_fingerprint FROM fraud_streams WHERE stream=%s FOR UPDATE',
                               (stream,)).fetchone()
            if row is None:
                raise KeyError('Unknown replay stream')
            if row[2] != self.models.fingerprint:
                raise Conflict('Model version differs from stream')
            existing = conn.execute('SELECT fingerprint, response FROM fraud_batches WHERE stream=%s AND sequence=%s',
                                    (stream, batch.sequence)).fetchone()
            if existing:
                if existing[0] != fingerprint:
                    raise Conflict('Sequence already committed with different content')
                return {**existing[1], 'duplicate': True}
            if batch.sequence != row[0] + 1 or batch.minute <= row[1]:
                raise Conflict('Expected next sequence and strictly later complete minute')
            if not self.models.first_minute <= batch.minute < self.models.end_minute:
                raise Conflict('Minute outside the pinned development serving interval')
            frame = batch.frame()
            ids = frame.event_id.tolist()
            if conn.execute('SELECT 1 FROM fraud_events WHERE stream=%s AND event_id=ANY(%s) LIMIT 1', (stream, ids)).fetchone():
                raise Conflict('Event ID was already processed')
            owners = sorted(set(frame.source) | set(frame.target))
            accounts = dict(conn.execute('SELECT owner,state FROM fraud_accounts WHERE stream=%s AND owner=ANY(%s)',
                                         (stream, owners)).fetchall())
            if len(accounts) != len(owners):
                raise Conflict('Account outside the explicitly bootstrapped replay scope')
            scores, vectors = self.models.score(frame, accounts)
            advance_accounts(frame, vectors, accounts)
            response = {'stream': stream, 'sequence': batch.sequence, 'minute': batch.minute.isoformat(),
                        'model_fingerprint': self.models.fingerprint, 'scores': scores, 'duplicate': False}
            with conn.cursor() as cur:
                cur.executemany('UPDATE fraud_accounts SET state=%s WHERE stream=%s AND owner=%s',
                                [(Jsonb(state), stream, owner) for owner, state in accounts.items()])
                cur.executemany('INSERT INTO fraud_events(stream,event_id,minute,event,score) VALUES (%s,%s,%s,%s,%s)',
                                [(stream, int(event['event_id']), batch.minute, Jsonb(event), Jsonb(score))
                                 for event, score in zip(frame.drop(columns='timestamp').to_dict('records'), scores)])
            conn.execute('INSERT INTO fraud_batches(stream,sequence,fingerprint,response) VALUES (%s,%s,%s,%s)',
                         (stream, batch.sequence, fingerprint, Jsonb(response)))
            conn.execute('UPDATE fraud_streams SET sequence=%s,last_minute=%s WHERE stream=%s',
                         (batch.sequence, batch.minute, stream))
            if fail_before_commit:
                raise RuntimeError('Injected failure before database commit')
        return response

    def status(self, stream):
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute('SELECT sequence,last_minute,model_fingerprint,snapshot_fingerprint, (SELECT count(*) FROM fraud_events e WHERE e.stream=s.stream) FROM fraud_streams s WHERE stream=%s',
                               (stream,)).fetchone()
            if row is None:
                raise KeyError('Unknown replay stream')
        return dict(sequence=row[0], last_minute=row[1].isoformat(), model_fingerprint=row[2],
                    snapshot_fingerprint=row[3], events=row[4])

    def batch(self, stream, sequence):
        with psycopg.connect(self.dsn) as conn:
            row = conn.execute('SELECT response FROM fraud_batches WHERE stream=%s AND sequence=%s', (stream,sequence)).fetchone()
        if row is None:
            raise KeyError('Unknown batch')
        return row[0]
