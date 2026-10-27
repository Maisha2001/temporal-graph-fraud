CREATE TABLE IF NOT EXISTS fraud_streams (
    stream text PRIMARY KEY,
    sequence bigint NOT NULL DEFAULT -1,
    last_minute timestamp NOT NULL,
    model_fingerprint text NOT NULL,
    snapshot_fingerprint text NOT NULL
);
CREATE TABLE IF NOT EXISTS fraud_accounts (
    stream text REFERENCES fraud_streams(stream),
    owner text NOT NULL,
    state jsonb NOT NULL,
    PRIMARY KEY (stream, owner)
);
CREATE TABLE IF NOT EXISTS fraud_events (
    stream text REFERENCES fraud_streams(stream),
    event_id bigint NOT NULL,
    minute timestamp NOT NULL,
    event jsonb NOT NULL,
    score jsonb NOT NULL,
    PRIMARY KEY (stream, event_id)
);
CREATE TABLE IF NOT EXISTS fraud_batches (
    stream text REFERENCES fraud_streams(stream),
    sequence bigint NOT NULL,
    fingerprint text NOT NULL,
    response jsonb NOT NULL,
    PRIMARY KEY (stream, sequence)
);
CREATE INDEX IF NOT EXISTS fraud_events_time ON fraud_events(stream, minute);
