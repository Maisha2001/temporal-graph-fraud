import argparse
import json
import os
from pathlib import Path

from temporal_fraud.data import digest
from temporal_fraud.store import Store
from temporal_fraud.streaming import Models


parser=argparse.ArgumentParser()
parser.add_argument('--prepared',required=True)
parser.add_argument('--stream',required=True)
args=parser.parse_args()
root=Path(args.prepared)
manifest=json.loads((root/'preparation.json').read_text())
for name,expected in manifest['sha256'].items():
    if not (root/name).resolve().is_relative_to(root.resolve()) or digest(root/name)!=expected:
        raise ValueError('Prepared artifact digest mismatch')
store=Store(os.environ['FRAUD_DSN'],Models(root/'models'))
store.initialize()
store.create(args.stream,json.loads((root/'snapshot.json').read_text()))
print(json.dumps(store.status(args.stream)))
