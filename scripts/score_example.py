"""Load released weights and score an authored transaction without a dataset."""
import argparse
import json
from pathlib import Path
from temporal_fraud.data import digest
from temporal_fraud.streaming import MinuteBatch, Models, empty_account

def score(directory):
    directory = Path(directory)
    for name, expected in json.loads((directory / 'release-models.json').read_text()).items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError('Release model checksum mismatch')
    batch = MinuteBatch(sequence=0, minute='2022-09-07T12:00:00', events=[
        dict(event_id=1, source='authored-A', target='authored-B', paid=10, received=10,
             receiving_currency='US Dollar', payment_currency='US Dollar', format='ACH')])
    values, _ = Models(directory).score(batch.frame(), {'authored-A':empty_account(),'authored-B':empty_account()})
    assert all(0 <= v <= 1 for k,v in values[0].items() if k != 'event_id')
    return {'notice':'Authored example; verifies model loading, not predictive accuracy.', 'scores':values}

if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--models',required=True)
    print(json.dumps(score(parser.parse_args().models),indent=2))
