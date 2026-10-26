# Temporal attention model

Each transaction queries both endpoint accounts. The model samples their latest
8 or 32 incident interactions with timestamps strictly earlier than the query
minute. All current-minute interactions stay invisible. Incoming and outgoing
interactions carry distinct signs; a self-transfer is indexed once. Ties use a
hash of observable transaction content, never labels or source-row identifiers.
Exact duplicate records have equivalent messages. Recent-neighbor sampling is
deterministic, not a uniform sample of all history.

The PyTorch Geometric `TransformerConv` layer performs two-head message passing
from the sampled interactions to the endpoint representations. Historical
messages encode transaction features and graph counts/gaps as they existed at
that interaction's timestamp. Edges encode log elapsed minutes, daily sine/cosine
and direction. Endpoint role embeddings distinguish source and destination;
account IDs have no learned embeddings. The two contexts and current transaction
representation feed a binary classifier. Empty neighborhoods use the root path.
This is a one-hop temporal attention model with 32 hidden units, inspired by
[temporal graph attention](https://arxiv.org/abs/2002.07962), using
[PyG's attention operator](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.nn.conv.TransformerConv.html).
It is not an exact TGAT reproduction and does not implement recurrent TGN memory
or multi-hop motif discovery.

## Fixed comparison

`configs/temporal-v1.json` fixes three epochs, AdamW (learning rate 0.001,
weight decay 0.0001), batches of 4,096, seed 42, and a square-root class weight.
All 1,851,794 eligible training events are used each epoch. Training batches are
shuffled offline; their graph queries still use only earlier interactions.
There are no negative subsamples, validation-selected epochs or hyperparameter
searches. The final third epoch is evaluated once per arm. A graph-free history
MLP uses the same preprocessing, labels, epochs, batch size and loss. It has fewer
parameters, so this is a practical baseline, not a parameter-matched ablation.
Both neighborhood budgets are reported regardless of their results.

The original account/time split and 24-hour label availability rule are unchanged.
The train-fitted transformer standardizes numerical features, compresses history
ages with signed log1p, and one-hot encodes training categories. Unknown categories
map to zero. Scalers and categories never fit validation data. Unlabeled context
from September 5 and earlier validation interactions is permitted, but withheld
accounts have no pre-fit history. September 8 onward is excluded by the loader
before labels are converted; it remains unscored.

## Reproduction and recovery

The README commands produce immutable per-epoch checkpoints, a latest checkpoint,
fitted preprocessing, a causal index, split membership, inputs with SHA-256 hashes,
validation predictions and reports. Checkpoints include optimizer state, completed
epoch, update count, training config and input fingerprint. `--resume` rejects
changed config/data or cached input hashes. Interrupted epochs restart from the
last completed epoch. Completed model predictions are retained and hash-checked.
Only load your own trusted local joblib/checkpoint files.

Every measured checkpoint is reloaded and checked on 512 validation events. Tests
also compare interrupted/resumed CPU optimization with uninterrupted optimization
and cover same-minute, future-edge, future-label, cold-account and empty-neighbor
behavior. CUDA attention reductions may be nondeterministic; identical seeds do
not promise bitwise weights on another device or software version. Exact saved
artifact hashes identify the measured run. Checkpoints and transaction-level
artifacts remain local; the repository includes aggregate evidence and commands
to reproduce them. No pretrained model or paid service is used.

The model is intentionally bounded to one hop and a small recent history. More
neighbors need not improve results. Class-weighted outputs are not calibrated
probabilities. The evidence concerns independent synthetic AML data under
CDLA-Sharing-1.0, not operational financial investigations. Torch and PyG retain
their own open-source licenses; the project code is MIT licensed.
