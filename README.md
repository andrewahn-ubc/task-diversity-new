# Task Diversity 2

This repository is a runnable JAX implementation of the requested modified
Figure 6 continual-learning experiment. It compares recurrent PPO with the
same learner augmented by Continual Backpropagation (CBP), runs seven
sequential Banyan task distributions at five diversity levels, and launches
non-destructive held-out adaptation branches after every training stage.

The cluster workflow is intentionally two commands. It builds a pinned,
offline-capable CUDA environment, validates it on a real GPU, checkpoints
inside every one-hour allocation, chains continuation jobs, and creates the
three requested aggregate plots only after every run is complete.

## Quick start on Narval

First copy or clone this repository onto a filesystem visible to compute
nodes, log into Narval as `taegyoem`, and `cd` to the repository root. The
defaults are configured for account `def-mijungp` and a concurrency request of
100.

```bash
bash scripts/setup_compute_canada.sh \
  --runtime-root /scratch/taegyoem/td2-runtime \
  --data-root /scratch/taegyoem/td2-data

bash scripts/submit_all.sh \
  --cluster narval \
  --account def-mijungp \
  --output-root /scratch/taegyoem/td2-results \
  --max-concurrent 100
```

These values are also the script defaults, so `bash scripts/submit_all.sh` is
equivalent after setup. Narval requests one A100 for each GPU job.
The setup command writes a small runtime locator under `.runtime/`, so the
second command works even though the large virtual environment is in scratch.

Do not run the setup command on a compute node. It downloads wheels and exact
source revisions once on the login node; all scheduled jobs then run offline.
The source packages are installed as non-editable snapshots, inherited
Python/CUDA paths are sanitized, and the wheelhouse is checksummed before
reuse, so queued jobs cannot silently pick up later checkout edits or a user
CUDA library ahead of JAX's pinned CUDA 12 wheels.
The repository, runtime, data, and result paths must remain available until
the experiment finishes. Scratch is fast but is not archival storage, so copy
the final results elsewhere afterward. Do not edit the repository, rerun setup,
or resume the same shared output root from another cluster while jobs are
active.

## What is run

| Item | Implemented protocol |
|---|---|
| Methods | `ppo`, `ppo_cbp` |
| Replicates | 5 seeds from the supplied YAML |
| Training sequence | 7 disjoint task distributions |
| Diversity | task-tree width `n` and layout width `n`, with `n²` in `{1, 16, 256, 4096, 65536}` |
| Per-stage budget | 100,000,000 nominal environment steps |
| Actual per-stage budget | 381 updates × 2048 envs × 128 steps = 99,876,864 steps |
| Main streams | 2 methods × 5 diversities × 5 seeds = 50 |
| Adaptation probes | 7 checkpoints per main stream = 350 branches |
| Held-out probe | the same 20 unseen task-tree/layout pairs for every stage, diversity, method, and seed |
| Adaptation budget | another 99,876,864 actual steps per branch (100M nominal) |
| Total work | about 69.9 billion environment steps, split evenly between main and adaptation runs |

Integer PPO rollouts cannot consume exactly 100M steps. The implementation
uses PureJaxRL's floor-division convention, records nominal and actual step
axes separately, and never silently rounds the budget upward.

The phrase “after each of the 10 training distributions” conflicts with the
explicit request to use seven distributions. The configuration resolves this
in favor of seven and creates a branch after each of stages 1 through 7.

Each adaptation branch loads the complete immutable parent learner state:
parameters, Adam moments and per-parameter counters, and—under CBP—unit ages
and utility estimates. It resets the environment, GRU carry, and per-phase
learning-rate schedule, trains only on the fixed held-out set, and writes to a
separate directory. No branch can alter the original continual trajectory.

## PPO and CBP implementation

`src/td2/ppo.py` ports the recurrent rollout, reverse-time GAE,
trajectory-minibatch PPO update, clipped policy/value losses, and recurrent
state behavior from PureJaxRL's `ppo_rnn.py` at commit
`31756b197773a52db763fdbe6d635e4b46522a73`. PureJaxRL is a research-code
repository rather than an importable library, so the audited algorithm is
ported into this environment adapter instead of imported at runtime.

The missing hidden-layer count is resolved with the closest PureJaxRL-style
recurrent topology:

```text
flat observation -> Dense(1024, ReLU) --+
previous-action one-hot ----------------+-> GRU(256)
                                      -> Dense(1024, ReLU) -> policy logits
                                      -> Dense(1024, ReLU) -> scalar value
```

That is one shared fully connected layer before the GRU and one hidden fully
connected layer in each head; output projections are not counted as hidden
layers.

The paper states that the recurrent state uses the current observation and
previous action but does not specify the PPO concatenation point. We use the
minimal convention consistent with its PQN description: concatenate the
previous-action one-hot after the observation embedding and before the GRU.
It is the all-zero vector at initialization and after every episode boundary.

CBP is applied after every optimizer step to the observation embedding, GRU,
actor hidden layer, and critic hidden layer. It uses the paper's CBP-specific
values: utility decay `0.986`, maturity `5000` gradient steps, and replacement
rate `5e-5` with fractional accumulation. The generate-and-test mechanics are
adapted from `loss-of-plasticity` commit
`a6b79580d85f3025bdb601566d3627c5f489f13b`: low-utility selection, Kaiming
incoming-weight regeneration for ReLU layers, zero outgoing weights, downstream
bias compensation, and resets of first moments, second moments, and
elementwise Adam counters. Regenerated GRU coordinates retain PureJaxRL's
Flax defaults (LeCun-normal input kernels and orthogonal recurrent kernels),
and replaced GRU coordinates are also zeroed in all live recurrent carries.

For a controlled comparison, PPO and PPO+CBP use the same optimizer and all
PPO hyperparameters from the supplied YAML. The paper's separately tuned CBP
learning rate, AdamW weight decay, and larger CBP network are therefore not
substituted for the requested YAML settings.

## Configuration decisions and known deviations

The original attached YAML is preserved verbatim in
`configs/source_hyperparameters.yaml`; `configs/experiment.yaml` inherits it.
Two environment settings are deliberately overridden because the YAML names a
private `TwoAgentsGridWorld`, while Figure 6 uses the released single-agent
Banyan environment:

- `TREE_DEPTH` is changed from 4 to 6, and depths 1–6 are sampled uniformly.
- `INCLUDE_RULES_IN_OBS` is changed from true to false, matching the paper's
  partially observable setup.
- The runnable environment has one agent, not the unavailable private
  two-agent environment.

All applicable PPO values remain from the YAML: learning rate `3e-4`, 5%
warmup and linear annealing, 2048 environments, 128 rollout steps, two epochs,
four trajectory minibatches, horizon 256, actor clip `0.2`, value clip `10`,
entropy coefficient `0.05`, `gamma=0.99`, `lambda=0.8`, value coefficient
`0.8`, gradient norm `0.5`, FC width 1024, and GRU width 256. Adam state is
continued across task distributions; only the requested warmup/annealing
schedule restarts at a distribution boundary.

Environment-only values from the private YAML are used where Banyan has the
same meaning, including the `-0.1` distractor penalty and 0.5 distractor-pair
density. `REW_SHAPING_HORIZON` is not applied: released Banyan implements the
paper's fixed first-pickup shaping reward and does not expose the private
environment's shaping schedule. W&B is disabled in jobs; CSV logs are the
authoritative record.

The source `EMBEDDING: float` and `EMBED_DIM: 512` settings are retained as
provenance but do not add another learned encoder: released Banyan already
returns one fixed flat float observation, which the selected PureJaxRL-style
`Dense(1024)` observation embedding consumes directly. `RULESET_EMBED_DIM` is
inactive because task rules are deliberately hidden from the observation.

Task generation uses the released Banyan depth-six, cross-round-disjoint
generator at commit `7bb5d385cb2b48f49ab01d0c9a47b9d404e2ecc3`. Lower
diversities are nested prefixes of the `n=256` task and layout banks. Layout
masks and depth-six topologies are disjoint across rounds, and producer
signatures do not repeat across rounds. The seven training rounds use Banyan's
public CLI at `n=256`, with a 200,000-attempt retry budget and a cap of 15 new
unary producer signatures per training round to preserve cross-round capacity.
The fixed held-out set uses the same pinned sampler at logical `n=20`, after
reconstructing all training blocker sets, and is padded only to the common
on-disk stride. This avoids sampling an unused eighth `n=256` round. Because
the held-out call uses a private function, its exact API name, commit, seed
derivation, logical/storage widths, and hashes are recorded in the manifest.

The paper states that leaves come from a 40-object pool sampled from an
8-shape × 10-color vocabulary; the manifest records an explicit seeded
40-of-80 selection. One upstream difference remains: the released generator's
deterministic producer mapping can emit intermediate or root objects from
Banyan's full 10-shape × 12-color domain. In the audited deterministic bundle,
38.13% of producer-rule occurrences fall outside the 80-object vocabulary.
Reproducing an 80-object-closed mapping would require changing the released
codec and fixed combination table, so this implementation keeps the pinned
behavior and makes the deviation explicit.

These choices make the run a controlled, executable adaptation of Figure 6,
not a claim of bit-for-bit reproduction of the authors' unpublished training
artifacts.

## Scheduler design

Every submitted batch file has `#SBATCH --time=01:00:00`. Main and adaptation
workers use at most 2,700 seconds (45 minutes) of useful work, retain 15 minutes
for compilation/checkpoint flex, and receive `USR1` ten minutes before Slurm's
hard deadline. Checkpoints are content-addressed, checksummed, fsynced, and
published through an atomic pointer. State and metadata fall back together if
the newest blob is corrupt, and every pointer is bound to hashes of the merged
configuration and immutable data manifest. A successful incomplete job submits
one `afterok` continuation; a completed stage writes an immutable checkpoint and
`DONE` marker before its adaptation branch can be submitted.

The dependency graph is:

```text
CPU data preparation
  -> real-GPU package/CUDA/PPO+CBP smoke gate
    -> 50-way main array
       -> stage checkpoints -> 350 independent adaptation branches
          -> completeness gate -> aggregate plots
```

`--max-concurrent` limits eligible main-array tasks. Because this experiment
has 50 main streams, the requested value of 100 leaves all 50 eligible at
once; it does not manufacture another 50 streams. At the end of each main
worker allocation, every newly durable stage releases its adaptation branch;
branches are not held until the full seven-stage stream finishes. Slurm and
your fair share determine actual concurrency.
The one-hour requests are scheduling-friendly, but no script can guarantee a
five-day completion time without measuring throughput and queue delay on your
allocation. The GPU smoke job fails the whole dependency chain before
production if package, CUDA, compilation, or checkpoint validation fails.

## Monitoring and recovery

Inspect jobs and logs with:

```bash
squeue --me
tail -f /scratch/taegyoem/td2-results/slurm-logs/*.out
```

Normal continuation is automatic. If jobs are cancelled or fail, wait until
no `td2-*` jobs are queued or running, then use the same paths with `--resume`:

```bash
bash scripts/submit_all.sh \
  --cluster narval \
  --account def-mijungp \
  --output-root /scratch/taegyoem/td2-results \
  --max-concurrent 100 \
  --resume
```

Resume verifies that no Task Diversity jobs are active, archives stale
submission locks, revalidates data and the GPU runtime, and continues only
incomplete checkpoints. Reusing an output root without `--resume` fails
closed, which prevents accidental duplicate experiments.

## Results

After all 50 main streams and 350 probes have `DONE` markers, the aggregate job
creates:

- `plots/figure6a_evaluation_success.{png,pdf}` — greedy evaluation success,
  with task-distribution boundaries and mean ± SEM over five seeds;
- `plots/training_episodic_return.{png,pdf}` — on-policy training episodic
  returns in the same continual format;
- `plots/heldout_adaptation.{png,pdf}` — one panel per source checkpoint,
  showing success during the 100M-step held-out branches;
- `plots/adaptation_summary.csv` — initial/final success, normalized AUC,
  time to 0.8 success, and threshold-reached fraction.

The plotter refuses to create a polished aggregate when any expected
method/diversity/seed/stage trajectory is absent.

## Validation performed

All 36 local tests pass. They cover configuration derivation, PPO/GAE and
learning-rate semantics, recurrent reset/carry behavior, CBP replacement and
optimizer-state resets, atomic checkpoint fallback, deterministic task/layout
generation, completeness checks, and plot aggregation. An end-to-end
production-sized data build completed all seven training rounds plus the
logical-20 held-out round in 7m41s, validated 1,812 unique layouts and all
cross-round exclusions, and published 25 hashed payloads; an immediate rerun
verified the bundle in under one second. Fresh CPU smoke runs covered PPO and
PPO+CBP main training, held-out adaptation, and checkpoint reload. A forced
mid-stage stop followed by resume produced exactly the same decoded learner
state and all 16 metric rows as an uninterrupted run. A complete synthetic
result matrix exercised every plotting output.

The cluster smoke gate additionally checks pinned PEP 440 public package
versions (while accepting Alliance's `+computecanada` local wheel label),
imports the pinned Banyan modules under Alliance Python 3.13.2, forces a CUDA matrix
operation, then compiles and executes one PPO+CBP update with the full
production shapes (2048 environments, 128 steps, FC 1024, GRU 256). It reloads
the resulting checkpoint and verifies the immutable stage marker before the
50-stream array is released. No Alliance job was submitted from this local
workspace, so that gate is intentionally the remaining hardware-specific
validation before production is released.

## Upstream sources

- [PureJaxRL](https://github.com/luchris429/purejaxrl/tree/31756b197773a52db763fdbe6d635e4b46522a73)
- [Loss of Plasticity implementation](https://github.com/shibhansh/loss-of-plasticity/tree/a6b79580d85f3025bdb601566d3627c5f489f13b)
- [Banyan](https://github.com/nhshah15/banyan/tree/7bb5d385cb2b48f49ab01d0c9a47b9d404e2ecc3)
- Purab Seth et al., “Task diversity produces systematic transfer but inhibits continual reinforcement learning,” arXiv:2606.00880v2 (2026), supplied with this task.

See `THIRD_PARTY_NOTICES.md` for licenses and the exact code-level provenance.
