# Task Diversity 2

This repository is a runnable JAX implementation of the requested modified
Figure 6 continual-learning experiment. It runs recurrent PPO with Continual
Backpropagation (CBP) across ten sequential Banyan task distributions at all
nine Figure 6 diversity levels, and launches non-destructive held-out
adaptation branches from every training-stage checkpoint. It also runs a
matched fixed-distribution control: each stream shares the main run's first
100M steps, then remains on distribution 1 for the rest of its 1B-step budget.

The cluster workflow is intentionally two commands. It builds a pinned,
offline-capable CUDA environment, validates it on a real GPU, checkpoints
inside every one-hour GPU allocation, requeues incomplete array elements, and
creates the four requested aggregate plots only after every run is complete.

## Quick start on Narval

First copy or clone this repository onto a filesystem visible to compute
nodes, log into Narval as `taegyoem`, and `cd` to the repository root. The
submission charges every job to Kelsey Allen's `def-krallen` allocation, with
a GPU array concurrency cap of 100.

```bash
bash scripts/setup_compute_canada.sh \
  --runtime-root /scratch/taegyoem/td2-runtime \
  --data-root /scratch/taegyoem/td2-data-10dist-9div-cbp-seeded100

bash scripts/submit_all.sh \
  --cluster narval \
  --data-root /scratch/taegyoem/td2-data-10dist-9div-cbp-seeded100 \
  --output-root /scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b \
  --max-concurrent 100
```

These values are also the script defaults, so `bash scripts/submit_all.sh` is
equivalent after setup. That one submission command launches the continual,
held-out, and fixed-distribution experiments. Narval requests one A100 for
each GPU job.
The setup command writes a small runtime locator under `.runtime/`, so the
second command works even though the large virtual environment is in scratch.

### Run the clean fixed-distribution 2B experiment

To rerun the fixed-distribution control from initialization with a proper 2B
learning-rate schedule, use one command:

```bash
bash scripts/submit_fixed_2b.sh
```

The launcher reuses the five existing seed-specific task bundles, but it does
not load any prior model checkpoint. It submits exactly 45 fresh streams (nine
diversity levels by five seeds), all charged to `def-krallen`.
It also excludes `ng31004` and `ng31104` by default because those nodes caused
the earlier Lustre failures. Results are written to
`/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh`.
If that submission is interrupted, wait until no `td2-*` jobs remain and run
`bash scripts/submit_fixed_2b.sh --resume`.

Each stream starts at optimizer step zero with the same deterministic seed and
distribution-1 task bank as before. Its schedule warms from zero to `3e-4`
over 3,048 optimizer steps (5% of the 2B horizon), then cosine-decays smoothly
to zero at optimizer step 60,960. There is no learning-rate discontinuity at
1B. The run consumes 7,620 PPO updates, or 1,997,537,280 actual environment
steps per stream (1,999,876,864 on the nominal plotting axis).

When all streams finish, the new root contains
`plots/fixed_distribution_evaluation_success_2b_fresh.{png,pdf}`. No held-out
branches, data-generation jobs, or original continual experiments are
launched, and none of the completed results are overwritten.

### Run the GRU-512 clean fixed-distribution 2B experiment

To run the same clean 2B experiment with only the recurrent width changed from
256 to 512, use:

```bash
bash scripts/submit_fixed_2b_gru512.sh
```

The existing seed-specific task bundles are reused. The wider GRU, its
pre-GRU projection, recurrent carry, CBP state, and actor/critic inputs are
created from initialization; no GRU-256 checkpoint is loaded. Results are
kept separately under
`/scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed2b-fresh-gru512`.
Resume an interrupted submission only after its earlier jobs have stopped:

```bash
bash scripts/submit_fixed_2b_gru512.sh --resume
```

Do not run the setup command on a compute node. It downloads wheels and exact
dependency revisions once on the login node; all scheduled jobs then run
offline. Binary dependencies are installed as non-editable snapshots, while
the launcher deliberately imports `src/` from the current checkout. Therefore
a normal `git pull` that changes only this repository's code does not require
another setup run. Inherited Python/CUDA paths are sanitized and the wheelhouse
is checksummed before reuse.
The repository, runtime, data, and result paths must remain available until
the experiment finishes. Scratch is fast but is not archival storage, so copy
the final results elsewhere afterward. Do not edit the repository, rerun setup,
or resume the same shared output root from another cluster while jobs are
active.

## What is run

| Item | Implemented protocol |
|---|---|
| Method | `ppo_cbp` only; no pure-PPO production baseline |
| Replicates | 5 seeds from the supplied YAML |
| Training sequence | 10 disjoint task distributions |
| Diversity | task-tree width `n` and layout width `n`, with `n²` in `{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536}` |
| Per-stage budget | 100,000,000 nominal environment steps |
| Actual per-stage budget | 381 updates × 2048 envs × 128 steps = 99,876,864 steps |
| Main streams | 1 method × 9 diversities × 5 seeds = 45 |
| Fixed-distribution control | The same 45 streams train only on distribution 1 for 1B nominal steps |
| Actual fixed budget | 10 × 99,876,864 = 998,768,640 steps per stream |
| Shared prefix | Each fixed stream reuses its main stream's exact first-100M checkpoint; those steps are computed once |
| Adaptation probes | 10 checkpoints per main stream = 450 branches |
| Held-out probe | 100 unseen task-tree/layout pairs per seed, fixed across that seed's stages and diversities |
| Adaptation budget | another 99,876,864 actual steps per branch (100M nominal) |
| Fixed held-out probes | None |
| Total unique work | 130,339,307,520 actual environment steps, counting the shared prefix only once |

Integer PPO rollouts cannot consume exactly 100M steps. The implementation
uses PureJaxRL's floor-division convention, records nominal and actual step
axes separately, and never silently rounds the budget upward.

Each adaptation branch loads the complete immutable parent learner state:
parameters, Adam moments and per-parameter counters, and—under CBP—unit ages
and utility estimates. It resets the environment, GRU carry, and a branch-local
learning-rate schedule, trains only on its seed's fixed 100-task held-out set,
and writes to a separate directory. No branch can alter the original continual
trajectory.

At the first stage boundary, the fixed-distribution offshoot loads the entire
immutable main runner checkpoint, including the environment and recurrent
state, RNG, PPO optimizer, CBP statistics, and learning-rate schedule. The main
stream then moves to distribution 2 while the offshoot continues on
distribution 1 without a reset or learning-rate restart. The fixed run writes
under `fixed/` and never launches a held-out branch.

## PPO and CBP implementation

`src/td2/ppo.py` ports the recurrent rollout, reverse-time GAE,
trajectory-minibatch PPO update, clipped policy/value losses, and recurrent
state behavior from PureJaxRL's `ppo_rnn.py` at commit
`31756b197773a52db763fdbe6d635e4b46522a73`. PureJaxRL is a research-code
repository rather than an importable library, so the audited algorithm is
ported into this environment adapter instead of imported at runtime.

The supplied authors' implementation resolves the previously missing
hidden-layer count and recurrent topology:

```text
flat observation -> Dense(512, ReLU) -> Dense(256) -> GRU(256)
                                                       |-> Dense(1024, ReLU)
                                                       |   -> Dense(1024, ReLU)
                                                       |   -> policy logits
                                                       `-> Dense(1024, ReLU)
                                                           -> Dense(1024, ReLU)
                                                           -> scalar value
```

The GRU consumes only the projected current observation and the episode-reset
mask, exactly as in that code; no previous-action one-hot is concatenated.
Output projections are not counted as hidden layers.

CBP is applied after every optimizer step to both pre-GRU layers, the GRU, and
both hidden layers in each actor/critic head. It uses the paper's CBP-specific
values: utility decay `0.986`, maturity `5000` gradient steps, and replacement
rate `5e-5` with fractional accumulation. The generate-and-test mechanics are
adapted from `loss-of-plasticity` commit
`a6b79580d85f3025bdb601566d3627c5f489f13b`: low-utility selection, Kaiming
incoming-weight regeneration for ReLU layers, zero outgoing weights, downstream
bias compensation, and resets of first moments, second moments, and
elementwise Adam counters. Regenerated GRU coordinates retain PureJaxRL's
Flax defaults (LeCun-normal input kernels and orthogonal recurrent kernels),
and replaced GRU coordinates are also zeroed in all live recurrent carries.

The production matrix contains only PPO+CBP. It uses the optimizer and PPO
hyperparameters from the supplied YAML; the paper's separately tuned CBP
learning rate, AdamW weight decay, and larger CBP network are not substituted
for those requested settings. Pure PPO remains available only in reduced unit
tests so the shared PPO core can still be regression-tested.

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
warmup and cosine annealing, 2048 environments, 128 rollout steps, two epochs,
four trajectory minibatches, horizon 256, actor clip `0.2`, value clip `10`,
entropy coefficient `0.05`, `gamma=0.99`, `lambda=0.8`, value coefficient
`0.8`, gradient norm `0.5`, embedding width 512, FC width 1024, and GRU width
256. Adam state and one continuous 1B-step learning-rate schedule are carried
across task distributions. The fixed offshoot inherits that same schedule at
100M and continues it without a restart. A fresh 100M-step schedule is used
only inside each independent held-out adaptation branch.

Environment-only values from the private YAML are used where Banyan has the
same meaning, including the `-0.1` distractor penalty and 0.5 distractor-pair
density. `REW_SHAPING_HORIZON` is not applied: released Banyan implements the
paper's fixed first-pickup shaping reward and does not expose the private
environment's shaping schedule. W&B is disabled in jobs; CSV logs are the
authoritative record.

The source `EMBEDDING: float` and `EMBED_DIM: 512` settings now directly define
the authors' learned float-observation embedding. `RULESET_EMBED_DIM` remains
inactive because task rules are deliberately hidden from the observation.

Task generation uses the released Banyan depth-six, cross-round-disjoint
generator at commit `7bb5d385cb2b48f49ab01d0c9a47b9d404e2ecc3`. Each learner
seed gets its own immutable bundle under `DATA_ROOT/seed_<seed>`, containing
ten training banks and one 100-task held-out set. Within one seed, all nine
diversity streams share those banks and lower diversities are nested prefixes
of the `n=256` task and layout banks. Thus comparisons across diversity are
paired within a seed, while different learner seeds use independently seeded
training and held-out tasks. Layout masks and depth-six topologies are disjoint
across rounds within each bundle, and producer signatures do not repeat across
rounds. The ten training rounds use Banyan's same pinned per-round sampler at
`n=256`, with its default 20,000-attempt retry budget and a cap of 10 new unary
producer signatures per training round, reserving at least 20 of the 120
possible unary signatures for the disjoint held-out round. Completed raw rounds
are atomically staged and reused after an interruption instead of restarting at
round one.
Each seed's fixed held-out set uses the same pinned sampler at logical `n=100`,
after reconstructing all training blocker sets, and is padded only to the
common on-disk stride. This avoids sampling an unused eleventh `n=256` round.
Because the held-out call uses a private function, its exact API name, commit,
seed derivation, logical/storage widths, and hashes are recorded in that
seed's manifest.

The paper states that each distribution's leaves come from a 40-object pool
sampled from an 8-shape × 10-color vocabulary; the manifest records each
round's deterministic 40-of-80 selection. One upstream difference remains: the
released generator's deterministic producer mapping can emit intermediate or
root objects from Banyan's full 10-shape × 12-color domain.
Reproducing an 80-object-closed mapping would require changing the released
codec and fixed combination table, so this implementation keeps the pinned
behavior and makes the deviation explicit.

These choices make the run a controlled, executable adaptation of Figure 6,
not a claim of bit-for-bit reproduction of the authors' unpublished training
artifacts.

## Scheduler design

Data preparation is a five-element CPU array, one independently seeded bundle
per learner seed. Each element requests four hours without a shorter hidden
watchdog; every completed task-distribution round remains resumable if an
element is interrupted. Every GPU job still requests one hour. Main, fixed,
and adaptation workers use at most 2,700 seconds (45 minutes) of
useful work, retain 15 minutes for compilation/checkpoint flex, and receive
`USR1` ten minutes before Slurm's hard deadline. Checkpoints are
content-addressed, checksummed, fsynced, and
published through an atomic pointer. State and metadata fall back together if
the newest blob is corrupt, and every pointer is bound to hashes of the merged
configuration and immutable data manifest. A successful incomplete job submits
itself for requeue under the same array task ID. Every completed stage writes an
immutable checkpoint and `DONE` marker before the held-out phase begins.

The launcher submits one array per production phase, all charged to
`def-krallen`. The dependency graph is:

```text
5-way per-seed CPU data preparation
  -> real-GPU package/CUDA/PPO+CBP smoke gate
    -> shared first-100M prefix for all 45 streams
       +--> continual continuation through distributions 2-10 --+
       +--> fixed continuation on distribution 1 through 1B ----+
                                                               -> 450-way held-out adaptation
                                                                  (continual checkpoints only)
                                                               -> aggregate plots
```

The shared prefix is an immutable checkpoint, not a second approximate run.
The continual and fixed continuations can then run in parallel without
recomputing their common first 100M steps. Incomplete tasks atomically save and
requeue the same array element, so continuations remain under their original
array throttles instead of becoming unbounded standalone jobs. The held-out
arrays still load each of the ten continual checkpoints independently; no
held-out jobs are created for `fixed/`.

The fixed `def-krallen` allocation is used by every submission. The launcher
applies `--max-concurrent 100` directly to each GPU array; the dependency graph
prevents production phases from overlapping, so at most 100 experiment GPU
tasks can run concurrently. Slurm and the allocation's fair share determine
how many tasks actually run.
The one-hour GPU and four-hour CPU requests are scheduling-friendly, but no
script can guarantee a five-day completion time without measuring throughput
and queue delay on your allocation. The GPU smoke job fails the whole
dependency chain before production if package, CUDA, compilation, or checkpoint
validation fails.

## Monitoring and recovery

Inspect jobs and logs with:

```bash
squeue --me
tail -f /scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b/slurm-logs/*.out
```

Data-preparation logs are named `data-a-<array-job-id>_<seed-index>.out` and
`data-b-<array-job-id>_<seed-index>.out`. Prefix, combined main/fixed
continuation, and adaptation logs similarly start with `prefix-a|b`,
`continuation-a|b`, and `adapt-a|b`.

Normal continuation is automatic. If jobs are cancelled or fail, wait until
no `td2-*` jobs are queued or running, then use the same paths with `--resume`:

```bash
bash scripts/submit_all.sh \
  --cluster narval \
  --data-root /scratch/taegyoem/td2-data-10dist-9div-cbp-seeded100 \
  --output-root /scratch/taegyoem/td2-results-10dist-9div-cbp-seeded100-fixed1b \
  --max-concurrent 100 \
  --resume
```

Resume verifies that no Task Diversity jobs are active, revalidates data and
the GPU runtime, and continues only incomplete checkpoints. Reusing an output
root without `--resume` fails closed, which prevents accidental duplicate
experiments. The old seven-distribution data and result roots, and any earlier
shared 20-task data bundle, are scientifically incompatible with this
protocol; do not pass them to `--resume`.

## Results

After all 45 main streams, 45 fixed streams, and 450 probes have `DONE`
markers, the aggregate job creates four graph types:

- `plots/figure6a_evaluation_success.{png,pdf}` — greedy evaluation success,
  with task-distribution boundaries and mean ± SEM over five seeds;
- `plots/training_episodic_return.{png,pdf}` — on-policy training episodic
  returns in the same continual format;
- `plots/heldout_adaptation.{png,pdf}` — one panel per source checkpoint,
  showing success during the 100M-step held-out branches;
- `plots/fixed_distribution_evaluation_success.{png,pdf}` — greedy evaluation
  success while training on distribution 1 for the full 1B-step budget, with
  the shared first-100M measurements and no distribution-boundary dotted
  lines;
- the separate clean 2B workflow writes
  `plots/fixed_distribution_evaluation_success_2b_fresh.{png,pdf}` in its own
  result root;
- `plots/adaptation_summary.csv` — initial/final success, normalized AUC,
  time to 0.8 success, and threshold-reached fraction.

The plotter refuses to create a polished aggregate when any expected
method/diversity/seed/stage trajectory is absent or when an unconfigured stream
(such as a pure-PPO baseline) is present.

## Validation performed

The local suite covers configuration and work-plan derivation, PPO/GAE and
learning-rate semantics, recurrent reset/carry behavior, CBP replacement and
optimizer-state resets, atomic checkpoint fallback, deterministic task/layout
generation, completeness checks, and plot aggregation. Reduced CPU smoke runs
cover both the shared PPO core and PPO+CBP, while the production launcher emits
only PPO+CBP work. The two-distribution smoke checkpoint verifies that the main
schedule continues across a distribution boundary while each held-out branch
uses a fresh local schedule.

The cluster smoke gate additionally checks pinned PEP 440 public package
versions (while accepting Alliance's `+computecanada` local wheel label),
imports the pinned Banyan modules under Alliance Python 3.13.2, forces a CUDA matrix
operation, then compiles and executes one PPO+CBP update with the full
production shapes (2048 environments, 128 steps, FC 1024, GRU 256). It reloads
the resulting checkpoint and verifies the immutable stage marker before the
45-stream array is released. No Alliance job was submitted from this local
workspace, so that gate is intentionally the remaining hardware-specific
validation before production is released.

## Upstream sources

- [PureJaxRL](https://github.com/luchris429/purejaxrl/tree/31756b197773a52db763fdbe6d635e4b46522a73)
- [Loss of Plasticity implementation](https://github.com/shibhansh/loss-of-plasticity/tree/a6b79580d85f3025bdb601566d3627c5f489f13b)
- [Banyan](https://github.com/nhshah15/banyan/tree/7bb5d385cb2b48f49ab01d0c9a47b9d404e2ecc3)
- Purab Seth et al., “Task diversity produces systematic transfer but inhibits continual reinforcement learning,” arXiv:2606.00880v2 (2026), supplied with this task.

See `THIRD_PARTY_NOTICES.md` for licenses and the exact code-level provenance.
