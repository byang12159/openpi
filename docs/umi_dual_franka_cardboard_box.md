# UMI dual-Franka cardboard-box fine-tuning and deployment

This guide fine-tunes the JAX π0.5 policy on dual-Franka UMI demonstrations and deploys it on a dual-Franka setup with two UMI fisheye cameras.
It covers two tasks and nine registered configs.
For the cardboard-box task ([`byang11259/cardboard_box_tcp_curated`](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated)) it supports two episode-construction choices: the recommended choice materializes one logical episode per physical box, and an explicit long-episode ablation trains directly on the original source episodes.
Each choice has a primary query-anchor-relative representation and a controlled absolute-action baseline.
Two cross-embodiment state modes, `gripper_only` and `relative_history`, remove absolute pose from the policy state entirely; they are applied to the cardboard-box exports and to the stack-cubes task (see [Stack-cubes task](#stack-cubes-task)).

> [!WARNING]
> A source dataset episode can contain **multiple physical cardboard boxes**.
> A source episode is therefore a recording/session boundary, not necessarily a
> valid task-episode boundary. For boundary-safe training, split every physical
> box into a logical episode or use a sampler and validity mask that reject
> chunks crossing a logical boundary. The stock LeRobot loader and
> `compute_norm_stats.py` do not consume a sidecar boundary manifest by
> themselves. The `_long_episode` configs intentionally preserve this behavior
> as an ablation: their chunks stay inside a source episode but can cross from
> one physical box instance into the next.

## Registered configs

Every value in this table is read from the `TrainConfig` entries in `src/openpi/training/config.py`.

| Config | Dataset repo | Episode path and purpose | `state_mode` | Actions | `image_crop` | Steps x batch |
| --- | --- | --- | --- | --- | --- | --- |
| `pi05_umi_dual_franka_cardboard_box_relative` | `local/cardboard_box_tcp_curated_logical_train` | logical box (recommended), relative primary | `full` | relative | none | 5,000 x 32 |
| `pi05_umi_dual_franka_cardboard_box_absolute` | `local/cardboard_box_tcp_curated_logical_train` | logical box (recommended), absolute baseline | `full` | absolute | none | 5,000 x 32 |
| `pi05_umi_dual_franka_cardboard_box_relative_long_episode` | `local/cardboard_box_tcp_curated_x264` | original source episodes (ablation), relative primary | `full` | relative | none | 5,000 x 32 |
| `pi05_umi_dual_franka_cardboard_box_absolute_long_episode` | `local/cardboard_box_tcp_curated_x264` | original source episodes (ablation), absolute baseline | `full` | absolute | none | 5,000 x 32 |
| `pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode` | `local/cardboard_box_tcp_curated_10s_x264` | 10 s resegmented source episodes, gripper-only cross-embodiment | `gripper_only` | relative | 224 | 10,000 x 128 |
| `pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode` | `local/cardboard_box_tcp_curated_x264` | original source episodes, gripper-only with the inscribed-square crop | `gripper_only` | relative | 272 | 10,000 x 128 |
| `pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54` | `local/cardboard_box_tcp_vid7to54_x264` | complete vid7to54 session-length episodes, gripper-only | `gripper_only` | relative | 224 | 10,000 x 128 |
| `pi05_umi_dual_franka_stack_cubes_relative_gripper_only` | `local/stack_cubes_tcp_x264` | stack-cubes task, gripper-only recipe | `gripper_only` | relative | 224 | 10,000 x 128 |
| `pi05_umi_dual_franka_stack_cubes_relative_history` | `local/stack_cubes_takes_x264` | stack-cubes short takes, relative-history state (`observation_horizon=2`) | `relative_history` | relative | 224 | 6,000 x 64 |

> [!NOTE]
> The full-state `_long_episode` pair and the `_crop272_long_episode` config point at `local/cardboard_box_tcp_curated_x264`, a local derived copy of the curated source dataset whose parquet/meta files are byte-identical and whose 36 videos are re-encoded near-losslessly (libx264, `-crf 14 -g 15 -bf 0`).
> The original HEVC exports use ~250-frame GOPs with B-frames; lerobot's torchcodec decode path (`seek_mode="approximate"`) returns wrong frames near GOP tails on such streams and fails its 1e-4 s timestamp-tolerance check.
> Dense keyframes and no B-frames make index-to-pts mapping exact and random access fast.
> Every other `local/...` repo above is a derived copy as well; see [Derived datasets](#derived-datasets) for the sources, the re-encode tool, and the `REENCODE_PROVENANCE.md` each re-encoded directory carries.

All nine configs:

- start from `gs://openpi-assets/checkpoints/pi05_base/params`;
- use the same two cameras, 6D rotation encoding, prompt plumbing, and
  horizon;
- consume the canonical post-repack keys `observation/state`,
  `observation/left_head`, `observation/right_head`, `actions`, and `prompt`;
- use an action horizon of 50 at the dataset's nominal 29.97 Hz, about
  1.67 seconds of predicted motion; and
- require their **own fresh normalization statistics**.

For either episode path, the full-state relative and absolute configs use identical absolute state20.
The relative action is fixed-query-anchor true-SE(3) action20; the baseline action is absolute action20.
The gripper-only configs keep the relative action20 but reduce the policy state to the 2-D absolute gripper vector (see [Gripper-only-state cross-embodiment variant](#gripper-only-state-cross-embodiment-variant)).
The relative-history config also keeps the relative action20 and replaces the state with the previous frame's pose expressed in the current TCP frame (see [Relative-history-state cross-embodiment variant](#relative-history-state-cross-embodiment-variant)).
Configs must not share norm stats or checkpoints even when their representation is the same.

The registered training recipes differ per config:

| Config | `batch_size` | `fsdp_devices` | `num_workers` | `num_train_steps` | `save_interval` |
| --- | --- | --- | --- | --- | --- |
| `pi05_umi_dual_franka_cardboard_box_relative` | 32 | 1 | 8 | 5,000 | 5,000 |
| `pi05_umi_dual_franka_cardboard_box_absolute` | 32 | 1 | 8 | 5,000 | 5,000 |
| `pi05_umi_dual_franka_cardboard_box_relative_long_episode` | 32 | 1 | 8 | 5,000 | 5,000 |
| `pi05_umi_dual_franka_cardboard_box_absolute_long_episode` | 32 | 1 | 8 | 5,000 | 5,000 |
| `pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode` | 128 | 8 | 8 | 10,000 | 5,000 |
| `pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode` | 128 | 8 | 8 | 10,000 | 5,000 |
| `pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54` | 128 | 8 | 32 | 10,000 | 5,000 |
| `pi05_umi_dual_franka_stack_cubes_relative_gripper_only` | 128 | 8 | 8 | 10,000 | 5,000 |
| `pi05_umi_dual_franka_stack_cubes_relative_history` | 64 | 4 | 16 | 6,000 | 2,000 |

The four full-state configs are portable one-H200 defaults: full fine-tuning on a single device.
The gripper-only configs register an 8-GPU recipe and the relative-history config a 4-GPU recipe.
`scripts/train.py` rejects a `batch_size` that the visible device count does not divide, and the mesh rejects a device count that `fsdp_devices` does not divide, so override `--batch-size` and `--fsdp-devices` together when running a multi-GPU recipe on a different node.
The vid7to54 config uses 32 data workers because per-sample video decode from its 46 session-length episodes is the bottleneck.
The relative-history config also pins `keep_period=2_000` to its save interval: orbax keeps only the most recent checkpoint plus those with `step % keep_period == 0`, so the default 5,000 would silently prune the 2k and 4k checkpoints.
All configs keep the upstream defaults `./assets` and `./checkpoints`, resolved from the repository root; override them per run with `--assets-base-dir` and `--checkpoint-base-dir` when training on a machine that keeps them elsewhere.

During training, the adapter maps each LeRobot row's `task` string to the model `prompt`, so a reviewed logical-episode manifest can supply a fold-only label.
If no prompt is present at inference, the cardboard-box configs fall back to the full-task prompt:

```text
Assemble the cardboard box and put it into the bin
```

The two stack-cubes configs fall back to their task's prompt instead:

```text
Stack the cubes into a tower
```

Do not change only the prompt to create a fold-only dataset. See
[Task labeling and temporal crops](#task-labeling-and-temporal-crops).

## Raw dataset contract

### Raw state and action: 16 values

The dataset stores each state as:

```text
[
  left_x, left_y, left_z,
  left_qx, left_qy, left_qz, left_qw,
  left_gripper,
  right_x, right_y, right_z,
  right_qx, right_qy, right_qz, right_qw,
  right_gripper,
]
```

| Slice | Meaning |
| --- | --- |
| `0:3` | left chopstick-tip TCP position |
| `3:7` | left TCP quaternion in `xyzw` order |
| `7` | left gripper, absolute, in `[0, 1]`, with `1 = open` |
| `8:11` | right chopstick-tip TCP position |
| `11:15` | right TCP quaternion in `xyzw` order |
| `15` | right gripper, absolute, in `[0, 1]`, with `1 = open` |

The raw action stored at index `t` is the demonstrated absolute next TCP
target:

```text
raw_action[t] == raw_state[t + 1]
```

This one-step shift is part of the dataset contract. The recommended
logical-episode materialization prevents it from crossing a physical-box
boundary. The explicit long-episode ablation retains source-episode
transitions, including any transition between two physical boxes.

Additional invariants:

- arm order is always left, then right;
- poses are absolute in the episode ArUco marker/world frame;
- the dataset's `-90°` yaw convention is already baked into the poses—do not
  apply it again;
- the end effector is the chopstick grasp-tip TCP, not the Franka flange;
- gripper values remain absolute in all model configurations; and
- the custom dataset convention `1 = open` is intentionally preserved. Do not
  replace it with a generic robot convention.

### No measured joints

The dataset does **not** contain `joint_pos`, `joint_vel`, or `joint_target`.
None of the configs invent them. IK solutions are controller-side commands or
pseudo-labels, not measured demonstration state, and must not be introduced as
ground truth.

The robot runtime obtains a live Cartesian TCP pose from Franka FK plus
calibration and the fixed flange-to-tip transform. Joint states may be used by
the controller, collision checker, and FK implementation, but they are not
policy state dimensions.

### Units are a hardware gate

Confirm the dataset translation units empirically before commanding either
robot. Check representative displacements against a known physical dimension,
record the result in the experiment manifest, and test round trips in those
units. Do not infer metres versus millimetres from value magnitude alone.
Normalization can make an incorrect unit scale look numerically plausible
while producing dangerous commands.

## Choose the episode path

### Stock OpenPI/LeRobot long-source semantics

The four `_long_episode` configs train on source-derived episodes that are not box-segmented, so a LeRobot chunk can still cross a physical-box boundary.
They do not read the Hugging Face download itself: the full-state pair and the `_crop272_long_episode` config point at `local/cardboard_box_tcp_curated_x264`, a mirror of `byang11259/cardboard_box_tcp_curated` whose parquet and meta files are byte-identical and whose videos are re-encoded, and the `_gripper_only_long_episode` config points at `local/cardboard_box_tcp_curated_10s_x264`, a 10 s time-resegmentation of the same source that is likewise not box-segmented (see [Derived datasets](#derived-datasets)).
All four use the stock OpenPI LeRobot action-chunk path ([OpenPI loader](../src/openpi/training/data_loader.py)):

```text
delta_timestamps["action"] =
    [0 / fps, 1 / fps, ..., 49 / fps]
```

For a query at source row `t`, LeRobot therefore requests the singular
`action` column at rows `t ... t + 49`. Because the dataset contract is
`action[t] = state[t + 1]`, those entries target states
`t + 1 ... t + 50`. At 29.97 Hz, the requested action samples span
`49 / 29.97 ≈ 1.64 s`, while the last physical target is approximately
`50 / 29.97 ≈ 1.67 s` after the query state.

LeRobot clips each requested index to the current **source episode** and marks
out-of-range values with `action_is_pad`
([pinned LeRobot implementation](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py#L665-L678)).
Thus:

- a chunk never enters a different source episode;
- near a source-episode end, the final stored source action is repeated;
- source-internal physical-box boundaries are invisible to the loader;
- a query before such a boundary can contain actions from the next box; and
- neither the current UMI transform nor `compute_norm_stats.py` consumes a
  physical-box sidecar mask or the padding flag.

For the long relative config, every one of those future absolute targets is
converted against the single state at query row `t`. A reset into the next
physical box can therefore become a large fixed-anchor relative target. This
is the intended long-episode ablation behavior, not a boundary-safe dataset.
Its norm stats and loss include those samples.

Use this path to test the episode-local, fixed-horizon hypothesis. Do not
describe it as one-box-per-episode training, leakage-free validation, or the
recommended hardware model. The logical path remains the default for
controlled train/validation/test splits and deployment claims.

### Relationship to the Hy-Embodied UMI loader

The long path is **inspired by**, but does not copy, the pinned Hy-Embodied UMI
target builder:

- Hy configures a 50-step horizon and downsamples the 30 Hz stream by three,
  producing a nominal 10 Hz, roughly five-second chunk
  ([config](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/config/dataset/umi_lance.yaml#L28-L50)).
- Hy limits sampling by dataset episode bounds
  ([episode bounds](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L157-L165))
  and clamps `c_id + 3k` to the last row, without an internal physical-box
  boundary check
  ([future index and clamp](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L520-L545),
  [assembled indices](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L590-L630)).
- Hy takes future `xyz + quaternion` from future `observation.state` rows and
  overwrites only gripper values from `action`
  ([target construction](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L525-L534)).
  The cardboard-box dataset already provides full absolute action16 targets,
  so OpenPI correctly chunks the singular `action` column instead of
  reproducing that builder.
- Hy's first future pose is the current state (`k = 0`), so its first relative
  pose is identity. OpenPI's first entry is `action[t] = state[t + 1]`, so its
  first relative pose is current-to-next. Hy also takes the prompt from the
  anchor row
  ([prompt behavior](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L580-L588)).

Both use absolute dual-TCP state20 and a fixed-anchor relative action20
concept, but the temporal sampling and target sources differ. See the pinned
[Hy transform implementation](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/utils/transform_utils.py#L16-L123).

## Recommended path: split multi-box recordings into logical episodes

Use these terms consistently:

- **collection session**: one acquisition session, potentially spanning
  several source files;
- **source episode**: the episode/file boundary in the published dataset;
- **box instance**: one physical cardboard box manipulated in that recording;
- **logical episode**: the contiguous frames for one box instance and one
  declared task variant.

All logical episodes derived from the same source episode or collection
session are correlated. Put the entire group into exactly one of train,
validation, or test. A random split by logical episode leaks the same camera
placement, operator, background, calibration, and adjacent motion into
validation.

### Annotation manifest

Copy the provided template and replace every illustrative boundary after
frame-by-frame review:

```bash
cp configs/cardboard_box_segments.example.json \
  configs/cardboard_box_segments.json
```

The splitter accepts this versioned JSON schema. `end_frame` is **exclusive**,
so every interval is `[start_frame, end_frame)`.

```json
{
  "version": 1,
  "source_repo_id": "byang11259/cardboard_box_tcp_curated",
  "segments": [
    {
      "source_episode": 0,
      "box_index": 0,
      "start_frame": 120,
      "end_frame": 2540,
      "split": "train",
      "task": "Assemble the cardboard box and put it into the bin"
    },
    {
      "source_episode": 0,
      "box_index": 1,
      "start_frame": 2710,
      "end_frame": 5160,
      "split": "train",
      "task": "Assemble the cardboard box and put it into the bin"
    },
    {
      "source_episode": 1,
      "box_index": 0,
      "start_frame": 90,
      "end_frame": 2480,
      "split": "validation",
      "task": "Assemble the cardboard box and put it into the bin"
    }
  ]
}
```

These numbers are examples, not reviewed annotations for the published data.
Keep richer annotation notes—collection-session ID, fold-complete frame,
bin-motion-start frame, annotator, calibration ID, camera serials, checksums,
and exclusions—in a versioned companion record. The materializer enforces one
split per `source_episode`. If one collection session spans several source
episodes, the reviewer must additionally assign all of those source episodes
to the same split; the current JSON schema cannot infer that relationship.

### Recommended preprocessing workflow

1. Inventory collection sessions and source episodes. Recover a stable
   `collection_session_id`; if none exists, conservatively group by source
   episode.
2. For every physical box, annotate its first task frame, first
   fully-folded frame, first bin-placement motion frame, and exclusive end.
3. Validate that intervals are ordered, non-overlapping, in range, and
   synchronized across both cameras and both arms.
4. Assign train/validation/test by collection-session group **before**
   materializing logical episodes. Every sibling box from a group inherits the
   same split.
5. Run a full-manifest dry run, then materialize train, validation, and test
   into separate local LeRobot repositories.
6. The splitter validates `action[t] = state[t + 1]` for every nonterminal
   logical frame and replaces the final action with an in-episode absolute
   no-op: `action[-1] = observation.state[-1]`.
7. It rewrites episode/frame/global indices and timestamps, trims both videos
   at the exact annotated frames, decodes them to verify frame count, and
   records source provenance in `meta/logical_segments.jsonl`.
   Numeric episode stats are recomputed. Cropped-video stats are omitted
   instead of incorrectly copying full-source-episode camera stats; this
   adapter's training normalization uses transformed state/actions only.
8. Point **both** registered configs to the same derived dataset revision.
   Record its immutable repository revision or content hash.
9. Only now compute the separate relative and absolute norm stats, then train.
10. Audit counts by split, source episode, session, task variant, and box
    instance. Manually inspect the first and last chunk of every logical
    episode.

Assuming the source LeRobot v2.1 dataset is under `HF_LEROBOT_HOME`, first
validate the **complete** manifest. The `--split train` argument selects a
nonempty output split, but `--dry-run` validates every segment and grouped
source-episode split in the manifest:

```bash
uv run python scripts/split_cardboard_box_lerobot_v21.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_train" \
  --manifest configs/cardboard_box_segments.json \
  --split train \
  --action-horizon 50 \
  --dry-run
```

Then materialize each declared split into an absent or empty destination:

```bash
uv run python scripts/split_cardboard_box_lerobot_v21.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_train" \
  --manifest configs/cardboard_box_segments.json \
  --split train \
  --action-horizon 50

uv run python scripts/split_cardboard_box_lerobot_v21.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_validation" \
  --manifest configs/cardboard_box_segments.json \
  --split validation \
  --action-horizon 50

uv run python scripts/split_cardboard_box_lerobot_v21.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_logical_test" \
  --manifest configs/cardboard_box_segments.json \
  --split test \
  --action-horizon 50
```

Run only splits present in the reviewed manifest. Each logical segment must
contain at least `action_horizon + 1`, currently 51, frames. The registered
recommended training configs already point to
`local/cardboard_box_tcp_curated_logical_train`.

Materializing logical episodes is preferred because the normal LeRobot
episode-aware sampler can then enforce boundaries. If data must remain in the
original source episodes **and boundary-safe training is still the goal**, a
manifest alone is insufficient: the dataset sampler, chunk builder,
normalization pass, and training loss must all consume the same
logical-boundary validity mask. In particular:

- reject any query/action pair crossing from one box instance to another;
- reject any 50-step chunk that reaches across a boundary;
- mask any deliberately padded terminal waypoint in both norm stats and loss;
- keep image, state, action, timestamp, and mask indices synchronized; and
- test that the online loader and offline preprocessing select identical
  frames.

The materialized route does not create a separate terminal-loss mask. LeRobot
may clamp a terminal chunk to the final in-episode no-op, but it can no longer
cross into the next physical box. If an experiment requires excluding repeated
terminal no-ops from norm stats or loss, implement and test that mask in both
paths; do not assume the sidecar manifest is consumed automatically.

The non-suffixed logical-config commands later in this guide assume one of
these boundary-safe preparations. The explicitly labeled `_long_episode`
commands intentionally skip them and must be reported as an unsegmented-source
ablation.

### Task labeling and temporal crops

For the full task, retain the folding and bin-placement frames and use the
dataset prompt exactly:

```text
Assemble the cardboard box and put it into the bin
```

For a fold-only experiment:

1. make a separate logical-episode view;
2. end it at the completed-fold boundary and before any bin-placement frame
   (normally the manifest uses
   `end_frame = fold_complete_frame + 1`, because `end_frame` is exclusive);
3. let the splitter validate the shifted actions and install the final no-op;
   and
4. relabel it consistently, for example:

```text
Fold the cardboard box
```

Merely replacing the full-task string while retaining motions that put the box
in the bin creates contradictory supervision. Conversely, cropping the motion
without changing the prompt removes the prompt's stated outcome. Keep one
prompt and one temporal definition per task variant, and do not mix variants
silently in the same evaluation.

The splitter writes this per-episode task label and the data adapter maps it to
the training prompt. At deployment, send that exact same fold-only prompt on
every request. For protection against a client accidentally omitting it,
register a dedicated fold-only config whose `default_prompt` is also
`Fold the cardboard box`.

The registered `_long_episode` configs intentionally retain the original
full-task source episodes and task labels. They are not fold-only configs.
Changing only a long config's prompt is contradictory.

## Stack-cubes task

The second task uses two public exports of dual-Franka UMI stacking demonstrations with the same raw 16-D state/action contract and the same two `front_equi` cameras.
Both carry a single task string, which the two stack-cubes configs also register as their `default_prompt`:

```text
Stack the cubes into a tower
```

| Source | Episodes | Frames | Native video frame | Config |
| --- | --- | --- | --- | --- |
| [`byang11259/stack_cubes_tcp`](https://huggingface.co/datasets/byang11259/stack_cubes_tcp) | 46 | 33,276 | 1920 x 1920 | `pi05_umi_dual_franka_stack_cubes_relative_gripper_only` |
| [`byang11259/stack_cubes_takes`](https://huggingface.co/datasets/byang11259/stack_cubes_takes) | 43 | 10,664 | 1920 x 1920 | `pi05_umi_dual_franka_stack_cubes_relative_history` |

Both exports publish 1920 px video frames, whereas the cardboard-box curated exports that the crop rules below were written for are 384 px.
The derived training copies `local/stack_cubes_tcp_x264` and `local/stack_cubes_takes_x264` therefore downscale every video to 384 px (`--scale 384`) before the same x264 GOP-15, no-B-frame re-encode used for the cardboard-box mirrors, so `image_crop=224` selects the same fraction of the field of view and the same ~1.71x magnification as on the cardboard-box configs.
The downscale happens once, in the derived dataset; the input transform still receives 384 x 384 frames, and deployment clients for these configs must downscale live frames to 384 x 384 the same way before sending them, uncropped.
See [Derived datasets](#derived-datasets) for the exact command.

`pi05_umi_dual_franka_stack_cubes_relative_gripper_only` is the cardboard-box gripper-only recipe applied to the new task (2-D gripper state, fixed-anchor relative action20, 224 px crop, batch 128 on 8 GPUs for 10,000 steps) plus raw rot6d through a neutralized stats file, a step the cardboard-box configs do not record.

`pi05_umi_dual_franka_stack_cubes_relative_history` trains on the smaller `stack_cubes_takes` export (43 takes of roughly 6.5 to 11 s each, about 3x fewer frames than `stack_cubes_tcp`).
The 10,000 x 128 recipe would have made about 120 passes over it, so the config is retuned to 6,000 steps at batch 64 on 4 GPUs (about 36 passes) with checkpoints every 2,000 steps and `keep_period` pinned to the same value so orbax retains the 2k and 4k checkpoints.
Its state mode is described under [Relative-history-state cross-embodiment variant](#relative-history-state-cross-embodiment-variant).

The repository has no splitter or manifest for this task.
The exports are used as published, and LeRobot chunking stays inside each recorded episode.

## Model representations

Define `T^A_B` as the pose of frame `B` expressed in frame `A`. Both configs
first convert `xyzw` quaternions to a paired 6D rotation representation.

### Absolute policy state: 20 values

The state is identical for the primary and baseline configs:

```text
[
  left_abs_xyz[3], left_abs_rot6d[6], left_current_gripper[1],
  right_abs_xyz[3], right_abs_rot6d[6], right_current_gripper[1],
]
```

| Slice | Meaning |
| --- | --- |
| `0:3` | left TCP absolute translation in marker/world |
| `3:9` | left TCP absolute rotation, 6D |
| `9` | left current absolute gripper, `1 = open` |
| `10:13` | right TCP absolute translation in marker/world |
| `13:19` | right TCP absolute rotation, 6D |
| `19` | right current absolute gripper, `1 = open` |

### Primary relative action: 20 values per waypoint

For query time `t`, arm `r`, and future waypoint `k`, the raw dataset has
already shifted the first action target to `t + 1`. The primary config computes

```text
T_rel[r, t, k] = inverse(T_world_tcp[r, t]) @ T_world_tcp[r, t + k + 1]
```

and encodes:

```text
[
  left_rel_xyz[3], left_rel_rot6d[6], left_future_gripper_abs[1],
  right_rel_xyz[3], right_rel_rot6d[6], right_future_gripper_abs[1],
]
```

The translation is expressed in the current TCP frame because this is a true
SE(3) relative transform, not component-wise position/quaternion subtraction.
Each arm has its own synchronized query-time anchor. Every waypoint in a
predicted chunk uses the **same** saved anchor:

```text
T_world_tcp_pred[r, k] = T_world_tcp_query[r] @ T_rel_pred[r, k]
```

Do not compose waypoint `k` against the robot pose reached at waypoint
`k - 1`, and do not replace the saved anchor with a moving live pose during
open-loop prefix execution. Capture new anchors only when the policy is
queried again.

### Absolute-action baseline: 20 values per waypoint

The baseline encodes:

```text
[
  left_abs_xyz[3], left_abs_rot6d[6], left_future_gripper_abs[1],
  right_abs_xyz[3], right_abs_rot6d[6], right_future_gripper_abs[1],
]
```

Its outputs are already world-frame TCP targets. Within either the logical
pair or the long pair, it uses the same state, rotation representation,
cameras, prompt, episode construction, horizon, and hyperparameters as the
relative primary so that pairwise comparison isolates action relativity.

### Gripper-only-state cross-embodiment variant

> [!NOTE]
> Four configs use `state_mode="gripper_only"`.
> `pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode` trains on `local/cardboard_box_tcp_curated_10s_x264` (the finer-segmented 107-episode 10 s dataset, x264 re-encode, episodes 100-103 relabeled from a placeholder to the standard task string) with `image_crop=224` (see [Cropped fisheye views](#cropped-fisheye-views-image_crop)).
> The `_crop272_long_episode` sibling keeps the original 18-episode dataset with the 272 px crop.
> `pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54` applies the 224 px recipe to the complete vid7to54 export.
> `pi05_umi_dual_franka_stack_cubes_relative_gripper_only` applies it to the stack-cubes task (see [Stack-cubes task](#stack-cubes-task)).

The gripper-only configs keep the fixed-anchor relative action20 exactly as the primary relative config but replace the 20-D absolute state with the 2-D absolute gripper vector:

```text
[left_current_gripper, right_current_gripper]
```

Rationale: with a single observation timestep, pose-relative proprioception is
identically zero, so the absolute pose is the only pose signal — and it is
scene/marker-specific. Dropping it forces vision-based control in the spirit
of the original UMI recipe and keeps the policy embodiment-agnostic. The
gripper values stay absolute in `[0, 1]` with `1 = open` and are already
embodiment-normalized.

Differences from the other configs:

- The policy never receives or emits absolute pose; inter-arm geometry must
  be inferred from the two fisheye views alone. The full-state relative and
  absolute configs remain available as controlled comparisons.
- Its `state` norm stats are 2-D; the relative-action stats follow the same
  distribution as the primary relative config. `state_mode="gripper_only"`
  requires `action_representation="relative"`; combining it with the absolute
  baseline is rejected.
- **The WebSocket response is a relative chunk, not absolute targets**: per
  arm `[rel_xyz(3), rel_quat_xyzw(4), gripper_abs(1)]`, all 50 waypoints
  expressed in that arm's query-time TCP frame. The robot client must compose
  `T_world_tcp_pred[k] = T_world_tcp_query[r] @ T_rel_pred[r, k]` with its own
  saved per-arm anchors. The same fixed-anchor rules apply: never compose
  waypoint `k` against the pose reached at waypoint `k - 1`, and never refresh
  the anchor during open-loop prefix execution.

### Relative-history-state cross-embodiment variant

`pi05_umi_dual_franka_stack_cubes_relative_history` uses `state_mode="relative_history"` with `observation_horizon=2`.
Like the gripper-only variant it carries no absolute pose, so it stays cross-embodiment and the server keeps returning relative chunks; unlike it, the state exposes the recent motion that a single-frame observation cannot provide.
The policy state is 20-D with the same per-arm layout as the relative action:

```text
[
  left_rel_xyz[3], left_rel_rot6d[6], left_current_gripper[1],
  right_rel_xyz[3], right_rel_rot6d[6], right_current_gripper[1],
]
```

For an observation window `s[t - H + 1], ..., s[t]` (oldest first, current frame last), the history pose is the window's first frame expressed in the **current** TCP frame of the same arm:

```text
T_rel_state[r, t] = inverse(T_world_tcp[r, t]) @ T_world_tcp[r, t - H + 1]
```

With `observation_horizon=2` that is the previous frame.
`raw16_history_to_relative_state20` reuses the relative-action encoder for the pose math, so the true-SE(3) convention and the row-major rot6d encoding cannot drift apart from the action targets; the gripper slots are overwritten with the current absolute values and are never differenced.

Data loading:

- `LeRobotUmiDualFrankaDataConfig(observation_horizon=2)` sets `DataConfig.state_sequence_keys=("observation.state",)` and `DataConfig.observation_horizon=2`; `state_mode="relative_history"` with `observation_horizon < 2` is rejected, because one frame carries no history.
- `create_torch_dataset` in `src/openpi/training/data_loader.py` then asks LeRobot for the window with negative delta timestamps, `[(t - H + 1) / fps for t in range(H)]`, which is `[-1 / fps, 0]` for `H = 2`, so each sample's `observation.state` arrives as an `(H, 16)` array oldest first.
- LeRobot clamps requests that reach before the episode start, so the history frame there equals the current frame and the relative pose degrades to identity ("no motion"), the same value a bare `(16,)` state produces.
- The relative-action anchor is always the window's **last** frame, the current state; `_build_inputs` takes it from that row in every state mode, and a policy test pins this, because anchoring on the history frame would silently shift every target.

Serving and normalization:

- The client sends `observation/state` as an `(observation_horizon, 16)` window, oldest first; a bare `(16,)` state is accepted but yields the identity history, so the policy then sees only the grippers.
- The output transform is `UmiDualFrankaRelativeGripperOnlyOutputs`, which `config.py` selects for every `state_mode` other than `"full"`: the response is a `[50, 16]` relative chunk that the client composes against the last frame of the window it sent.
- `state_mode="relative_history"` requires `action_representation="relative"`; combining it with the absolute baseline is rejected.
- The `state` norm stats are 20-D. `scripts/neutralize_rot6d_norm_stats.py` detects this state mode and neutralizes the state's rot6d dims together with the action's, so rot6d stays raw in both (see [Raw-rot6d variant of the quantile stats](#raw-rot6d-variant-of-the-quantile-stats)).

### Rotation and internal padding

The implemented 6D convention is the first two **rows** of the rotation matrix,
flattened row-major. Its paired decoder orthonormalizes those rows and projects
the result to SO(3). Use the repository implementation in both offline
preprocessing and online serving; do not mix it with a column-based decoder or
transpose the decoded matrix. Tests must cover identity, random rotations,
near-degenerate 6D vectors, and quaternion sign equivalence (`q` and `-q`).

The physical action dimension is always 20, and the full-state and relative-history configs use a 20-D state (the gripper-only state is 2-D).
π0.5 zero-pads state and
actions to its native 32-dimensional model tensors internally and slices
predictions back to 20 before the robot-specific output transform. The padded
values have no robot meaning and must never reach a controller or enter the
physical norm stats.

## Cameras and visual preprocessing

The two dataset streams are:

| Dataset key | Policy slot | Source |
| --- | --- | --- |
| `left_head` | `left_wrist_0_rgb` | left Insta360 X5 `front_equi` |
| `right_head` | `right_wrist_0_rgb` | right Insta360 X5 `front_equi` |

### Cropped fisheye views (`image_crop`)

The `image_crop` option on `LeRobotUmiDualFrankaDataConfig` applies a centered
square crop to **both** camera views at load time, inside the shared input
transform (`center_crop_image` in
`src/openpi/policies/umi_dual_franka_policy.py`). The dataset videos are never
modified; the review clips under the derived dataset's `crop_review/` folder
are visualization artifacts only.

Per-sample pipeline order:

```text
decode 384x384 frame
  -> parse/validate (CHW->HWC, dtype/range, finiteness)
  -> centered crop: top = left = (384 - side) // 2
     (side 272 -> rows/cols 56..327; side 224 -> rows/cols 80..303)
  -> masked base_0_rgb placeholder created at the cropped shape
  -> ResizeImages(224, 224) model transform
  -> pi0.5 image tensor
```

Because the crop then upsamples into the fixed 224 x 224 model input, it
*increases* effective workspace resolution: 272 px keeps the largest square
inscribed in the fisheye circle (kills the dead black corners, ~1.41x
magnification); 224 px equals the model's native input size, so the resize
becomes a no-op — zero resampling, pixel-for-pixel sharp — with ~1.71x
magnification, and also removes most wall/operator periphery — pixels that
show the human demonstrator during collection but a robot arm at deployment,
a guaranteed train/deploy mismatch.

Rules:

- The crop runs identically at training, norm-stats computation, and serving.
  Deployment clients keep sending full 384 x 384 frames; the server crops.
  **Never pre-crop client-side** — that would crop twice.
- Side lengths outside `(0, 384]` are rejected. Odd remainders round the
  offset down (a 225 px crop sits half a pixel off-center).
- Changing the crop is a new experiment: register a new config name, compute
  its own fresh norm-stat tree, and retrain. Checkpoints are never valid
  across different crops (or between cropped and uncropped configs).
- State and action processing are completely unaffected by the crop.

The two source cameras are the same model and projection.
Their images are `front_equi` fisheye views at nominal 29.97 Hz, 384 x 384 in the cardboard-box curated exports and in the downscaled stack-cubes copies (see [Stack-cubes task](#stack-cubes-task)).
The π0.5 base/exterior slot is filled with a placeholder and marked absent in the image mask; it is not a third real camera.

Deployment must use the same camera model and projection as the demonstrations:

```text
raw fisheye -> same fixed crop/resize/pad -> π0.5 image tensor
```

Do **not** rectify the deployment fisheye images to pinhole unless the entire
training dataset is regenerated with that exact rectification. Do not rectify
one side only. Before training and again before robot execution, compare:

- camera model, serial/configuration, `front_equi` projection, and distortion;
- native resolution, aspect ratio, crop, resize, interpolation, and padding;
- left/right ordering and timestamp synchronization;
- mount extrinsics, roll/pitch/yaw, tool visibility, and occlusions; and
- field of view and approximate pixels per known workspace distance.

Log a transformed training pair and a transformed live pair side by side. The
same object should occupy comparable pixels with matching handedness and tool
placement.

## World, base, flange, and TCP transforms

For each arm `r`, FK supplies the flange pose in its own base,
`T^B_r_F_r`. Calibration supplies `T^W_B_r`, and the measured fixed tool
transform supplies `T^F_r_TCP_r`. The live state sent to the policy is:

```text
T_world_tcp[r] =
    T_world_base[r] @ T_base_flange[r] @ T_flange_chopstick_tip[r]
```

The complete command path is:

```text
Franka joints
  -> FK in each robot base
  -> fixed flange-to-chopstick-tip transform
  -> calibrated common marker/world TCP poses
  -> policy state (state_mode "full": absolute state20;
     "gripper_only": 2-D gripper state;
     "relative_history": 20-D previous frame in the current TCP frame)
  -> π0.5 action20
  -> (state_mode "full", relative actions) policy output transform composes
     the saved query-time TCP anchors server-side
  -> (state_mode "gripper_only" or "relative_history") the server returns the
     relative chunk and the robot client composes its own saved query-time
     TCP anchors
  -> absolute world TCP targets
  -> transform into each Franka base
  -> convert TCP target to flange target if required by the controller
  -> Cartesian controller or online IK
```

For a flange-target controller:

```text
T_world_flange_des =
    T_world_tcp_des @ inverse(T_flange_tcp)

T_base_flange_des =
    inverse(T_world_base) @ T_world_flange_des
```

For a controller that natively accepts the calibrated TCP:

```text
T_base_tcp_des = inverse(T_world_base) @ T_world_tcp_des
```

Never feed a chopstick-tip target to a flange controller without the fixed tool
transform. Keep separate calibration records for the left and right robot
bases, and stop if calibration, FK, timestamps, or camera frames are stale.

## Blank H200 machine setup

The upstream openpi project is tested on Ubuntu 22.04.
Full JAX fine-tuning requires more than 70 GB of GPU memory; the full-state configs' portable defaults target one H200 with `fsdp_devices=1` and batch size 32, while the gripper-only and relative-history recipes expect 8 or 4 GPUs (see the recipe table under [Registered configs](#registered-configs)).
Expose one H200 with `CUDA_VISIBLE_DEVICES=0` in the single-device commands below.
The current JAX script supports multi-device meshes on one node, but not multi-node training.

Start with a working NVIDIA driver:

```bash
nvidia-smi
```

System CUDA libraries are not required by openpi; the locked `jax[cuda12]`
environment installs user-space CUDA dependencies. A compatible NVIDIA driver
is still required.

On a blank Ubuntu machine:

```bash
sudo apt-get update
sudo apt-get install -y git git-lfs curl ffmpeg
git lfs install

curl -LsSf https://astral.sh/uv/install.sh | sh
source "$HOME/.local/bin/env"

git clone --recurse-submodules https://github.com/byang12159/openpi.git
cd openpi
git checkout umi-dual-franka
git submodule update --init --recursive

GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Clone or move the repository onto local SSD with enough space for the datasets, the π0.5 base checkpoint, one independent norm-stat tree per config, and experiment checkpoints.
The configs use `./assets` and `./checkpoints`, resolved from the repository working directory.
Always run commands from the repository root.

Set the following from the repository root.
`OPENPI_ROOT` records that root for commands run from other directories, such as the client install in the deployment section.
The upstream checkpoint downloader caches in `~/.cache/openpi` by default; a large local SSD cache can be selected before any download:

```bash
export OPENPI_ROOT=$(pwd)
export OPENPI_DATA_HOME=/mnt/localssd/openpi-cache
export HF_HOME=/mnt/localssd/huggingface
export HF_LEROBOT_HOME=/mnt/localssd/lerobot_home

mkdir -p "$OPENPI_DATA_HOME" "$HF_HOME" "$HF_LEROBOT_HOME"
```

If access requires authentication, run `uv run huggingface-cli login`.
Download the pinned source revision into the layout expected by the splitter:

```bash
uv run huggingface-cli download byang11259/cardboard_box_tcp_curated \
  --repo-type dataset \
  --revision a366d2b92723d795b6b93e6303f57708e87b63ea \
  --local-dir "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated"
```

For the logical path, annotate next, then run the dry-run and materialization commands above and record content hashes for the local train/validation/test datasets.
The long path does not run the splitter; it re-encodes this pinned download into `local/cardboard_box_tcp_curated_x264` with the tool described under [Derived datasets](#derived-datasets), because no config reads the HEVC download directly.

Verify JAX sees the intended devices:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run python -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

The expected output is the GPU backend and one H200, or the 8 or 4 devices a multi-GPU recipe needs.
Do not continue if the device count is incompatible with the config's `fsdp_devices` or if either camera/calibration check fails.
For the recommended logical path, also stop if materialization and grouped-split review are incomplete.
The long-episode and vid7to54 configs are the intentional exceptions that skip the logical split.

## Derived datasets

None of the nine configs reads a Hugging Face download directly.
Every `repo_id` is a `local/...` LeRobot v2.1 directory under `HF_LEROBOT_HOME`, derived from a public source as follows:

| `repo_id` | Public source | Derivation | Reproducible from this repository |
| --- | --- | --- | --- |
| `local/cardboard_box_tcp_curated_logical_train` | `byang11259/cardboard_box_tcp_curated` (pinned revision above) | `scripts/split_cardboard_box_lerobot_v21.py` with a reviewed `configs/cardboard_box_segments.json` | yes, given the reviewed manifest |
| `local/cardboard_box_tcp_curated_x264` | `byang11259/cardboard_box_tcp_curated` (pinned revision above) | `scripts/reencode_lerobot_v21_videos.py` without `--scale` | yes |
| `local/cardboard_box_tcp_curated_10s_x264` | `byang11259/cardboard_box_tcp_curated` | resegmentation into 107 episodes of about 10 s, x264 re-encode, episodes 100-103 relabeled from a placeholder to the standard task string | **no**: neither the resegmentation nor the relabel is scripted here |
| `local/cardboard_box_tcp_vid7to54_x264` | [`byang11259/cardboard_box_tcp_vid7to54`](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_vid7to54) (46 session-length episodes, 205,213 frames, 960 x 960 declared) | `scripts/reencode_lerobot_v21_videos.py` with a `--scale` this repository does not record (confirm it from the training copy's `REENCODE_PROVENANCE.md` or by probing its videos with `ffprobe`), then episodes 37-39 relabeled from the placeholder `TODO: set task description` to the standard task string after visual review | re-encode provisional until the scale is confirmed; the relabel is a manual edit that is not scripted here |
| `local/stack_cubes_tcp_x264` | [`byang11259/stack_cubes_tcp`](https://huggingface.co/datasets/byang11259/stack_cubes_tcp) (46 episodes, 33,276 frames, 1920 px) | `scripts/reencode_lerobot_v21_videos.py --scale 384` | yes |
| `local/stack_cubes_takes_x264` | [`byang11259/stack_cubes_takes`](https://huggingface.co/datasets/byang11259/stack_cubes_takes) (43 episodes, 10,664 frames, 1920 px) | `scripts/reencode_lerobot_v21_videos.py --scale 384` | yes |

### Re-encoding videos with `scripts/reencode_lerobot_v21_videos.py`

The re-encode exists because lerobot's torchcodec decode path fails on the sparse-keyframe, B-frame source exports (see the note under [Registered configs](#registered-configs)).
The tool mirrors a LeRobot v2.1 dataset root into a new one:

```text
uv run python scripts/reencode_lerobot_v21_videos.py --src PATH --dst PATH [--scale INT] [--crf INT] [--gop INT] [--preset STR] [--workers INT] [--overwrite]
```

`--src` and `--dst` are required.
The defaults are no `--scale` (the source size is kept), `--crf 14`, `--gop 15`, `--preset medium`, `--workers 4`, and `--no-overwrite`.
The tool exits with an error for a `--scale` that is not a positive even width (yuv420p needs even dimensions), a `--crf` outside `[0, 51]`, a `--gop` below 1, an empty `--preset`, a `--workers` below 1, a `--src` without `meta/info.json` or without any `videos/**/*.mp4`, a `--dst` that equals, contains, or lies inside `--src`, and a missing `ffmpeg` on `PATH`.

- Every file that is not a `videos/**/*.mp4` is copied byte-identically into `DST` (`data/` parquet, `meta/`, and anything else); `meta/info.json` is then rewritten as described below, and `REENCODE_PROVENANCE.md` is added.
- Every `videos/**/*.mp4` is re-encoded as `ffmpeg -hide_banner -loglevel error -nostdin -y -i SRC -map 0:v:0 [-vf scale=N:-2] -an -c:v libx264 -preset PRESET -crf CRF -g GOP -bf 0 -pix_fmt yuv420p -movflags +faststart -f mp4 OUT`.
  No `-r` is passed, so the source frame rate is preserved.
- With `--scale N` the filter `-vf scale=N:-2` is applied: the width becomes `N` and the height keeps the aspect ratio rounded to an even number, so a square source becomes `N x N`.
- ffmpeg writes `<name>.mp4.partial`, which is renamed to the final name only after ffmpeg exits successfully; a failed encode deletes the partial file and aborts the run.
- An output that already exists is kept (not re-encoded) unless `--overwrite`, but it is verified against the current arguments either way.
  An interrupted run can therefore be resumed by repeating the same command, and a mirror made with a different `--scale` fails verification until it is re-encoded with `--overwrite`.
- Every output is decoded once with PyAV and checked against its source: the same frame count, the same frame rate, no B-frames, the first frame is a keyframe, no keyframe gap larger than `GOP` (the tail after the last keyframe counts as a gap), codec `h264` with pixel format `yuv420p`, and the expected size (the source size without `--scale`, or `N` wide by the even aspect-preserving height that `scale=N:-2` produces).
  An undecodable output counts as a violation.
  Violations are collected across all videos, and the tool exits non-zero naming every failing file.
- Videos are processed on `--workers` threads, each running one multithreaded ffmpeg; ffmpeg is located with `shutil.which`.
- All videos of one video key must decode to the same size; otherwise the tool exits with an error before writing metadata.
- On success `meta/info.json` is rewritten for every feature with `dtype: video`.
  The feature's own `shape` field (`[H, W, 3]`) is updated with `--scale` and left alone without it.
  Among the keys that the pinned lerobot 0.1.0 `get_video_info` writes into `features[key]["info"]`, `video.height` and `video.width` are updated when present and only with `--scale`, while `video.codec` becomes `h264` and `video.pix_fmt` becomes `yuv420p` always.
  The same edits are applied to a non-empty `features[key]["video_info"]` dict, the key the portable exporter writes; in either dict `video.channels` becomes 3 when present, and a true `has_audio` becomes false with its `audio.*` keys removed because audio is dropped.
- The tool warns without failing when `info.json` declares a `total_videos` that differs from the number of files found, or when a source feature's declared `shape` disagrees with the decoded source video.
- `DST/REENCODE_PROVENANCE.md` is deleted when a run starts and written at its end.
  A successful run's file records the resolved absolute source and destination paths, a UTC timestamp, the first line of `ffmpeg -version`, the output-side ffmpeg video arguments, how many videos were encoded in this run versus kept from an earlier one, every argument (`--src`, `--dst`, `--scale`, `--crf`, `--gop`, `--preset`, `--workers`, `--overwrite`), and a per-video table (relative path, source frames, output frames, output width x height).
  A run that fails verification writes the same file with a `FAILED` banner listing the offending files and `FAILED` table rows, and leaves `meta/info.json` as the unmodified source copy, so a directory left behind by a failed run never passes for a finished mirror.

Reproduce the four re-encoded copies from `huggingface-cli download --repo-type dataset` targets laid out like the curated download above:

```bash
uv run python scripts/reencode_lerobot_v21_videos.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_x264"

# Provisional: this repository does not record whether the vid7to54 training copy was downscaled.
# Confirm the frame size from that copy's REENCODE_PROVENANCE.md, if it has one, or with ffprobe
# on its videos, and add "--scale 384" if it shows 384 px frames.
uv run python scripts/reencode_lerobot_v21_videos.py \
  --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_vid7to54" \
  --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_vid7to54_x264"

uv run python scripts/reencode_lerobot_v21_videos.py \
  --src "$HF_LEROBOT_HOME/byang11259/stack_cubes_tcp" \
  --dst "$HF_LEROBOT_HOME/local/stack_cubes_tcp_x264" \
  --scale 384

uv run python scripts/reencode_lerobot_v21_videos.py \
  --src "$HF_LEROBOT_HOME/byang11259/stack_cubes_takes" \
  --dst "$HF_LEROBOT_HOME/local/stack_cubes_takes_x264" \
  --scale 384
```

The defaults `--crf 14 --gop 15` are the settings recorded for the existing copies.
After the vid7to54 re-encode, episodes 37-39 still carry the placeholder task string; the training copy had them relabeled by hand, and this repository does not script that edit.
The public `byang11259/cardboard_box_tcp_vid7to54` export declares 960 x 960 video frames in its `meta/info.json`, unlike the 384 px curated exports, and nothing in this repository records whether its derived copy was downscaled.
The vid7to54 command above is therefore provisional: confirm the frame size from the training copy's `REENCODE_PROVENANCE.md`, if it has one, or by probing its videos with `ffprobe`, add `--scale 384` if that copy was downscaled, and do not assume the 384 px crop geometry described under [Cropped fisheye views](#cropped-fisheye-views-image_crop) until then.
The `local/cardboard_box_tcp_curated_10s_x264` copy cannot be regenerated with this tool at all: it is a resegmentation of the curated source into about 10 s episodes, and neither that segmentation nor its relabel of episodes 100-103 is scripted here.
An existing `stack_cubes_tcp_x264` copy (46 episodes, 33,276 frames) made before this tool existed decodes as h264 384 x 384 at 29.97 fps with no B-frames, a keyframe on frame 0, and no keyframe gap above 15 in any of its 92 videos, but its `meta/info.json` still declares shape `[1920, 1920, 3]` and `video.codec: hevc` because its metadata was never rewritten.
A copy rebuilt with the tool has the same stream properties and differs from it in `meta/info.json` and in the added `REENCODE_PROVENANCE.md`.
Treat each derived directory's `REENCODE_PROVENANCE.md` and content hash as part of the experiment record.

## CPU preflight

The focused policy tests do not require a GPU. Force CPU mode so this step is
safe on a login node:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run pytest -q src/openpi/policies/umi_dual_franka_policy_test.py
```

Run the config and logical-episode splitter tests as well:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run pytest -q \
    src/openpi/training/config_test.py \
    scripts/split_cardboard_box_lerobot_v21_test.py \
    scripts/inspect_umi_dual_franka_dataset_test.py \
    scripts/eval_umi_dual_franka_open_loop_test.py
```

Inspect full, unnormalized 50-step chunks and transform round trips before
computing stats. Recommended logical relative and absolute:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative \
  --repo-id local/cardboard_box_tcp_curated_logical_train \
  --num-samples 32

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute \
  --repo-id local/cardboard_box_tcp_curated_logical_train \
  --num-samples 32
```

Original long-source relative and absolute:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_long_episode \
  --repo-id local/cardboard_box_tcp_curated_x264 \
  --num-samples 32

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute_long_episode \
  --repo-id local/cardboard_box_tcp_curated_x264 \
  --num-samples 32
```

The four gripper-only configs, each against its own registered repo:

```bash
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode \
  --repo-id local/cardboard_box_tcp_curated_10s_x264 \
  --num-samples 32

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode \
  --repo-id local/cardboard_box_tcp_curated_x264 \
  --num-samples 32

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54 \
  --repo-id local/cardboard_box_tcp_vid7to54_x264 \
  --num-samples 32

JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' \
  uv run python scripts/inspect_umi_dual_franka_dataset.py \
  --config-name pi05_umi_dual_franka_stack_cubes_relative_gripper_only \
  --repo-id local/stack_cubes_tcp_x264 \
  --num-samples 32
```

The inspector verifies raw16 shapes, `action[t] = state[t + 1]`, transformed state/action shapes (state20 for full-state configs, 2-D gripper state for the gripper-only configs, action20 for all), image masks, and raw16 round trips; for the gripper-only configs it emulates the documented client-side anchor composition before checking the round trip.
It does not support `pi05_umi_dual_franka_stack_cubes_relative_history`: it requests only the current state row and derives its expected state dimension from the output transform type, so that config's 20-D relative-history state fails its 2-D check.
Preflight that config with the config tests above and a training smoke run instead.
On a source-episode repo, its reported episode count means **source** episodes.
It samples chunks that fit within those source bounds; it does not infer or mask physical-box boundaries.
Review known boundary-adjacent chunks separately.

These are schema/transform preflights, not a substitute for a small
dataloader inspection, GPU compilation test, camera dry run, or
controller-disabled hardware rehearsal.

## Compute fresh normalization statistics

The published dataset's raw absolute-16 statistics (for example,
`meta/stats.json`) are invalid for the transformed absolute-state-20 and
relative-action-20 distributions. The absolute baseline also changes
quaternions to 6D rotations and therefore needs new statistics.

Every config needs stats computed after its final action transform. Never share
stats across relative versus absolute or logical versus long episode paths.

### Logical-episode stats (recommended)

Compute these after logical-episode construction and grouped split selection.
The configured repo ID selects the logical **train** dataset, so
validation/test episodes are not included. The splitter's terminal target is
an explicit in-episode no-op rather than an action from the next box. If a
separate padding mask is introduced, the norm-stat pass and training loader
must apply the same mask.

Relative config:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative
```

Absolute baseline:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute
```

The two files are written separately:

```text
assets/pi05_umi_dual_franka_cardboard_box_relative/local/cardboard_box_tcp_curated_logical_train/norm_stats.json
assets/pi05_umi_dual_franka_cardboard_box_absolute/local/cardboard_box_tcp_curated_logical_train/norm_stats.json
```

Do not copy, symlink, or share one with the other. Before training, inspect
`mean`, `std`, `q01`, and `q99` for all 20 physical dimensions and confirm that
the asset ID is `local/cardboard_box_tcp_curated_logical_train`. Tiny spreads,
unexpected translation magnitudes, gripper polarity changes, or padded
dimensions in the physical stats are stop signs.

### Raw-rot6d variant of the quantile stats

rot6d components are rotation-matrix entries, inherently bounded in [-1, 1] and geometrically meaningful, so per-dim quantile scaling distorts the rotation manifold the loss is computed on.
For runs that keep openpi's quantile normalization on translation and gripper dims (and the 2-D gripper-only state) but pass rot6d through raw, so that the network output feeds `rotation_6d_to_matrix` directly, neutralize the rot6d dims of the stats FILE after computing it:

```bash
uv run scripts/neutralize_rot6d_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode
```

This writes neutral parameters (mean 0 / std 1 / q01 -1 / q99 +1) into the rot6d entries, making the quantile normalizer an identity for them, and keeps a one-time `norm_stats.json.pre_rot6d_neutral.bak` backup.
The 12 rot6d action dims are always neutralized.
The script also reads the config's `state_mode`: for `relative_history` it neutralizes the 12 rot6d dims of the 20-D `state` entry as well, because that state shares the `[xyz(3), rot6d(6), gripper(1)]` arm layout; for `full` and `gripper_only` it leaves the state alone, and a 2-D gripper-only state is skipped even when asked because it has no rot6d dims.
Override the automatic choice with `--include-state True` or `--include-state False`.
The file-level approach keeps every checkpoint self-consistent: each embeds the stats it trained with, so older full-normalization checkpoints under the same config still serve exactly as trained.
Only the two stack-cubes configs record raw rot6d as part of their recipe; their neutralize commands follow their norm-stat commands below.
For every cardboard-box config, including vid7to54, neutralizing is an optional raw-rot6d variant rather than the recorded recipe.

### Original long-source stats (ablation)

These commands intentionally traverse source-derived episodes that are not box-segmented (three configs read the source episodes as recorded, and `_gripper_only_long_episode` reads the 10 s time-resegmented copy) through the stock 50-step action-chunk path:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_long_episode

uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute_long_episode

uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode

uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode
```

They write four additional, independent files (paths relative to the
configured `assets_base_dir`):

```text
assets/pi05_umi_dual_franka_cardboard_box_relative_long_episode/local/cardboard_box_tcp_curated_x264/norm_stats.json
assets/pi05_umi_dual_franka_cardboard_box_absolute_long_episode/local/cardboard_box_tcp_curated_x264/norm_stats.json
assets/pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode/local/cardboard_box_tcp_curated_10s_x264/norm_stats.json
assets/pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode/local/cardboard_box_tcp_curated_x264/norm_stats.json
```

The long relative stats include any fixed-anchor jumps across physical-box boundaries.
All long stats include source-end clamping/repetition.
The two gripper-only files' `state` stats are 2-D; their action stats follow the same relative distribution as the relative config.
Treat those distributions as part of the ablation; do not replace them with logical stats to make the run appear better behaved.
Inspect all nine files (these four, the two logical files, and the three below) and record their asset IDs with the experiments.

### vid7to54 and stack-cubes stats

The vid7to54 config traverses the complete session-length export, and the stack-cubes configs traverse their own derived repos:

```bash
uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54

uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_stack_cubes_relative_gripper_only

uv run scripts/compute_norm_stats.py \
  --config-name pi05_umi_dual_franka_stack_cubes_relative_history
```

They write three more independent files:

```text
assets/pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54/local/cardboard_box_tcp_vid7to54_x264/norm_stats.json
assets/pi05_umi_dual_franka_stack_cubes_relative_gripper_only/local/stack_cubes_tcp_x264/norm_stats.json
assets/pi05_umi_dual_franka_stack_cubes_relative_history/local/stack_cubes_takes_x264/norm_stats.json
```

The vid7to54 pass decodes 205,213 frames from 46 session-length videos per camera; the config's 32 data workers exist for this step as much as for training.
The two gripper-only files' `state` stats are 2-D; the relative-history file's `state` stats are 20-D (previous-frame pose in the current TCP frame plus the current grippers).
The recorded recipe of the two stack-cubes configs keeps rot6d raw; neutralize each of their files after computing it (the relative-history run patches the `state` entry too):

```bash
uv run scripts/neutralize_rot6d_norm_stats.py \
  --config-name pi05_umi_dual_franka_stack_cubes_relative_gripper_only

uv run scripts/neutralize_rot6d_norm_stats.py \
  --config-name pi05_umi_dual_franka_stack_cubes_relative_history
```

The vid7to54 config copies the 10 s gripper-only recipe, and neither of them records a neutralize step.
To train vid7to54 as an optional raw-rot6d variant instead, neutralize its file the same way and record that choice with the run:

```bash
uv run scripts/neutralize_rot6d_norm_stats.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54
```

## JAX training on H200

### Logical-episode training (recommended)

Run the two experiments separately with the same dataset revision, grouped
split, seed, batch construction, and evaluation protocol. First compile the
complete data/model path and execute two steps with a small batch:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_umi_dual_franka_cardboard_box_relative \
  --exp-name=fold_box_relative_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_umi_dual_franka_cardboard_box_absolute \
  --exp-name=fold_box_absolute_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled
```

Inspect the loaded norm-stat paths, first camera batch, transformed tensor
shapes/ranges, finite losses, and checkpoints before starting full runs.

Primary relative representation:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_umi_dual_franka_cardboard_box_relative \
  --exp-name=fold_box_relative_v1
```

Absolute-action baseline:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py pi05_umi_dual_franka_cardboard_box_absolute \
  --exp-name=fold_box_absolute_v1
```

These use the registered one-H200 defaults: batch 32, one FSDP device, eight
workers, and 5,000 steps. Add `--no-wandb-enabled` if the machine should not
contact Weights & Biases.

Use `--resume` to continue an existing experiment. Use `--overwrite` only when
deliberately deleting an experiment directory. Watch the first logged camera
batch, transformed state/action ranges, loss, gradient norm, and device memory.
Stop on NaNs, wrong camera ordering, unmasked base images, cross-box chunks, or
implausible Cartesian magnitudes.

### Original long-source training (ablation)

After computing the four long-source norm-stat files, compile and execute two steps for each long config.
The two full-state commands are shown here; the two gripper-only long configs follow the multi-GPU pattern under [Gripper-only and relative-history training](#gripper-only-and-relative-history-training):

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_relative_long_episode \
  --exp-name=fold_box_relative_long_episode_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_absolute_long_episode \
  --exp-name=fold_box_absolute_long_episode_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled
```

Confirm that each smoke run loads its own
`local/cardboard_box_tcp_curated_x264` asset tree under its config-name
directory. Then run the full long-source experiments:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_relative_long_episode \
  --exp-name=fold_box_relative_long_episode_v1

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_absolute_long_episode \
  --exp-name=fold_box_absolute_long_episode_v1
```

The registered one-H200 defaults apply to these two full-state configs.
Cross-box chunks are expected here; stop only if behavior differs from the documented stock semantics, values are non-finite, or transforms/cameras are wrong.
Compare long relative versus long absolute to isolate the action representation within the unsegmented source path.
Comparing a long config with a logical config also changes episode construction, sample distribution, terminal targets, and train/validation availability, so it is not a pure action-representation ablation.

### Gripper-only and relative-history training

The four gripper-only configs register an 8-GPU recipe (batch 128, `fsdp_devices=8`, 10,000 steps) and the relative-history config a 4-GPU recipe (batch 64, `fsdp_devices=4`, 6,000 steps, checkpoints every 2,000).
Smoke-test each on one GPU by overriding the mesh and the batch together, because the training script rejects a batch size that the visible device count does not divide and a device count that `fsdp_devices` does not divide:

```bash
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode \
  --exp-name=fold_box_relative_gripper_only_smoke \
  --num-train-steps=2 \
  --batch-size=1 \
  --fsdp-devices=1 \
  --num-workers=0 \
  --save-interval=1 \
  --no-wandb-enabled
```

Repeat with the other four config names and their own `--exp-name`.
Confirm that each smoke run loads its own asset tree (`local/cardboard_box_tcp_curated_10s_x264`, `local/cardboard_box_tcp_curated_x264`, `local/cardboard_box_tcp_vid7to54_x264`, `local/stack_cubes_tcp_x264`, or `local/stack_cubes_takes_x264`) under its config-name directory, and that the loaded stats carry the neutralized rot6d entries when the recipe calls for them.
Then run the registered recipes on the matching number of GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode \
  --exp-name=fold_box_relative_gripper_only_v1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode \
  --exp-name=fold_box_relative_gripper_only_crop272_v1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54 \
  --exp-name=fold_box_relative_gripper_only_vid7to54_v1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_stack_cubes_relative_gripper_only \
  --exp-name=stack_cubes_relative_gripper_only_v1

CUDA_VISIBLE_DEVICES=0,1,2,3 XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
  uv run scripts/train.py \
  pi05_umi_dual_franka_stack_cubes_relative_history \
  --exp-name=stack_cubes_relative_history_v1
```

The two gripper-only `_long_episode` configs and the vid7to54 config train on source-derived episodes that are not box-segmented (the 10 s copy behind `_gripper_only_long_episode` is time-resegmented, not box-segmented), so LeRobot chunks can still cross a physical-box boundary and the long-source caveats above apply to them as well.
For the relative-history run, additionally watch the first logged state batch: it must be 20-D, and rows at an episode's first frame carry an exact identity history (zero translation and rot6d `[1, 0, 0, 0, 1, 0]` per arm) because LeRobot clamps the window there.

## Rank checkpoints on held-out logical episodes

This section applies to the recommended logical configs. The long configs train
on the full source repo, including source episodes from which the logical
validation set is derived; evaluating them there is not held out and must not
be labeled validation.

Use only `local/cardboard_box_tcp_curated_logical_validation`, whose source
episodes and collection sessions are absent from training. The evaluation
script reconstructs the training transforms, selects only complete 50-step
chunks, and reports normalized 20D MSE overall and for translation, 6D
rotation, and gripper dimensions. It loads normalization stats from each
checkpoint's embedded `assets/<asset_id>` tree, matching `serve_policy.py`;
stale or missing local `./assets` stats do not define the score.

Rank relative checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/eval_umi_dual_franka_open_loop.py \
  --config-name pi05_umi_dual_franka_cardboard_box_relative \
  --repo-id local/cardboard_box_tcp_curated_logical_validation \
  --episodes all \
  --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_relative/fold_box_relative_v1/1000 \
  --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_relative/fold_box_relative_v1/4999
```

Rank absolute checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 \
  uv run python scripts/eval_umi_dual_franka_open_loop.py \
  --config-name pi05_umi_dual_franka_cardboard_box_absolute \
  --repo-id local/cardboard_box_tcp_curated_logical_validation \
  --episodes all \
  --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_absolute/fold_box_absolute_v1/1000 \
  --checkpoint checkpoints/pi05_umi_dual_franka_cardboard_box_absolute/fold_box_absolute_v1/4999
```

Pass only checkpoint steps that exist. Lower held-out MSE is useful for ranking
checkpoints **within a representation**, but it is not a hardware success rate
and does not replace calibration replay, safety checks, or controlled robot
evaluation. Because the two logical configs have distinct normalization
statistics, report each component and physical-space rollout metrics when
comparing the representations.

## Serve a matching config with JAX

Replace `<step>` with a saved checkpoint step. The config and checkpoint must
match exactly; this is also how the server locates the correct norm stats.

Recommended logical relative policy:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_relative \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_relative/fold_box_relative_v1/<step>
```

Recommended logical absolute baseline:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_absolute \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_absolute/fold_box_absolute_v1/<step>
```

Long-source relative ablation:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_relative_long_episode \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_relative_long_episode/fold_box_relative_long_episode_v1/<step>
```

Long-source absolute ablation:

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_absolute_long_episode \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_absolute_long_episode/fold_box_absolute_long_episode_v1/<step>
```

Cross-embodiment variants (every config whose `state_mode` is not `"full"` serves a **relative** chunk that the robot client must compose with its own saved anchors; see the deployment section):

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_relative_gripper_only_long_episode/fold_box_relative_gripper_only_v1/<step>

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_relative_gripper_only_crop272_long_episode/fold_box_relative_gripper_only_crop272_v1/<step>

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54 \
  --policy.dir=checkpoints/pi05_umi_dual_franka_cardboard_box_relative_gripper_only_vid7to54/fold_box_relative_gripper_only_vid7to54_v1/<step>

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_stack_cubes_relative_gripper_only \
  --policy.dir=checkpoints/pi05_umi_dual_franka_stack_cubes_relative_gripper_only/stack_cubes_relative_gripper_only_v1/<step>

CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_umi_dual_franka_stack_cubes_relative_history \
  --policy.dir=checkpoints/pi05_umi_dual_franka_stack_cubes_relative_history/stack_cubes_relative_history_v1/<step>
```

The relative-history server additionally expects `observation/state` as an `(observation_horizon, 16)` window, oldest frame first (see [Relative-history-state cross-embodiment variant](#relative-history-state-cross-embodiment-variant)).

Serving support does not make a long-source checkpoint deployment-qualified.
Keep it in offline/controller-disabled evaluation until it passes the same
held-out, calibration, and safety gates as the recommended path.

The WebSocket server listens on all interfaces and defaults to port 8000.
Restrict access with the machine firewall or a private robot network; the
server does not replace a robot safety layer.

## Remote WebSocket robot deployment

Install the lightweight client in the robot environment:

```bash
cd "$OPENPI_ROOT/packages/openpi-client"
pip install -e .
```

Create one `WebsocketClientPolicy` outside the control loop and connect it to
the H200 server:

```python
from openpi_client import websocket_client_policy

policy = websocket_client_policy.WebsocketClientPolicy(
    host="POLICY_SERVER_IP",
    port=8000,
)
```

At each policy query, send synchronized, unnormalized observations:

```python
observation = {
    "observation/state": raw_absolute_state16,
    "observation/left_head": left_front_equi_uint8_hwc,
    "observation/right_head": right_front_equi_uint8_hwc,
    "prompt": "Assemble the cardboard box and put it into the bin",
}
action_chunk = policy.infer(observation)["actions"]
```

For `pi05_umi_dual_franka_stack_cubes_relative_history`, send the state as the observation window instead, oldest first with the current frame last; its `observation_horizon` is 2, so the window is the previous and the current raw state:

```python
observation = {
    "observation/state": np.stack([previous_raw_absolute_state16, raw_absolute_state16]),
    "observation/left_head": left_front_equi_uint8_hwc,
    "observation/right_head": right_front_equi_uint8_hwc,
    "prompt": "Stack the cubes into a tower",
}
```

A bare `(16,)` state is accepted by that config but degrades to "no motion": the history frame is taken to be the current frame, so the relative-history entries are identity and only the grippers carry information.

The state must describe both live chopstick-tip TCPs in the calibrated common
world frame, have a floating dtype, and use `xyzw`, left-then-right, and
`1 = open`. Prefer native 384 × 384 `uint8` HWC images; the adapter also accepts
CHW and validated floating image ranges before the shared resize/pad. Images
must be timestamp-matched to the state. The server applies checkpoint
normalization; the client must not pre-normalize state or actions.

The response type follows the config's `state_mode`, not its name.
With `state_mode="full"` (the four full-state configs) the WebSocket response is an absolute raw action chunk with shape `[50, 16]` and the same left/right, `xyz + xyzw + gripper` layout as the raw state.
With any other `state_mode` (`gripper_only` in four configs, `relative_history` in one) the server has no absolute pose to compose with and returns a `[50, 16]` **relative** chunk in the same layout (see [Gripper-only-state cross-embodiment variant](#gripper-only-state-cross-embodiment-variant)); the client must compose the chunk with its own saved query-time anchors, and for the relative-history config that anchor is the last frame of the window it sent.
For the full-state relative configs, the server-side output transform uses the unnormalized state20 derived from that query as one immutable anchor for every waypoint, then returns absolute world-frame targets.
Do not compose these returned targets a second time on the client.
Either absolute baseline also returns absolute world-frame targets, directly decoded from action20.

Log the state16 sent with each request as the relative policy's anchor record,
then apply the world/base/flange command chain to the returned absolute
targets. Regardless of representation, reject malformed, late, non-finite,
out-of-workspace, or discontinuous responses before the controller sees them.

## Action horizon and receding-horizon execution

Fifty waypoints at 29.97 Hz span approximately:

```text
50 / 29.97 ≈ 1.67 seconds
```

That is a prediction horizon, not permission to execute 1.67 seconds open
loop. Start with a short, conservative prefix under controller-disabled replay
and low-speed testing. Select the executed prefix from measured:

- network and inference latency;
- camera/state age and synchronization;
- Cartesian controller tracking error;
- contact dynamics and force/torque margins;
- inter-arm coordination and collision clearance; and
- how quickly model predictions become stale.

After the prefix, acquire a new synchronized observation, query again, and
re-anchor both arms. Log query timestamp, state/image timestamps, inference
latency, chosen prefix length, anchors, predicted chunk, filtered commands, and
executed commands.

If control frequency is changed, resample images, state, actions, timestamps,
episode boundaries, and masks together; then choose a horizon representing the
intended physical duration and recompute every affected config's norm stats.
Changing only the nominal frequency or horizon silently changes supervision.

## Safety gates

Use a controller-independent safety layer for both arms. Before any powered
run, require:

- verified translation units, quaternion order, gripper polarity, arm order,
  and tool transform;
- current world-to-base calibration for both robots;
- workspace and table/fixture exclusion volumes;
- Cartesian translation/rotation velocity and acceleration limits;
- force/torque and contact limits appropriate for cardboard interaction;
- Franka joint position, velocity, torque, and singularity margins;
- self-collision, inter-arm collision, tool collision, and camera collision
  checks;
- IK convergence and continuity checks when IK is used;
- finite-value, jump, timestamp-age, and network-timeout checks;
- an accessible hardware emergency stop and a tested software stop; and
- stop-on-stale behavior for either camera, FK, calibration, policy response,
  or controller state.

Commission in this order:

1. offline transform and chunk replay;
2. policy server with recorded observations;
3. controller disabled, commands logged only;
4. single waypoint at very low speed with no box;
5. one arm at a time in a clear workspace;
6. both arms at low speed with collision monitoring;
7. compliant contact with an expendable box; and
8. short receding-horizon prefixes, increased only after log review.

Do not enable both robots merely because model inference and IK succeed.

## Required validation and tests

The preflight suite and experiment checklist must cover:

- raw-state length 16, left/right ordering, quaternion `xyzw`, and
  `action[t] = state[t + 1]`;
- absence of invented joint-position, joint-velocity, or joint-target policy
  dimensions;
- measured translation units and the no-double-`-90°` rule;
- raw absolute16 → state/action20 → raw absolute16 round trips;
- quaternion sign equivalence (`q` and `-q`);
- pure tool-frame translations and rotations with known signs;
- identity and random-rotation 6D encode/decode round trips;
- identical fixed anchor use for every waypoint in a relative chunk;
- independent, synchronized left and right anchors;
- grippers remaining future absolute values with `1 = open`;
- physical 20D stats and correct zero-padding/slicing to/from model 32D;
- nine separate norm-stat trees across representation, state mode, image
  crop, and task/dataset path;
- for the gripper-only configs: a 2-D gripper state (no pose dimensions), a
  relative served chunk, and client-side anchor composition matching the
  offline transform;
- for the relative-history config: a 20-D state with no absolute pose, the
  previous frame expressed in the current TCP frame, an oldest-first
  observation window whose last frame anchors the relative actions, and an
  identity history for a bare single-frame state;
- online WebSocket transforms matching offline training transforms;
- exactly two real camera streams, correct ordering, and an absent/masked base
  slot;
- identical fisheye projection and preprocessing between data and deployment;
- for logical configs, no sampled query, shifted action, or 50-step chunk
  crossing a physical-box boundary;
- for long configs, exact `0..49 / fps` action deltas, source-episode clamping,
  ignored padding/boundary masks, and intentionally possible cross-box chunks;
- the materializer's final absolute no-op, frame-accurate two-video trims, and
  rewritten LeRobot indices/timestamps;
- when the masked alternative is used, identical terminal/boundary masks in
  sampling, norm stats, and loss;
- session/source-grouped train/validation/test splits with no sibling leakage;
- fold-only crops containing no bin-placement motion;
- full-task crops and prompts retaining bin placement;
- held-out logical-episode checkpoint ranking with no train-session leakage;
- world/base/flange/TCP calibration-chain round trips for both arms;
- stale-data, non-finite, workspace, IK, and collision rejection; and
- controller-disabled and low-speed hardware rehearsals.

Archive the selected episode path, source revision, logical manifest and
derived-data hash when applicable, norm stats, config name, checkpoint step,
git commit, calibration IDs, camera settings, and safety-limit version with
every reported result.

## Repository entry points

- [Logical-episode splitter](../scripts/split_cardboard_box_lerobot_v21.py)
- [Reviewed-manifest template](../configs/cardboard_box_segments.example.json)
- [LeRobot v2.1 video re-encoder](../scripts/reencode_lerobot_v21_videos.py)
- [rot6d norm-stat neutralizer](../scripts/neutralize_rot6d_norm_stats.py)
- [Dual-Franka UMI policy transforms](../src/openpi/policies/umi_dual_franka_policy.py)
- [Training data and model configs](../src/openpi/training/config.py)
- [Dataset/chunk transform inspector](../scripts/inspect_umi_dual_franka_dataset.py)
- [Held-out open-loop checkpoint ranking](../scripts/eval_umi_dual_franka_open_loop.py)
- [Policy transform tests](../src/openpi/policies/umi_dual_franka_policy_test.py)
- [Config tests](../src/openpi/training/config_test.py)
- [Splitter tests](../scripts/split_cardboard_box_lerobot_v21_test.py)
- [Inspector tests](../scripts/inspect_umi_dual_franka_dataset_test.py)
- [Checkpoint-ranking tests](../scripts/eval_umi_dual_franka_open_loop_test.py)

## Official and source references

- [Cardboard-box dataset card and schema](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated#schema)
- [Dataset coordinate conventions](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated#coordinate-conventions-read-before-using)
- [Dataset relative-representation notes](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated#using-the-hy-vla--relative-representation)
- [Dataset normalization notes](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated#normalization)
- [Pinned dataset revision](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_curated/tree/a366d2b92723d795b6b93e6303f57708e87b63ea)
- [Complete vid7to54 cardboard-box export](https://huggingface.co/datasets/byang11259/cardboard_box_tcp_vid7to54)
- [Stack-cubes export](https://huggingface.co/datasets/byang11259/stack_cubes_tcp)
- [Stack-cubes short-takes export](https://huggingface.co/datasets/byang11259/stack_cubes_takes)
- [Hy-VLA SE(3) transform implementation, pinned revision](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/utils/transform_utils.py#L16-L123)
- [Hy UMI dataset config, pinned revision](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/config/dataset/umi_lance.yaml#L28-L50)
- [Hy UMI dataset loader, pinned revision](https://github.com/Tencent-Hunyuan/Hy-Embodied-0.5-VLA/blob/8ba4c8cbdf42a4bcf0a19be4bd2841405dfe15e9/hy_vla/data/umi_dataset.py#L520-L630)
- [Physical Intelligence π0.5 overview](https://www.physicalintelligence.company/blog/pi05)
- [Official openpi repository](https://github.com/Physical-Intelligence/openpi)
- [OpenPI LeRobot chunk construction in this branch](../src/openpi/training/data_loader.py)
- [Pinned LeRobot episode clamp/padding behavior](https://github.com/huggingface/lerobot/blob/0cf864870cf29f4738d3ade893e6fd13fbd7cdb5/lerobot/common/datasets/lerobot_dataset.py#L665-L678)
- [Local openpi normalization guide](norm_stats.md)
- [Local openpi remote-inference guide](remote_inference.md)
