"""End-to-end tests for scripts/reencode_lerobot_v21_videos.py on synthetic LeRobot v2.1 datasets."""

import dataclasses
import itertools
import json
import os
import pathlib
import shutil
import subprocess

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts import reencode_lerobot_v21_videos as tool

if shutil.which("ffmpeg") is None:
    pytest.skip("ffmpeg is not on PATH", allow_module_level=True)

FRAMES = 40
FPS = 30
SIZE = 128
GOP = 15
TASK = "Stack the cubes into a tower"
# One feature carries lerobot 0.1.0's ``info`` dict (with height/width), the
# other the exporter's ``video_info`` dict (without them).
INFO_KEY = "observation.images.left_head"
VIDEO_INFO_KEY = "observation.images.right_head"
VIDEO_KEYS = (INFO_KEY, VIDEO_INFO_KEY)
INFO_JSON = pathlib.Path("meta/info.json")


@dataclasses.dataclass(frozen=True)
class DatasetSpec:
    """Shape of a synthetic source dataset: every video shares one size and frame count."""

    width: int = SIZE
    height: int = SIZE
    frames: int = FRAMES
    episodes: int = 1

    @property
    def videos(self) -> tuple[str, ...]:
        return tuple(_video_relpath(key, episode) for episode in range(self.episodes) for key in VIDEO_KEYS)


NON_SQUARE = DatasetSpec(width=160, height=96)
ONE_FRAME = DatasetSpec(frames=1)
TWO_EPISODES = DatasetSpec(episodes=2)


@dataclasses.dataclass(frozen=True)
class Probe:
    frames: int
    width: int
    height: int
    codec: str
    fps: float
    b_frames: int
    keyframes: tuple[int, ...]

    @property
    def max_keyframe_gap(self) -> int:
        marks = [*self.keyframes, self.frames]
        return max(later - earlier for earlier, later in itertools.pairwise(marks))


def _probe(path: pathlib.Path) -> Probe:
    """Independent PyAV probe (does not reuse the tool's own verification code)."""
    keyframes = []
    b_frames = 0
    frames = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for index, frame in enumerate(container.decode(stream)):
            frames += 1
            if frame.key_frame:
                keyframes.append(index)
            if int(frame.pict_type) == int(av.video.frame.PictureType.B):
                b_frames += 1
        return Probe(
            frames=frames,
            width=stream.width,
            height=stream.height,
            codec=stream.codec.canonical_name,
            fps=float(stream.base_rate),
            b_frames=b_frames,
            keyframes=tuple(keyframes),
        )


def _write_source_video(path: pathlib.Path, *, width: int, height: int, frames: int) -> None:
    """A long-GOP clip with B-frames, i.e. the stream layout the tool exists to fix."""
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"testsrc=size={width}x{height}:rate={FPS}",
        "-frames:v",
        str(frames),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "ultrafast",
        "-g",
        "250",
        "-bf",
        "3",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.run(command, check=True)


def _video_feature(info_key: str, *, spec: DatasetSpec, with_dims: bool) -> dict:
    video_info = {
        "video.fps": FPS,
        "video.codec": "hevc",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }
    if with_dims:
        video_info = {"video.height": spec.height, "video.width": spec.width, **video_info, "video.channels": 3}
    return {
        "dtype": "video",
        "shape": [spec.height, spec.width, 3],
        "names": ["height", "width", "channels"],
        info_key: video_info,
    }


def _scalar_feature(dtype: str) -> dict:
    return {"dtype": dtype, "shape": [1], "names": None}


def _write_parquet(path: pathlib.Path, *, spec: DatasetSpec, episode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = spec.frames
    states = np.random.default_rng(episode).standard_normal((frames, 16)).astype(np.float32)
    actions = np.concatenate([states[1:], states[-1:]], axis=0)
    vector = pa.list_(pa.float32(), 16)
    first = episode * frames
    table = pa.table(
        {
            "observation.state": pa.array(states.tolist(), type=vector),
            "action": pa.array(actions.tolist(), type=vector),
            "timestamp": pa.array(np.arange(frames) / FPS, type=pa.float32()),
            "frame_index": pa.array(range(frames), type=pa.int64()),
            "episode_index": pa.array([episode] * frames, type=pa.int64()),
            "index": pa.array(range(first, first + frames), type=pa.int64()),
            "task_index": pa.array([0] * frames, type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def _video_relpath(key: str, episode: int = 0) -> str:
    return f"videos/chunk-000/{key}/episode_{episode:06d}.mp4"


def _parquet_relpath(episode: int = 0) -> str:
    return f"data/chunk-000/episode_{episode:06d}.parquet"


def _write_dataset(root: pathlib.Path, spec: DatasetSpec) -> pathlib.Path:
    """Write a LeRobot v2.1 dataset of SPEC's shape with two video features under ROOT."""
    meta = root / "meta"
    meta.mkdir(parents=True)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "portable_bimanual",
        "total_episodes": spec.episodes,
        "total_frames": spec.frames * spec.episodes,
        "total_tasks": 1,
        "total_videos": len(spec.videos),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": FPS,
        "splits": {"train": f"0:{spec.episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {"dtype": "float32", "shape": [16], "names": [f"d{i}" for i in range(16)]},
            "action": {"dtype": "float32", "shape": [16], "names": [f"d{i}" for i in range(16)]},
            "timestamp": _scalar_feature("float32"),
            "frame_index": _scalar_feature("int64"),
            "episode_index": _scalar_feature("int64"),
            "index": _scalar_feature("int64"),
            "task_index": _scalar_feature("int64"),
            INFO_KEY: _video_feature("info", spec=spec, with_dims=True),
            VIDEO_INFO_KEY: _video_feature("video_info", spec=spec, with_dims=False),
        },
    }
    (meta / "info.json").write_text(json.dumps(info, indent=2) + "\n")
    episodes = range(spec.episodes)
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps({"episode_index": e, "tasks": [TASK], "length": spec.frames}) + "\n" for e in episodes)
    )
    (meta / "episodes_stats.jsonl").write_text(
        "".join(json.dumps({"episode_index": e, "stats": {}}) + "\n" for e in episodes)
    )
    (meta / "tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": TASK}) + "\n")
    for episode in episodes:
        _write_parquet(root / _parquet_relpath(episode), spec=spec, episode=episode)
        for key in VIDEO_KEYS:
            _write_source_video(
                root / _video_relpath(key, episode), width=spec.width, height=spec.height, frames=spec.frames
            )
    return root


@pytest.fixture
def source_dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_dataset(tmp_path / "src", DatasetSpec())


def _assert_valid_output(path: pathlib.Path, *, width: int, height: int, frames: int = FRAMES) -> Probe:
    probe = _probe(path)
    assert probe.frames == frames
    assert probe.b_frames == 0
    assert probe.keyframes[0] == 0
    assert probe.max_keyframe_gap <= GOP
    assert (probe.width, probe.height) == (width, height)
    assert probe.codec == "h264"
    assert probe.fps == FPS
    return probe


def _assert_copied_byte_identical(src: pathlib.Path, dst: pathlib.Path) -> None:
    """Every non-video source file except meta/info.json is in DST byte-identically."""
    copied = [
        path.relative_to(src)
        for path in sorted(src.rglob("*"))
        if path.is_file() and path.suffix != ".mp4" and path.relative_to(src) != INFO_JSON
    ]
    assert copied
    for relative in copied:
        assert (dst / relative).read_bytes() == (src / relative).read_bytes(), relative


def _provenance(dst: pathlib.Path) -> str:
    return (dst / "REENCODE_PROVENANCE.md").read_text()


def test_source_fixture_has_the_layout_the_tool_fixes(source_dataset: pathlib.Path):
    probe = _probe(source_dataset / _video_relpath(INFO_KEY))
    assert probe.frames == FRAMES
    assert probe.b_frames > 0
    assert probe.max_keyframe_gap > GOP


def test_reencode_keeps_resolution_and_rewrites_codec_keys(source_dataset: pathlib.Path, tmp_path: pathlib.Path):
    dst = tmp_path / "dst"

    tool.main(source_dataset, dst, gop=GOP)

    _assert_copied_byte_identical(source_dataset, dst)
    for key in VIDEO_KEYS:
        _assert_valid_output(dst / _video_relpath(key), width=SIZE, height=SIZE)

    source_info = json.loads((source_dataset / INFO_JSON).read_text())
    info = json.loads((dst / INFO_JSON).read_text())
    assert {k: v for k, v in info.items() if k != "features"} == {
        k: v for k, v in source_info.items() if k != "features"
    }
    for key, feature in source_info["features"].items():
        if feature["dtype"] != "video":
            assert info["features"][key] == feature

    left = info["features"][INFO_KEY]
    assert left["shape"] == [SIZE, SIZE, 3]
    assert left["info"] == {
        "video.height": SIZE,
        "video.width": SIZE,
        "video.fps": FPS,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
        "video.channels": 3,
    }
    right = info["features"][VIDEO_INFO_KEY]
    assert right["shape"] == [SIZE, SIZE, 3]
    assert right["video_info"] == {
        "video.fps": FPS,
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "has_audio": False,
    }

    provenance = _provenance(dst)
    assert provenance.startswith("# Re-encode provenance\n")
    assert "FAILED" not in provenance
    assert f"- Source: `{source_dataset.resolve()}`" in provenance
    assert "- ffmpeg: `ffmpeg version" in provenance
    assert "- `--scale`: `None`" in provenance
    assert "- `--crf`: `14`" in provenance
    assert f"- `--gop`: `{GOP}`" in provenance
    assert "- `--preset`: `medium`" in provenance
    assert "- `--workers`: `4`" in provenance
    assert "- `--overwrite`: `False`" in provenance
    assert "-c:v libx264 -preset medium -crf 14 -g 15 -bf 0 -pix_fmt yuv420p -movflags +faststart" in provenance
    for key in VIDEO_KEYS:
        assert f"| {_video_relpath(key)} | {FRAMES} | {FRAMES} | {SIZE}x{SIZE} |" in provenance


def test_reencode_with_scale_updates_shape_and_dimensions(source_dataset: pathlib.Path, tmp_path: pathlib.Path):
    dst = tmp_path / "dst"

    tool.main(source_dataset, dst, scale=64, gop=GOP)

    _assert_copied_byte_identical(source_dataset, dst)
    for key in VIDEO_KEYS:
        _assert_valid_output(dst / _video_relpath(key), width=64, height=64)

    info = json.loads((dst / INFO_JSON).read_text())
    left = info["features"][INFO_KEY]
    assert left["shape"] == [64, 64, 3]
    assert left["info"]["video.height"] == 64
    assert left["info"]["video.width"] == 64
    assert left["info"]["video.codec"] == "h264"
    assert left["info"]["video.fps"] == FPS
    right = info["features"][VIDEO_INFO_KEY]
    assert right["shape"] == [64, 64, 3]
    assert "video.height" not in right["video_info"]
    assert "video.width" not in right["video_info"]
    assert right["video_info"]["video.codec"] == "h264"

    provenance = _provenance(dst)
    assert "- `--scale`: `64`" in provenance
    assert "-vf scale=64:-2 -an -c:v libx264" in provenance
    for key in VIDEO_KEYS:
        assert f"| {_video_relpath(key)} | {FRAMES} | {FRAMES} | 64x64 |" in provenance


def test_non_square_scale_keeps_aspect_ratio_with_even_height(tmp_path: pathlib.Path):
    src = _write_dataset(tmp_path / "src", NON_SQUARE)
    dst = tmp_path / "dst"

    tool.main(src, dst, scale=64, gop=GOP)  # 160x96 -> 64x38.4, and scale=64:-2 rounds to an even 38

    _assert_copied_byte_identical(src, dst)
    for key in VIDEO_KEYS:
        _assert_valid_output(dst / _video_relpath(key), width=64, height=38)

    info = json.loads((dst / INFO_JSON).read_text())
    for key in VIDEO_KEYS:
        assert info["features"][key]["shape"] == [38, 64, 3]
    assert info["features"][INFO_KEY]["info"]["video.height"] == 38
    assert info["features"][INFO_KEY]["info"]["video.width"] == 64
    for key in VIDEO_KEYS:
        assert f"| {_video_relpath(key)} | {FRAMES} | {FRAMES} | 64x38 |" in _provenance(dst)


@pytest.mark.parametrize(
    ("width", "height", "scale"),
    [(160, 96, 64), (128, 72, 16), (128, 72, 48), (300, 100, 64), (128, 128, 64)],
    ids=["160x96->64", "128x72->16 (half rounds away from zero)", "128x72->48", "300x100->64", "square"],
)
def test_expected_output_size_matches_ffmpeg_scale_filter(tmp_path: pathlib.Path, width: int, height: int, scale: int):
    source = tmp_path / "source.mp4"
    _write_source_video(source, width=width, height=height, frames=1)
    settings = tool.EncodeSettings(scale=scale, crf=14, gop=GOP, preset="ultrafast")
    output = tmp_path / "output.mp4"

    tool.encode_video(source, output, ffmpeg="ffmpeg", settings=settings)

    probe = _probe(output)
    assert probe.width == scale
    assert probe.height % 2 == 0
    assert tool.expected_output_size(tool.probe_video(source), scale) == (probe.width, probe.height)


@pytest.mark.parametrize(
    ("first_scale", "second_scale", "kept_size", "expected_size"),
    [(64, None, "64x64", f"{SIZE}x{SIZE}"), (None, 64, f"{SIZE}x{SIZE}", "64x64")],
    ids=["scaled mirror re-run without --scale", "unscaled mirror re-run with --scale"],
)
def test_kept_outputs_of_another_size_fail_verification_until_overwritten(
    source_dataset: pathlib.Path,
    tmp_path: pathlib.Path,
    first_scale: int | None,
    second_scale: int | None,
    kept_size: str,
    expected_size: str,
):
    dst = tmp_path / "dst"
    tool.main(source_dataset, dst, scale=first_scale, gop=GOP)

    with pytest.raises(tool.VideoVerificationError, match="2 of 2 videos failed") as excinfo:
        tool.main(source_dataset, dst, scale=second_scale, gop=GOP)
    message = str(excinfo.value)
    for key in VIDEO_KEYS:
        assert f"{dst / _video_relpath(key)}: size {kept_size}, expected {expected_size}" in message
    # The failed run must not leave the earlier run's info.json or provenance behind as if it had succeeded.
    assert (dst / INFO_JSON).read_bytes() == (source_dataset / INFO_JSON).read_bytes()
    assert _provenance(dst).startswith("# Re-encode provenance: FAILED\n")

    tool.main(source_dataset, dst, scale=second_scale, gop=GOP, overwrite=True)
    healed_width, healed_height = (int(part) for part in expected_size.split("x"))
    for key in VIDEO_KEYS:
        _assert_valid_output(dst / _video_relpath(key), width=healed_width, height=healed_height)
    info = json.loads((dst / INFO_JSON).read_text())
    for key in VIDEO_KEYS:
        assert info["features"][key]["shape"] == [healed_height, healed_width, 3]
    assert info["features"][INFO_KEY]["info"]["video.height"] == healed_height
    assert info["features"][INFO_KEY]["info"]["video.width"] == healed_width
    provenance = _provenance(dst)
    assert provenance.startswith("# Re-encode provenance\n")
    assert f"- `--scale`: `{second_scale}`" in provenance
    assert "(2 encoded in this run, 0 kept from an earlier run)" in provenance


def test_single_frame_videos_round_trip(tmp_path: pathlib.Path):
    src = _write_dataset(tmp_path / "src", ONE_FRAME)
    dst = tmp_path / "dst"

    tool.main(src, dst, gop=GOP)

    _assert_copied_byte_identical(src, dst)
    for key in VIDEO_KEYS:
        _assert_valid_output(dst / _video_relpath(key), width=SIZE, height=SIZE, frames=1)
    provenance = _provenance(dst)
    assert "- Videos: 2 (2 encoded in this run, 0 kept from an earlier run)" in provenance
    for key in VIDEO_KEYS:
        assert f"| {_video_relpath(key)} | 1 | 1 | {SIZE}x{SIZE} |" in provenance


def test_two_episodes_mirror_every_parquet_and_video(tmp_path: pathlib.Path):
    src = _write_dataset(tmp_path / "src", TWO_EPISODES)
    dst = tmp_path / "dst"

    tool.main(src, dst, gop=GOP)

    _assert_copied_byte_identical(src, dst)
    for episode in range(TWO_EPISODES.episodes):
        parquet = _parquet_relpath(episode)
        assert (dst / parquet).read_bytes() == (src / parquet).read_bytes()
        assert pq.read_table(dst / parquet).column("episode_index").to_pylist() == [episode] * FRAMES
    assert len(TWO_EPISODES.videos) == 4
    for relative in TWO_EPISODES.videos:
        _assert_valid_output(dst / relative, width=SIZE, height=SIZE)
    provenance = _provenance(dst)
    assert "- Videos: 4 (4 encoded in this run, 0 kept from an earlier run)" in provenance
    for relative in TWO_EPISODES.videos:
        assert f"| {relative} | {FRAMES} | {FRAMES} | {SIZE}x{SIZE} |" in provenance


def test_existing_outputs_are_kept_unless_overwrite(source_dataset: pathlib.Path, tmp_path: pathlib.Path):
    dst = tmp_path / "dst"
    tool.main(source_dataset, dst, gop=GOP)
    output = dst / _video_relpath(INFO_KEY)
    old_ns = 1_000_000_000 * 10**9  # 2001-09-09, clearly not "now"
    os.utime(output, ns=(old_ns, old_ns))

    tool.main(source_dataset, dst, gop=GOP)
    assert output.stat().st_mtime_ns == old_ns
    assert "(0 encoded in this run, 2 kept from an earlier run)" in _provenance(dst)

    tool.main(source_dataset, dst, gop=GOP, overwrite=True)
    assert output.stat().st_mtime_ns != old_ns
    assert "(2 encoded in this run, 0 kept from an earlier run)" in _provenance(dst)
    _assert_valid_output(output, width=SIZE, height=SIZE)


def test_truncated_output_fails_verification_until_overwritten(source_dataset: pathlib.Path, tmp_path: pathlib.Path):
    dst = tmp_path / "dst"
    tool.main(source_dataset, dst, gop=GOP)
    victim = dst / _video_relpath(INFO_KEY)
    payload = victim.read_bytes()
    victim.write_bytes(payload[: len(payload) // 2])

    with pytest.raises(tool.VideoVerificationError, match="1 of 2 videos failed") as excinfo:
        tool.main(source_dataset, dst, gop=GOP)
    assert str(victim) in str(excinfo.value)
    assert _video_relpath(VIDEO_INFO_KEY) not in str(excinfo.value)

    tool.main(source_dataset, dst, gop=GOP, overwrite=True)
    _assert_valid_output(victim, width=SIZE, height=SIZE)


def test_long_gop_b_frame_output_fails_verification(source_dataset: pathlib.Path, tmp_path: pathlib.Path):
    dst = tmp_path / "dst"
    tool.main(source_dataset, dst, gop=GOP)
    victim = dst / _video_relpath(VIDEO_INFO_KEY)
    shutil.copyfile(source_dataset / _video_relpath(VIDEO_INFO_KEY), victim)

    with pytest.raises(tool.VideoVerificationError) as excinfo:
        tool.main(source_dataset, dst, gop=GOP)
    message = str(excinfo.value)
    assert str(victim) in message
    assert "B-frames" in message
    assert f"exceeds gop {GOP}" in message


def test_failed_run_records_the_failure_in_provenance(source_dataset: pathlib.Path, tmp_path: pathlib.Path):
    dst = tmp_path / "dst"
    tool.main(source_dataset, dst, gop=GOP)
    good_provenance = _provenance(dst)
    victim = dst / _video_relpath(INFO_KEY)
    payload = victim.read_bytes()
    victim.write_bytes(payload[: len(payload) // 2])

    with pytest.raises(tool.VideoVerificationError, match="recorded in"):
        tool.main(source_dataset, dst, gop=GOP)

    provenance = _provenance(dst)
    assert provenance != good_provenance
    lines = provenance.splitlines()
    assert lines[0] == "# Re-encode provenance: FAILED"
    assert "**This run FAILED: 1 of 2 videos did not pass verification" in lines[2]
    banner = "\n".join(lines[:8])
    assert f"- {victim}: " in banner
    assert str(dst / _video_relpath(VIDEO_INFO_KEY)) not in banner
    assert f"| {_video_relpath(INFO_KEY)} | {FRAMES} | FAILED | FAILED |" in provenance
    assert f"| {_video_relpath(VIDEO_INFO_KEY)} | {FRAMES} | {FRAMES} | {SIZE}x{SIZE} |" in provenance
    # The destination's info.json is the untouched source copy, not the rewritten one of the earlier run.
    assert (dst / INFO_JSON).read_bytes() == (source_dataset / INFO_JSON).read_bytes()

    tool.main(source_dataset, dst, gop=GOP, overwrite=True)
    provenance = _provenance(dst)
    assert provenance.startswith("# Re-encode provenance\n")
    assert "FAILED" not in provenance


def test_stale_provenance_is_removed_before_encoding(
    source_dataset: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    dst = tmp_path / "dst"
    tool.main(source_dataset, dst, gop=GOP)
    assert (dst / "REENCODE_PROVENANCE.md").is_file()

    def broken_encode(*_args, **_kwargs):
        raise tool.ReencodeError("ffmpeg failed for the test")

    monkeypatch.setattr(tool, "encode_video", broken_encode)
    with pytest.raises(tool.ReencodeError, match="ffmpeg failed for the test"):
        tool.main(source_dataset, dst, gop=GOP, overwrite=True)
    assert not (dst / "REENCODE_PROVENANCE.md").exists()


def test_missing_ffmpeg_is_a_clear_error(
    source_dataset: pathlib.Path, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(tool.shutil, "which", lambda _name: None)

    with pytest.raises(tool.ReencodeError, match="ffmpeg was not found on PATH"):
        tool.main(source_dataset, tmp_path / "dst", gop=GOP)


@pytest.mark.parametrize(
    ("arguments", "match"),
    [
        ({"scale": 63}, "--scale must be a positive even width"),
        ({"scale": 0}, "--scale must be a positive even width"),
        ({"crf": 99}, r"--crf must be within \[0, 51\]"),
        ({"gop": 0}, "--gop must be at least 1"),
        ({"preset": ""}, "--preset must not be empty"),
        ({"workers": 0}, "--workers must be at least 1"),
    ],
    ids=["odd scale", "zero scale", "crf out of range", "zero gop", "empty preset", "zero workers"],
)
def test_invalid_arguments_are_reencode_errors(
    source_dataset: pathlib.Path, tmp_path: pathlib.Path, arguments: dict, match: str
):
    """Every user mistake is a ReencodeError, which the CLI logs as one line and turns into exit 1."""
    dst = tmp_path / "dst"

    with pytest.raises(tool.ReencodeError, match=match):
        tool.main(source_dataset, dst, **{"gop": GOP, **arguments})
    assert not dst.exists()
