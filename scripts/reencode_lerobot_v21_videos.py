"""Mirror a LeRobot v2.1 dataset with its videos re-encoded as short-GOP x264 streams.

lerobot's torchcodec decode path (``seek_mode="approximate"``) returns wrong
frames near GOP tails of long-GOP, B-frame streams (the HEVC exports use
~250-frame GOPs) and then fails its 1e-4 s timestamp-tolerance check. Dense
keyframes and no B-frames make the index-to-pts mapping exact and random
access fast, so training reads a derived x264 copy instead of the export.

The tool mirrors SRC into DST:

* every file outside ``videos/`` is copied byte-identically (``data/`` parquet,
  ``meta/``); ``meta/info.json`` is the only rewritten file;
* every ``videos/**/*.mp4`` is re-encoded with ffmpeg, preserving the source
  frame rate::

      -an -c:v libx264 -preset PRESET -crf CRF -g GOP -bf 0 -pix_fmt yuv420p -movflags +faststart

  ``--scale N`` adds ``-vf scale=N:-2`` (width N, height keeps the aspect
  ratio, so a square source becomes N x N);
* every output is verified with PyAV: same frame count and frame rate as the
  source, the expected size (the source size, or with ``--scale`` width N and
  the even aspect-preserving height ffmpeg's ``scale=N:-2`` produces), no
  B-frames, no keyframe gap larger than GOP, h264/yuv420p. Any violation makes
  the tool exit non-zero and name the file;
* ``meta/info.json`` gets ``video.codec``/``video.pix_fmt`` updated for every
  video feature; with ``--scale`` the feature ``shape`` ([H, W, 3]) and the
  ``video.height``/``video.width`` keys are updated as well;
* ``DST/REENCODE_PROVENANCE.md`` records the absolute source path, every
  argument, the ``ffmpeg -version`` first line, and a per-video table. It is
  removed when a run starts; a run that fails verification writes one whose
  first lines say FAILED and list the offending files, so a destination left
  behind by a failed run never passes for a finished mirror.

Existing outputs are kept (but still verified, against the current arguments)
unless ``--overwrite``; a mirror made with another ``--scale`` therefore fails
verification until it is re-encoded with ``--overwrite``.

The four derived training copies, from the HF exports under ``HF_LEROBOT_HOME``::

    uv run python scripts/reencode_lerobot_v21_videos.py \\
      --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_curated" \\
      --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_curated_x264"

    uv run python scripts/reencode_lerobot_v21_videos.py \\
      --src "$HF_LEROBOT_HOME/byang11259/cardboard_box_tcp_vid7to54" \\
      --dst "$HF_LEROBOT_HOME/local/cardboard_box_tcp_vid7to54_x264"

    uv run python scripts/reencode_lerobot_v21_videos.py \\
      --src "$HF_LEROBOT_HOME/byang11259/stack_cubes_tcp" \\
      --dst "$HF_LEROBOT_HOME/local/stack_cubes_tcp_x264" --scale 384

    uv run python scripts/reencode_lerobot_v21_videos.py \\
      --src "$HF_LEROBOT_HOME/byang11259/stack_cubes_takes" \\
      --dst "$HF_LEROBOT_HOME/local/stack_cubes_takes_x264" --scale 384

Not reproduced here: the 10-second resegmentation behind
``local/cardboard_box_tcp_curated_10s_x264`` and the manual task-string
relabels of a few episodes (for example vid7to54 episodes 37-39); those were
applied to the source metadata by hand before or after the re-encode.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
import copy
import dataclasses
import datetime
from fractions import Fraction
import itertools
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
from typing import Any

import av
import tyro

logger = logging.getLogger(__name__)

VIDEO_DIR = "videos"
PROVENANCE_NAME = "REENCODE_PROVENANCE.md"
OUTPUT_CODEC = "h264"
OUTPUT_PIX_FMT = "yuv420p"
# Sub-dicts of a video feature that describe the stream. lerobot 0.1.0's
# ``LeRobotDataset.update_video_info`` writes ``info`` (from
# ``video_utils.get_video_info``); the portable exporter writes ``video_info``.
VIDEO_INFO_KEYS = ("info", "video_info")


class ReencodeError(RuntimeError):
    """A condition that must fail the run: missing ffmpeg, bad inputs, failed encode."""


class VideoVerificationError(ReencodeError):
    """One or more outputs violate the decode constraints training relies on."""


@dataclasses.dataclass(frozen=True)
class EncodeSettings:
    """The libx264 recipe shared by every video of one run."""

    scale: int | None
    crf: int
    gop: int
    preset: str

    def validate(self) -> None:
        """Reject settings ffmpeg would refuse or misuse; raises ``ReencodeError`` so the CLI exits cleanly."""
        if self.scale is not None and (self.scale <= 0 or self.scale % 2):
            raise ReencodeError(
                f"--scale must be a positive even width (yuv420p needs even dimensions), got {self.scale}"
            )
        if not 0 <= self.crf <= 51:
            raise ReencodeError(f"--crf must be within [0, 51], got {self.crf}")
        if self.gop < 1:
            raise ReencodeError(f"--gop must be at least 1, got {self.gop}")
        if not self.preset:
            raise ReencodeError("--preset must not be empty")

    def ffmpeg_args(self) -> list[str]:
        """Output-side ffmpeg arguments (everything after ``-i SRC -map 0:v:0``)."""
        args: list[str] = []
        if self.scale is not None:
            args += ["-vf", f"scale={self.scale}:-2"]
        args += ["-an", "-c:v", "libx264", "-preset", self.preset, "-crf", str(self.crf), "-g", str(self.gop)]
        args += ["-bf", "0", "-pix_fmt", OUTPUT_PIX_FMT, "-movflags", "+faststart"]
        return args


@dataclasses.dataclass(frozen=True)
class VideoJob:
    """One source video and where its re-encoded copy goes."""

    relative_path: pathlib.Path
    source: pathlib.Path
    output: pathlib.Path

    @property
    def video_key(self) -> str:
        # v2.1 layout: videos/chunk-XXX/<video_key>/episode_XXXXXX.mp4
        return self.relative_path.parent.name


@dataclasses.dataclass(frozen=True)
class VideoProbe:
    """Facts about one video collected in a single PyAV decode pass."""

    frames: int
    width: int
    height: int
    codec: str
    pix_fmt: str
    frame_rate: Fraction | None
    b_frames: int
    first_keyframe: int | None
    max_keyframe_gap: int


@dataclasses.dataclass(frozen=True)
class VideoResult:
    job: VideoJob
    source: VideoProbe
    output: VideoProbe | None
    encoded: bool
    error: str | None = None


def find_ffmpeg() -> str:
    binary = shutil.which("ffmpeg")
    if binary is None:
        raise ReencodeError("ffmpeg was not found on PATH; install ffmpeg with libx264 support or add it to PATH")
    return binary


def ffmpeg_version_line(ffmpeg: str) -> str:
    completed = subprocess.run([ffmpeg, "-version"], check=True, capture_output=True, text=True)
    return completed.stdout.splitlines()[0].strip()


def is_video_file(relative_path: pathlib.Path) -> bool:
    return relative_path.parts[:1] == (VIDEO_DIR,) and relative_path.suffix == ".mp4"


def plan_videos(src: pathlib.Path, dst: pathlib.Path) -> list[VideoJob]:
    """List every ``videos/**/*.mp4`` of SRC with its mirrored output path."""
    video_root = src / VIDEO_DIR
    if not video_root.is_dir():
        raise ReencodeError(f"{src} has no {VIDEO_DIR}/ directory; is it a LeRobot v2.1 dataset root?")
    jobs: list[VideoJob] = []
    for source in sorted(video_root.rglob("*.mp4")):
        if source.is_file():
            relative_path = source.relative_to(src)
            jobs.append(VideoJob(relative_path=relative_path, source=source, output=dst / relative_path))
    return jobs


def copy_non_video_files(src: pathlib.Path, dst: pathlib.Path) -> list[pathlib.Path]:
    """Copy every file that is not a ``videos/**/*.mp4`` byte-identically into DST."""
    copied: list[pathlib.Path] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(src)
        if is_video_file(relative_path):
            continue
        target = dst / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(relative_path)
    return copied


def probe_video(path: pathlib.Path) -> VideoProbe:
    """Decode a video once and collect what verification and provenance need."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        frames = 0
        b_frames = 0
        keyframes: list[int] = []
        for index, frame in enumerate(container.decode(stream)):
            frames += 1
            if frame.key_frame:
                keyframes.append(index)
            if int(frame.pict_type) == int(av.video.frame.PictureType.B):
                b_frames += 1
        width, height = stream.width, stream.height
        codec = stream.codec.canonical_name
        pix_fmt = stream.pix_fmt
        frame_rate = stream.base_rate

    gaps = [later - earlier for earlier, later in itertools.pairwise(keyframes)]
    if keyframes:
        gaps.append(frames - keyframes[-1])  # the tail after the last keyframe is a GOP too
    return VideoProbe(
        frames=frames,
        width=width,
        height=height,
        codec=codec,
        pix_fmt=pix_fmt,
        frame_rate=Fraction(frame_rate) if frame_rate is not None else None,
        b_frames=b_frames,
        first_keyframe=keyframes[0] if keyframes else None,
        max_keyframe_gap=max(gaps, default=frames),
    )


def expected_output_size(source: VideoProbe, scale: int | None) -> tuple[int, int]:
    """The (width, height) an output of SOURCE must have: the source size, or ``scale=N:-2`` applied to it.

    ffmpeg's ``-2`` makes the height ``av_rescale(N, source.height, 2 * source.width) * 2``:
    the aspect-preserving height rounded to the nearest even number, halves away
    from zero (so 128x72 scaled to 16 wide is 16x10, not 16x8).
    """
    if scale is None:
        return source.width, source.height
    return scale, 2 * ((scale * source.height + source.width) // (2 * source.width))


def verify_video(path: pathlib.Path, *, source: VideoProbe, gop: int, scale: int | None = None) -> VideoProbe:
    """Check one output against its source; raise ``VideoVerificationError`` naming the file."""
    try:
        output = probe_video(path)
    except (av.FFmpegError, IndexError, OSError) as exc:
        raise VideoVerificationError(f"{path}: cannot decode ({exc})") from exc

    violations: list[str] = []
    if output.frames != source.frames:
        violations.append(f"{output.frames} frames, source has {source.frames}")
    if output.b_frames:
        violations.append(f"{output.b_frames} B-frames")
    if output.first_keyframe != 0:
        violations.append("first frame is not a keyframe")
    if output.max_keyframe_gap > gop:
        violations.append(f"keyframe gap {output.max_keyframe_gap} exceeds gop {gop}")
    if output.codec != OUTPUT_CODEC or output.pix_fmt != OUTPUT_PIX_FMT:
        violations.append(f"codec {output.codec}/{output.pix_fmt}, expected {OUTPUT_CODEC}/{OUTPUT_PIX_FMT}")
    if output.frame_rate != source.frame_rate:
        violations.append(f"frame rate {output.frame_rate}, source has {source.frame_rate}")
    expected_width, expected_height = expected_output_size(source, scale)
    if (output.width, output.height) != (expected_width, expected_height):
        violations.append(f"size {output.width}x{output.height}, expected {expected_width}x{expected_height}")
    if violations:
        raise VideoVerificationError(f"{path}: {'; '.join(violations)}")
    return output


def encode_video(source: pathlib.Path, output: pathlib.Path, *, ffmpeg: str, settings: EncodeSettings) -> None:
    """Re-encode SOURCE into OUTPUT; the file appears only once ffmpeg finished successfully."""
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(output.name + ".partial")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(source), "-map", "0:v:0"]
    command += [*settings.ffmpeg_args(), "-f", "mp4", str(partial)]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        partial.unlink(missing_ok=True)
        raise ReencodeError(f"ffmpeg failed for {source}: {exc.stderr.strip()}") from exc
    os.replace(partial, output)


def reencode_one(job: VideoJob, *, ffmpeg: str, settings: EncodeSettings, overwrite: bool) -> VideoResult:
    """Encode one video unless its output already exists (or OVERWRITE), then verify the output."""
    source = probe_video(job.source)
    encoded = overwrite or not job.output.exists()
    if encoded:
        encode_video(job.source, job.output, ffmpeg=ffmpeg, settings=settings)
    try:
        output = verify_video(job.output, source=source, gop=settings.gop, scale=settings.scale)
    except VideoVerificationError as exc:
        return VideoResult(job=job, source=source, output=None, encoded=encoded, error=str(exc))
    return VideoResult(job=job, source=source, output=output, encoded=encoded)


def run_jobs(
    jobs: list[VideoJob], *, ffmpeg: str, settings: EncodeSettings, overwrite: bool, workers: int
) -> list[VideoResult]:
    """Process JOBS on WORKERS threads; results come back in plan order."""
    results: dict[int, VideoResult] = {}
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(reencode_one, job, ffmpeg=ffmpeg, settings=settings, overwrite=overwrite): index
            for index, job in enumerate(jobs)
        }
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results[futures[future]] = result
                if result.error:
                    logger.error("[%d/%d] FAILED %s", done, len(jobs), result.error)
                    continue
                assert result.output is not None
                logger.info(
                    "[%d/%d] %s %s: %d frames, %dx%d",
                    done,
                    len(jobs),
                    "encoded" if result.encoded else "kept",
                    result.job.relative_path,
                    result.output.frames,
                    result.output.width,
                    result.output.height,
                )
        except BaseException:
            executor.shutdown(wait=False, cancel_futures=True)
            raise
    return [results[index] for index in range(len(jobs))]


def output_dims_by_key(results: list[VideoResult]) -> dict[str, tuple[int, int]]:
    """Map each video key to its (height, width); every video of a key must agree."""
    dims: dict[str, set[tuple[int, int]]] = {}
    for result in results:
        assert result.output is not None
        dims.setdefault(result.job.video_key, set()).add((result.output.height, result.output.width))
    inconsistent = {key: sorted(sizes) for key, sizes in dims.items() if len(sizes) != 1}
    if inconsistent:
        raise ReencodeError(f"videos of one key must share one size, got {inconsistent}")
    return {key: next(iter(sizes)) for key, sizes in dims.items()}


def update_info_json(info: dict[str, Any], *, output_dims: dict[str, tuple[int, int]] | None) -> dict[str, Any]:
    """Return a copy of info.json describing the re-encoded streams.

    ``video.codec``/``video.pix_fmt`` (and ``has_audio``, since audio is dropped)
    are rewritten in every non-empty ``info``/``video_info`` dict of a video
    feature. OUTPUT_DIMS maps video key -> (height, width) and is only given
    when the videos were scaled; it rewrites ``shape`` and, where present,
    ``video.height``/``video.width``.
    """
    updated = copy.deepcopy(info)
    for key, feature in updated["features"].items():
        if feature.get("dtype") != "video":
            continue
        dims = None if output_dims is None else output_dims.get(key)
        if dims is not None:
            feature["shape"] = [*dims, *feature["shape"][2:]]
        for info_key in VIDEO_INFO_KEYS:
            video_info = feature.get(info_key)
            if not isinstance(video_info, dict) or not video_info:
                continue
            video_info["video.codec"] = OUTPUT_CODEC
            video_info["video.pix_fmt"] = OUTPUT_PIX_FMT
            if "video.channels" in video_info:
                video_info["video.channels"] = 3
            if dims is not None:
                if "video.height" in video_info:
                    video_info["video.height"] = dims[0]
                if "video.width" in video_info:
                    video_info["video.width"] = dims[1]
            if video_info.get("has_audio"):
                video_info["has_audio"] = False
                for audio_key in [name for name in video_info if name.startswith("audio.")]:
                    del video_info[audio_key]
    return updated


def warn_on_stale_shapes(info: dict[str, Any], results: list[VideoResult]) -> None:
    """Warn when the source info.json shape disagrees with the decoded source videos."""
    seen: set[str] = set()
    for result in results:
        key = result.job.video_key
        feature = info["features"].get(key)
        if key in seen or feature is None or feature.get("dtype") != "video":
            continue
        seen.add(key)
        declared = tuple(feature["shape"][:2])
        actual = (result.source.height, result.source.width)
        if declared != actual:
            logger.warning(
                "Source info.json declares %s as %s but %s decodes to %s",
                key,
                declared,
                result.job.relative_path,
                actual,
            )


def write_provenance(
    dst: pathlib.Path,
    *,
    src: pathlib.Path,
    arguments: dict[str, Any],
    ffmpeg_version: str,
    ffmpeg_args: list[str],
    results: list[VideoResult],
) -> pathlib.Path:
    """Write DST/REENCODE_PROVENANCE.md and return its path.

    When any result carries an error the file opens with a FAILED banner that
    lists the offending files, so a destination left behind by a failed run
    never passes for a finished mirror.
    """
    failures = [result for result in results if result.error]
    encoded = sum(result.encoded for result in results)
    if failures:
        lines = [
            "# Re-encode provenance: FAILED",
            "",
            f"**This run FAILED: {len(failures)} of {len(results)} videos did not pass verification, "
            "so this directory is not a usable mirror.** Offending files:",
            "",
            *(f"- {result.error}" for result in failures),
            "",
            "Re-run the tool (with `--overwrite` to replace kept outputs) until this banner is gone.",
            "`meta/info.json` is still an unmodified copy of the source's; only a successful run rewrites it.",
        ]
    else:
        lines = [
            "# Re-encode provenance",
            "",
            "Every file outside `videos/` is a byte-identical copy of the source.",
            "`meta/info.json` is the only rewritten file.",
        ]
    lines += [
        "",
        "Generated by `scripts/reencode_lerobot_v21_videos.py`.",
        "",
        f"- Source: `{src}`",
        f"- Destination: `{dst}`",
        f"- Generated: {datetime.datetime.now(datetime.UTC).isoformat(timespec='seconds')}",
        f"- ffmpeg: `{ffmpeg_version}`",
        f"- ffmpeg video arguments: `{' '.join(ffmpeg_args)}`",
        f"- Videos: {len(results)} ({encoded} encoded in this run, {len(results) - encoded} kept from an earlier run)",
        "- Arguments:",
        *(f"  - `--{name.replace('_', '-')}`: `{value}`" for name, value in arguments.items()),
        "",
        "| Video | Source frames | Output frames | Output size (w x h) |",
        "| --- | ---: | ---: | --- |",
    ]
    for result in results:
        if result.output is None:
            lines.append(f"| {result.job.relative_path} | {result.source.frames} | FAILED | FAILED |")
            continue
        lines.append(
            f"| {result.job.relative_path} | {result.source.frames} | {result.output.frames} "
            f"| {result.output.width}x{result.output.height} |"
        )
    path = dst / PROVENANCE_NAME
    path.write_text("\n".join(lines) + "\n")
    return path


def main(
    src: pathlib.Path,
    dst: pathlib.Path,
    *,
    scale: int | None = None,
    crf: int = 14,
    gop: int = 15,
    preset: str = "medium",
    workers: int = 4,
    overwrite: bool = False,
) -> None:
    """Mirror SRC into DST with every video re-encoded as x264 (keyframe gap <= GOP, no B-frames).

    Args:
        src: Source LeRobot v2.1 dataset root.
        dst: Destination root; created if missing. Existing videos are kept unless --overwrite.
        scale: Output width in pixels (height keeps the aspect ratio). Default keeps the source size.
        crf: libx264 constant rate factor; lower is closer to lossless.
        gop: Maximum keyframe interval in frames.
        preset: libx264 speed/quality preset.
        workers: Videos processed concurrently (each ffmpeg is itself multithreaded).
        overwrite: Re-encode videos whose output already exists.
    """
    settings = EncodeSettings(scale=scale, crf=crf, gop=gop, preset=preset)
    settings.validate()
    if workers < 1:
        raise ReencodeError(f"--workers must be at least 1, got {workers}")
    src = src.resolve()
    dst = dst.resolve()
    info_path = src / "meta" / "info.json"
    if not info_path.is_file():
        raise ReencodeError(f"{src} is not a LeRobot v2.1 dataset root: {info_path} is missing")
    if src == dst or src in dst.parents or dst in src.parents:
        raise ReencodeError(f"source and destination must not nest: {src} vs {dst}")
    arguments = {
        "src": src,
        "dst": dst,
        "scale": scale,
        "crf": crf,
        "gop": gop,
        "preset": preset,
        "workers": workers,
        "overwrite": overwrite,
    }

    ffmpeg = find_ffmpeg()
    ffmpeg_version = ffmpeg_version_line(ffmpeg)
    logger.info("Using %s", ffmpeg_version)
    info = json.loads(info_path.read_text())
    jobs = plan_videos(src, dst)
    if not jobs:
        raise ReencodeError(f"{src / VIDEO_DIR} contains no .mp4 files")
    total_videos = info.get("total_videos")
    if total_videos is not None and int(total_videos) != len(jobs):
        logger.warning("info.json declares total_videos=%s but %d videos were found", total_videos, len(jobs))

    dst.mkdir(parents=True, exist_ok=True)
    # An earlier run's provenance would make a destination this run leaves unfinished look complete.
    (dst / PROVENANCE_NAME).unlink(missing_ok=True)
    copied = copy_non_video_files(src, dst)
    logger.info("Copied %d non-video files byte-identically into %s", len(copied), dst)
    logger.info("Processing %d videos on %d workers: %s", len(jobs), workers, " ".join(settings.ffmpeg_args()))

    results = run_jobs(jobs, ffmpeg=ffmpeg, settings=settings, overwrite=overwrite, workers=workers)
    failures = [result.error for result in results if result.error]
    if failures:
        provenance = write_provenance(
            dst,
            src=src,
            arguments=arguments,
            ffmpeg_version=ffmpeg_version,
            ffmpeg_args=settings.ffmpeg_args(),
            results=results,
        )
        raise VideoVerificationError(
            f"{len(failures)} of {len(results)} videos failed verification (recorded in {provenance}):\n"
            + "\n".join(failures)
        )
    warn_on_stale_shapes(info, results)

    output_dims = output_dims_by_key(results)
    updated = update_info_json(info, output_dims=output_dims if scale is not None else None)
    (dst / "meta" / "info.json").write_text(json.dumps(updated, indent=4, ensure_ascii=False) + "\n")
    provenance = write_provenance(
        dst,
        src=src,
        arguments=arguments,
        ffmpeg_version=ffmpeg_version,
        ffmpeg_args=settings.ffmpeg_args(),
        results=results,
    )
    encoded = sum(result.encoded for result in results)
    logger.info(
        "Verified %d videos (%d encoded, %d kept); wrote %s", len(results), encoded, len(results) - encoded, provenance
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    try:
        tyro.cli(main)
    except ReencodeError as error:
        logger.error("%s", error)
        sys.exit(1)
