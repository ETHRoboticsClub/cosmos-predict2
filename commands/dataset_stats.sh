#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.env.paths" ]]; then
  source "${REPO_ROOT}/.env.paths"
fi

DATASET_PATH="${DATASET_PATH:-/nvme/datasets/teleop/preprocessed}"
FFPROBE_BIN="${FFPROBE_BIN:-ffprobe}"
DATASET_STATS_CSV="${DATASET_STATS_CSV:-}"

usage() {
  cat <<EOF
Usage: DATASET_PATH=/path/to/preprocessed $0

Environment:
  DATASET_PATH        Cosmos-style dataset root with videos/ and metas/.
                      Default: /nvme/datasets/teleop/preprocessed
  FFPROBE_BIN         ffprobe binary to use for video durations.
                      Default: ffprobe
  DATASET_STATS_CSV   Optional path to also write per-task CSV stats.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

VIDEOS_DIR="${DATASET_PATH}/videos"
METAS_DIR="${DATASET_PATH}/metas"

if [[ ! -d "${VIDEOS_DIR}" ]]; then
  echo "Missing videos directory: ${VIDEOS_DIR}" >&2
  exit 2
fi

if [[ ! -d "${METAS_DIR}" ]]; then
  echo "Missing metas directory: ${METAS_DIR}" >&2
  exit 2
fi

python3 - "${DATASET_PATH}" "${FFPROBE_BIN}" "${DATASET_STATS_CSV}" <<'PY'
import csv
import shutil
import statistics
import struct
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

dataset_path = Path(sys.argv[1])
ffprobe_bin = sys.argv[2]
csv_path = Path(sys.argv[3]) if sys.argv[3] else None
videos_dir = dataset_path / "videos"
metas_dir = dataset_path / "metas"


def iter_boxes(fp, end):
    while fp.tell() < end:
        start = fp.tell()
        header = fp.read(8)
        if len(header) < 8:
            return
        size, box_type = struct.unpack(">I4s", header)
        header_size = 8
        if size == 1:
            large = fp.read(8)
            if len(large) < 8:
                return
            size = struct.unpack(">Q", large)[0]
            header_size = 16
        elif size == 0:
            size = end - start
        if size < header_size:
            return
        yield box_type, start, size, header_size
        fp.seek(start + size)


def read_mvhd_duration(path):
    file_size = path.stat().st_size
    with path.open("rb") as fp:
        for box_type, start, size, header_size in iter_boxes(fp, file_size):
            if box_type != b"moov":
                continue
            moov_end = start + size
            fp.seek(start + header_size)
            for child_type, child_start, child_size, child_header_size in iter_boxes(fp, moov_end):
                if child_type != b"mvhd":
                    continue
                fp.seek(child_start + child_header_size)
                version_flags = fp.read(4)
                if len(version_flags) < 4:
                    return None
                version = version_flags[0]
                if version == 1:
                    payload = fp.read(28)
                    if len(payload) < 28:
                        return None
                    timescale = struct.unpack(">I", payload[16:20])[0]
                    duration = struct.unpack(">Q", payload[20:28])[0]
                else:
                    payload = fp.read(16)
                    if len(payload) < 16:
                        return None
                    timescale = struct.unpack(">I", payload[8:12])[0]
                    duration = struct.unpack(">I", payload[12:16])[0]
                if timescale == 0:
                    return None
                return duration / timescale
    return None


def ffprobe_duration(path):
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    if result.returncode != 0:
        return None
    try:
        duration = float(result.stdout.strip())
    except ValueError:
        return None
    return duration if duration >= 0 else None


has_ffprobe = shutil.which(ffprobe_bin) is not None


def video_duration(path):
    if has_ffprobe:
        duration = ffprobe_duration(path)
        if duration is not None:
            return duration, "ffprobe"
    duration = read_mvhd_duration(path)
    if duration is not None:
        return duration, "mp4-header"
    return None, "failed"


def percentile(values, q):
    if not values:
        return None
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def fmt_minutes(seconds):
    return f"{seconds / 60.0:.2f}"


def fmt_seconds(seconds):
    return "n/a" if seconds is None else f"{seconds:.1f}"


tasks = defaultdict(list)
missing_videos = []
empty_metas = []
probe_failures = []
duration_sources = defaultdict(int)

for meta_path in sorted(metas_dir.glob("*.txt")):
    task = meta_path.read_text(encoding="utf-8", errors="replace").strip()
    if not task:
        empty_metas.append(meta_path)
        task = "<empty instruction>"

    video_path = videos_dir / f"{meta_path.stem}.mp4"
    if not video_path.exists():
        missing_videos.append(video_path)
        continue

    duration, source = video_duration(video_path)
    duration_sources[source] += 1
    if duration is None:
        probe_failures.append(video_path)
        continue

    tasks[task].append(duration)

rows = []
for task, durations in tasks.items():
    total_seconds = sum(durations)
    rows.append(
        {
            "task": task,
            "episodes": len(durations),
            "minutes": total_seconds / 60.0,
            "total_seconds": total_seconds,
            "mean_seconds": statistics.fmean(durations),
            "median_seconds": statistics.median(durations),
            "min_seconds": min(durations),
            "p25_seconds": percentile(durations, 25),
            "p75_seconds": percentile(durations, 75),
            "max_seconds": max(durations),
        }
    )

rows.sort(key=lambda row: (-row["minutes"], row["task"]))
all_durations = [duration for durations in tasks.values() for duration in durations]

print(f"Dataset: {dataset_path}")
print(f"Videos:  {videos_dir}")
print(f"Metas:   {metas_dir}")
print()
print("Overall")
print(f"  tasks:          {len(rows)}")
print(f"  episodes:       {len(all_durations)}")
print(f"  minutes:        {fmt_minutes(sum(all_durations))}")
print(f"  mean episode:   {fmt_seconds(statistics.fmean(all_durations) if all_durations else None)}s")
print(f"  median episode: {fmt_seconds(statistics.median(all_durations) if all_durations else None)}s")
print(f"  min/p25/p75/max:{fmt_seconds(percentile(all_durations, 0))}s / {fmt_seconds(percentile(all_durations, 25))}s / {fmt_seconds(percentile(all_durations, 75))}s / {fmt_seconds(percentile(all_durations, 100))}s")
print(f"  duration source:{', '.join(f'{key}={value}' for key, value in sorted(duration_sources.items())) or 'none'}")
print(f"  missing videos: {len(missing_videos)}")
print(f"  empty metas:    {len(empty_metas)}")
print(f"  probe failures: {len(probe_failures)}")
print()

if rows:
    headers = [
        "minutes",
        "episodes",
        "mean_s",
        "median_s",
        "min_s",
        "p25_s",
        "p75_s",
        "max_s",
        "task",
    ]
    table_rows = [
        [
            f"{row['minutes']:.2f}",
            str(row["episodes"]),
            f"{row['mean_seconds']:.1f}",
            f"{row['median_seconds']:.1f}",
            f"{row['min_seconds']:.1f}",
            f"{row['p25_seconds']:.1f}",
            f"{row['p75_seconds']:.1f}",
            f"{row['max_seconds']:.1f}",
            row["task"],
        ]
        for row in rows
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in table_rows))
        for index in range(len(headers))
    ]
    print("Per task")
    print("  " + "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  " + "  ".join("-" * width for width in widths))
    for row in table_rows:
        print("  " + "  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

if csv_path:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp,
            fieldnames=[
                "task",
                "episodes",
                "minutes",
                "total_seconds",
                "mean_seconds",
                "median_seconds",
                "min_seconds",
                "p25_seconds",
                "p75_seconds",
                "max_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"Wrote CSV: {csv_path}")

if missing_videos:
    print()
    print("First missing videos:")
    for path in missing_videos[:10]:
        print(f"  {path}")

if probe_failures:
    print()
    print("First probe failures:")
    for path in probe_failures[:10]:
        print(f"  {path}")
PY
