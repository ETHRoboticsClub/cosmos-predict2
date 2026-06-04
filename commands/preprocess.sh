#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

source "${REPO_ROOT}/.env.paths"

RAW_ROOT="${RAW_ROOT:-/nvme/datasets/teleop/raw}"
OUT_ROOT="${OUT_ROOT:-/nvme/datasets/teleop/preprocessed}"
MODE="${MODE:-symlink}"
CAMERA_GLOB="${CAMERA_GLOB:-camera_top-images-rgb.mp4}"
OVERWRITE="${OVERWRITE:-1}"
TARGET_FPS="${TARGET_FPS-10}"
VIDEO_ENCODER="${VIDEO_ENCODER:-auto}"
FFMPEG_BIN="${FFMPEG_BIN:-ffmpeg}"
VIDEO_CRF="${VIDEO_CRF:-18}"
NVENC_CQ="${NVENC_CQ:-23}"
NVENC_PRESET="${NVENC_PRESET:-p4}"
FFMPEG_LOGLEVEL="${FFMPEG_LOGLEVEL:-error}"
WORKERS="${WORKERS:-32}"

VIDEOS_DIR="${OUT_ROOT}/videos"
METAS_DIR="${OUT_ROOT}/metas"

if [[ "${MODE}" != "symlink" && "${MODE}" != "copy" ]]; then
  echo "MODE must be either 'symlink' or 'copy'." >&2
  exit 2
fi

if [[ -n "${TARGET_FPS}" && ! "${TARGET_FPS}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "TARGET_FPS must be numeric, or empty to disable FPS downsampling." >&2
  exit 2
fi

if [[ "${VIDEO_ENCODER}" != "auto" && "${VIDEO_ENCODER}" != "h264_nvenc" && "${VIDEO_ENCODER}" != "libx264" ]]; then
  echo "VIDEO_ENCODER must be auto, h264_nvenc, or libx264." >&2
  exit 2
fi

if [[ ! "${WORKERS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "WORKERS must be a positive integer." >&2
  exit 2
fi

if [[ ! -d "${RAW_ROOT}" ]]; then
  echo "Raw dataset root does not exist: ${RAW_ROOT}" >&2
  exit 2
fi

if [[ "${OVERWRITE}" == "1" ]]; then
  if [[ -z "${OUT_ROOT}" || "${OUT_ROOT}" == "/" ]]; then
    echo "Refusing to remove unsafe OUT_ROOT: ${OUT_ROOT}" >&2
    exit 2
  fi
  echo "OVERWRITE=1: removing existing output root: ${OUT_ROOT}"
  rm -rf "${OUT_ROOT}"
fi

mkdir -p "${VIDEOS_DIR}" "${METAS_DIR}"

echo "Preprocessing teleop dataset"
echo "  raw root:        ${RAW_ROOT}"
echo "  output root:     ${OUT_ROOT}"
echo "  mode:            ${MODE}"
echo "  target fps:      ${TARGET_FPS:-native}"
echo "  encoder:         ${VIDEO_ENCODER}"
echo "  ffmpeg loglevel: ${FFMPEG_LOGLEVEL}"
echo "  camera glob:     ${CAMERA_GLOB}"
echo "  workers:         ${WORKERS}"
echo "Scanning for session_meta.json files..."
mapfile -d '' META_PATHS < <(find "${RAW_ROOT}" -mindepth 3 -maxdepth 3 -type f -name session_meta.json -print0 | sort -z)
total_metas="${#META_PATHS[@]}"
echo "Found ${total_metas} metadata files."

created=0
skipped=0
missing_videos=0
missing_instruction=0
STATUS_DIR="$(mktemp -d "${OUT_ROOT}/.preprocess-status.XXXXXX")"
trap 'rm -rf "${STATUS_DIR}"' EXIT

extract_instruction() {
  local meta_path="$1"
  python3 - "$meta_path" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
try:
    data = json.loads(path.read_text())
except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    sys.exit(1)

instruction = str(data.get("instruction", "")).strip()
if instruction:
    print(instruction)
PY
}

nvenc_available() {
  if ! "${FFMPEG_BIN}" -hide_banner -encoders 2>/dev/null | grep -q "h264_nvenc"; then
    return 1
  fi

  "${FFMPEG_BIN}" -hide_banner -loglevel error -y \
    -f lavfi -i "color=size=16x16:rate=1:duration=1" \
    -frames:v 1 -an -c:v h264_nvenc -preset "${NVENC_PRESET}" -cq "${NVENC_CQ}" -b:v 0 \
    -f null - >/dev/null 2>&1
}

resolve_video_encoder() {
  if [[ "${VIDEO_ENCODER}" != "auto" ]]; then
    echo "${VIDEO_ENCODER}"
    return 0
  fi

  if nvenc_available; then
    echo "h264_nvenc"
  else
    echo "libx264"
  fi
}

write_video() {
  local src="$1"
  local dst="$2"

  if [[ -e "${dst}" || -L "${dst}" ]]; then
    if [[ "${OVERWRITE}" == "1" ]]; then
      rm -f "${dst}"
    else
      return 1
    fi
  fi

  if [[ -n "${TARGET_FPS}" ]]; then
    if ! command -v "${FFMPEG_BIN}" >/dev/null 2>&1; then
      echo "TARGET_FPS=${TARGET_FPS}, but '${FFMPEG_BIN}' was not found." >&2
      echo "Install ffmpeg on the preprocessing node/container, or run TARGET_FPS= to disable downsampling." >&2
      exit 2
    fi

    local encoder
    local tmp_dst
    local -a encode_args
    encoder="${RESOLVED_VIDEO_ENCODER:-$(resolve_video_encoder)}"
    tmp_dst="${dst%.mp4}.tmp.$$.mp4"

    if [[ "${encoder}" == "h264_nvenc" ]]; then
      encode_args=(-c:v h264_nvenc -preset "${NVENC_PRESET}" -cq "${NVENC_CQ}" -b:v 0)
    else
      encode_args=(-c:v libx264 -preset veryfast -crf "${VIDEO_CRF}")
    fi

    "${FFMPEG_BIN}" -hide_banner -loglevel "${FFMPEG_LOGLEVEL}" -y -i "${src}"       -vf "fps=${TARGET_FPS}" -an "${encode_args[@]}"       -pix_fmt yuv420p -movflags +faststart "${tmp_dst}"

    mv "${tmp_dst}" "${dst}"
    return 0
  fi

  if [[ "${MODE}" == "copy" ]]; then
    cp "${src}" "${dst}"
  else
    ln -s "${src}" "${dst}"
  fi
}

process_meta() {
  local index="$1"
  local meta_path="$2"
  local episode_dir
  local episode_name
  local date_name
  local instruction
  local video_path
  local video_file
  local camera_name
  local sample_name
  local out_video
  local out_meta
  local -a videos
  local created_count=0
  local skipped_count=0
  local missing_videos_count=0
  local missing_instruction_count=0

  episode_dir="$(dirname "${meta_path}")"
  episode_name="$(basename "${episode_dir}")"
  date_name="$(basename "$(dirname "${episode_dir}")")"

  if ! instruction="$(extract_instruction "${meta_path}")"; then
    echo "Skipping ${episode_dir}: failed to parse session_meta.json" >&2
    missing_instruction_count=1
    printf "%s %s %s %s\n" "${created_count}" "${skipped_count}" "${missing_videos_count}" "${missing_instruction_count}" > "${STATUS_DIR}/${index}.status"
    return 0
  fi

  if [[ -z "${instruction}" ]]; then
    echo "Skipping ${episode_dir}: no instruction in session_meta.json" >&2
    missing_instruction_count=1
    printf "%s %s %s %s\n" "${created_count}" "${skipped_count}" "${missing_videos_count}" "${missing_instruction_count}" > "${STATUS_DIR}/${index}.status"
    return 0
  fi

  shopt -s nullglob
  videos=("${episode_dir}"/${CAMERA_GLOB})
  shopt -u nullglob

  if (( ${#videos[@]} == 0 )); then
    echo "Skipping ${episode_dir}: no videos matching ${CAMERA_GLOB}" >&2
    missing_videos_count=1
    printf "%s %s %s %s\n" "${created_count}" "${skipped_count}" "${missing_videos_count}" "${missing_instruction_count}" > "${STATUS_DIR}/${index}.status"
    return 0
  fi

  for video_path in "${videos[@]}"; do
    video_file="$(basename "${video_path}")"
    camera_name="${video_file%-images-rgb.mp4}"
    sample_name="${date_name}_${episode_name}_${camera_name}"
    out_video="${VIDEOS_DIR}/${sample_name}.mp4"
    out_meta="${METAS_DIR}/${sample_name}.txt"

    echo "[${index}/${total_metas}] Writing ${sample_name}.mp4 from ${video_path}"

    if write_video "${video_path}" "${out_video}"; then
      printf "%s\n" "${instruction}" > "${out_meta}"
      created_count=$((created_count + 1))
    else
      skipped_count=$((skipped_count + 1))
    fi
  done

  printf "%s %s %s %s\n" "${created_count}" "${skipped_count}" "${missing_videos_count}" "${missing_instruction_count}" > "${STATUS_DIR}/${index}.status"
}

calculate_video_seconds() {
  if command -v ffprobe >/dev/null 2>&1; then
    local failed=0
    local duration
    local video_path
    local durations=()

    while IFS= read -r -d '' video_path <&3; do
      if ! duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "${video_path}" 2>/dev/null)"; then
        failed=$((failed + 1))
        continue
      fi

      if [[ "${duration}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
        durations+=("${duration}")
      else
        failed=$((failed + 1))
      fi
    done 3< <(find "${VIDEOS_DIR}" -maxdepth 1 \( -type f -o -type l \) -name '*.mp4' -print0)

    python3 - "${failed}" "ffprobe" "${durations[@]}" <<'PYPERCENT'
import sys

failed = int(sys.argv[1])
source = sys.argv[2]
durations = sorted(float(value) for value in sys.argv[3:])


def percentile(values, q):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def fmt(value):
    return "unknown" if value is None else f"{value:.1f}"

percentiles = [fmt(percentile(durations, q)) for q in (0, 25, 50, 75, 100)]
total_seconds = int(sum(durations))
print(":".join([str(total_seconds), str(failed), source, *percentiles]))
PYPERCENT
    return 0
  fi

  python3 - "${VIDEOS_DIR}" <<'PYMP4'
import struct
import sys
from pathlib import Path


def iter_boxes(fp, end):
    while fp.tell() < end:
        start = fp.tell()
        header = fp.read(8)
        if len(header) < 8:
            return
        size, box_type = struct.unpack('>I4s', header)
        header_size = 8
        if size == 1:
            large = fp.read(8)
            if len(large) < 8:
                return
            size = struct.unpack('>Q', large)[0]
            header_size = 16
        elif size == 0:
            size = end - start
        if size < header_size:
            return
        yield box_type, start, size, header_size
        fp.seek(start + size)


def read_mvhd_duration(path):
    file_size = path.stat().st_size
    with path.open('rb') as fp:
        for box_type, start, size, header_size in iter_boxes(fp, file_size):
            if box_type != b'moov':
                continue
            moov_start = start + header_size
            moov_end = start + size
            fp.seek(moov_start)
            for child_type, child_start, child_size, child_header_size in iter_boxes(fp, moov_end):
                if child_type != b'mvhd':
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
                    timescale = struct.unpack('>I', payload[16:20])[0]
                    duration = struct.unpack('>Q', payload[20:28])[0]
                else:
                    payload = fp.read(16)
                    if len(payload) < 16:
                        return None
                    timescale = struct.unpack('>I', payload[8:12])[0]
                    duration = struct.unpack('>I', payload[12:16])[0]
                if timescale == 0:
                    return None
                return duration / timescale
    return None


def percentile(values, q):
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q / 100.0
    lower = int(pos)
    upper = min(lower + 1, len(values) - 1)
    weight = pos - lower
    return values[lower] * (1.0 - weight) + values[upper] * weight


def fmt(value):
    return "unknown" if value is None else f"{value:.1f}"


durations = []
failed = 0
for video_path in sorted(Path(sys.argv[1]).glob('*.mp4')):
    try:
        seconds = read_mvhd_duration(video_path)
    except Exception:
        seconds = None
    if seconds is None:
        failed += 1
    else:
        durations.append(seconds)

durations.sort()
percentiles = [fmt(percentile(durations, q)) for q in (0, 25, 50, 75, 100)]
total_seconds = int(sum(durations))
print(":".join([str(total_seconds), str(failed), "mp4-header", *percentiles]))
PYMP4
}

RESOLVED_VIDEO_ENCODER="${VIDEO_ENCODER}"
if [[ -n "${TARGET_FPS}" ]]; then
  if ! command -v "${FFMPEG_BIN}" >/dev/null 2>&1; then
    echo "TARGET_FPS=${TARGET_FPS}, but '${FFMPEG_BIN}' was not found." >&2
    echo "Install ffmpeg on the preprocessing node/container, or run TARGET_FPS= to disable downsampling." >&2
    exit 2
  fi

  RESOLVED_VIDEO_ENCODER="$(resolve_video_encoder)"
  echo "Resolved video encoder: ${RESOLVED_VIDEO_ENCODER}"
fi

if (( WORKERS == 1 )); then
  for index in "${!META_PATHS[@]}"; do
    process_meta "$((index + 1))" "${META_PATHS[$index]}"
  done
else
  export RAW_ROOT OUT_ROOT MODE CAMERA_GLOB OVERWRITE TARGET_FPS VIDEO_ENCODER RESOLVED_VIDEO_ENCODER FFMPEG_BIN VIDEO_CRF NVENC_CQ NVENC_PRESET FFMPEG_LOGLEVEL
  export VIDEOS_DIR METAS_DIR STATUS_DIR total_metas
  export -f extract_instruction nvenc_available resolve_video_encoder write_video process_meta

  if (( total_metas > 0 )); then
    for index in "${!META_PATHS[@]}"; do
      printf "%s\0%s\0" "$((index + 1))" "${META_PATHS[$index]}"
    done | xargs -0 -n 2 -P "${WORKERS}" bash -euo pipefail -c 'process_meta "$1" "$2"' _
  fi
fi

for status_file in "${STATUS_DIR}"/*.status; do
  [[ -e "${status_file}" ]] || continue
  read -r created_count skipped_count missing_videos_count missing_instruction_count < "${status_file}"
  created=$((created + created_count))
  skipped=$((skipped + skipped_count))
  missing_videos=$((missing_videos + missing_videos_count))
  missing_instruction=$((missing_instruction + missing_instruction_count))
done

duration_summary="$(calculate_video_seconds)"
video_minutes="unknown"
video_probe_failures="0"
video_seconds_p0="unknown"
video_seconds_p25="unknown"
video_seconds_p50="unknown"
video_seconds_p75="unknown"
video_seconds_p100="unknown"
IFS=':' read -r total_video_seconds video_probe_failures duration_source   video_seconds_p0 video_seconds_p25 video_seconds_p50 video_seconds_p75 video_seconds_p100 <<< "${duration_summary}"
video_minutes="$(python3 -c 'import sys; print(f"{int(sys.argv[1]) / 60:.1f}")' "${total_video_seconds}")"

echo "Preprocessed teleop dataset:"
echo "  raw root:        ${RAW_ROOT}"
echo "  output root:     ${OUT_ROOT}"
echo "  mode:            ${MODE}"
echo "  target fps:      ${TARGET_FPS:-native}"
echo "  encoder:         ${RESOLVED_VIDEO_ENCODER}"
echo "  camera glob:     ${CAMERA_GLOB}"
echo "  workers:         ${WORKERS}"
echo "  created:         ${created}"
echo "  skipped:         ${skipped}"
echo "  no videos:       ${missing_videos}"
echo "  no instruction:  ${missing_instruction}"
echo "  video minutes:   ${video_minutes} (${duration_source})"
echo "  video seconds:   0th=${video_seconds_p0}, 25th=${video_seconds_p25}, 50th=${video_seconds_p50}, 75th=${video_seconds_p75}, 100th=${video_seconds_p100}"
if [[ "${video_probe_failures}" != "unknown" && "${video_probe_failures}" != "0" ]]; then
  echo "  probe failures:  ${video_probe_failures}"
fi
echo
echo "Dataset is ready for VideoDataset at:"
echo "  ${OUT_ROOT}"
