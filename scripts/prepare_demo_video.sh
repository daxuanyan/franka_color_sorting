#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: bash scripts/prepare_demo_video.sh INPUT_VIDEO SLUG [START_SECONDS] [DURATION_SECONDS]" >&2
  exit 2
fi

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is required. Install it with your system package manager." >&2
  exit 1
fi

input_video=$1
slug=$2
start_seconds=${3:-0}
duration_seconds=${4:-20}

if [[ ! -f "$input_video" ]]; then
  echo "Input video does not exist: $input_video" >&2
  exit 1
fi

if [[ ! "$slug" =~ ^[a-z0-9][a-z0-9_-]*$ ]]; then
  echo "SLUG may contain only lowercase letters, digits, underscores, and hyphens." >&2
  exit 1
fi

output_dir=media/demos
mkdir -p "$output_dir"

ffmpeg -y -ss "$start_seconds" -i "$input_video" -t "$duration_seconds" \
  -vf "scale=1280:-2:flags=lanczos,fps=30" \
  -c:v libx264 -preset medium -crf 24 -pix_fmt yuv420p \
  -movflags +faststart -an "$output_dir/$slug.mp4"

palette_file=$(mktemp --suffix=.png)
trap 'rm -f "$palette_file"' EXIT

ffmpeg -y -ss "$start_seconds" -i "$input_video" -t "$duration_seconds" \
  -vf "fps=10,scale=720:-2:flags=lanczos,palettegen=max_colors=128" \
  "$palette_file"
ffmpeg -y -ss "$start_seconds" -i "$input_video" -t "$duration_seconds" \
  -i "$palette_file" \
  -lavfi "fps=10,scale=720:-2:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer" \
  "$output_dir/$slug.gif"

echo "Created:"
du -h "$output_dir/$slug.mp4" "$output_dir/$slug.gif"
echo "Keep each committed preview small; shorten the duration if the GIF exceeds about 10 MB."
