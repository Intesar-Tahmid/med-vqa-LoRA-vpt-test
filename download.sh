#!/bin/bash
# Simple Google Drive ZIP downloader → extracts into current directory
# Usage:
#   ./setup_from_drive.sh "<drive_link_or_file_id>"
# or:
#   GDRIVE_LINK="<drive_link_or_file_id>" ./setup_from_drive.sh

set -euo pipefail

say()  { echo -e "\033[1;34m$*\033[0m"; }
err()  { echo -e "\033[1;31m$*\033[0m" >&2; }
need() { command -v "$1" >/dev/null 2>&1; }

DATA="$PWD"
GDRIVE_LINK="${GDRIVE_LINK:-${1:-}}"
[ -n "$GDRIVE_LINK" ] || { err "❌ Missing Google Drive URL or file ID."; exit 1; }

# --- ensure unzip and gdown ---
if ! need unzip; then
  say "Installing unzip..."
  if need apt-get; then sudo apt-get update -y && sudo apt-get install -y unzip
  elif need dnf; then sudo dnf install -y unzip
  elif need yum; then sudo yum install -y unzip
  elif need pacman; then sudo pacman -Sy --noconfirm unzip
  elif need zypper; then sudo zypper install -y unzip
  elif need apk; then sudo apk add --no-cache unzip
  else err "Install unzip manually."; exit 1; fi
fi

if ! need gdown; then
  say "Installing gdown..."
  if need pip3; then pip3 install --user gdown
  elif need pip; then pip install --user gdown
  else err "pip not found. Please install Python3 & pip."; exit 1; fi
  export PATH="$HOME/.local/bin:$PATH"
fi

# --- extract file ID from Google Drive URL ---
extract_file_id() {
  local url="$1"
  if [[ "$url" =~ /file/d/([a-zA-Z0-9_-]+) ]]; then
    echo "${BASH_REMATCH[1]}"
  elif [[ "$url" =~ ^[a-zA-Z0-9_-]+$ ]]; then
    echo "$url"
  else
    err "Invalid Google Drive URL format"
    return 1
  fi
}

# --- download and extract ---
OUT_ZIP="$DATA/dataset.zip"
say "Downloading ZIP from Google Drive..."

# Extract file ID from URL
FILE_ID=$(extract_file_id "$GDRIVE_LINK")
[ -n "$FILE_ID" ] || { err "Could not extract file ID from URL"; exit 1; }

say "File ID: $FILE_ID"

# Try different download methods
if gdown "https://drive.google.com/uc?id=$FILE_ID" -O "$OUT_ZIP" --fuzzy; then
  say "✅ Downloaded successfully"
elif gdown "$FILE_ID" -O "$OUT_ZIP" --fuzzy; then
  say "✅ Downloaded successfully (direct ID)"
else
  err "❌ Download failed with all methods"
  exit 1
fi

[ -f "$OUT_ZIP" ] || { err "Download failed - file not found."; exit 1; }

# Check if file is valid ZIP
if ! unzip -t "$OUT_ZIP" >/dev/null 2>&1; then
  err "❌ Downloaded file is not a valid ZIP archive"
  rm -f "$OUT_ZIP"
  exit 1
fi

say "Extracting ZIP into: $DATA"
if unzip -o "$OUT_ZIP" -d "$DATA"; then
  say "✅ Extraction successful"
else
  err "❌ Extraction failed"
  exit 1
fi

# Optional: delete after extraction
rm -f "$OUT_ZIP"

say "✅ Done! Files extracted to: $DATA"