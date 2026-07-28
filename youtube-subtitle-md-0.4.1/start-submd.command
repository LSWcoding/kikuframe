#!/bin/zsh
set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

show_error() {
  local message="$1"
  /usr/bin/osascript -e "display dialog \"$message\" buttons {\"知道了\"} default button 1 with icon stop" >/dev/null 2>&1 || true
  print "$message"
  read "?按回车键关闭窗口…"
  exit 1
}

PYTHON_BIN=""
for candidate in /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 python3.12; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON_BIN="$candidate"
    break
  fi
done

if [[ -z "$PYTHON_BIN" ]]; then
  show_error "没有找到 Python 3.12。请先安装 Python 3.12，再次双击启动。"
fi

for dependency in ffmpeg ffprobe deno; do
  if ! command -v "$dependency" >/dev/null 2>&1; then
    show_error "没有找到 $dependency。请先运行 brew install ffmpeg deno，再次双击启动。"
  fi
done

if [[ ! -x ".venv/bin/python" ]]; then
  "$PYTHON_BIN" -m venv .venv
fi

if ! .venv/bin/python -c "import submd" >/dev/null 2>&1; then
  print "首次启动，正在安装 SubMD…"
  .venv/bin/python -m pip install -U pip
  .venv/bin/python -m pip install -e .
else
  .venv/bin/python -m pip install -e . --quiet
fi

exec .venv/bin/submd ui
