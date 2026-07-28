#!/bin/zsh
set -u

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

show_error() {
  local message="$1"
  /usr/bin/osascript -e "display dialog \"$message\" buttons {\"知道了\"} default button 1 with icon stop" >/dev/null 2>&1 || true
  print "$message"
  exit 1
}

if [[ ! -f "$PROJECT_DIR/pyproject.toml" \
  || ! -f "$PROJECT_DIR/start-submd.command" \
  || ! -d "$PROJECT_DIR/src/submd" ]]; then
  show_error "安全检查失败：当前文件夹不像完整的 SubMD 项目，未执行删除。"
fi

confirmation=$(
  /usr/bin/osascript -e '
    button returned of (display dialog "确定把整个 SubMD 项目移到废纸篓吗？\n\n这会移除程序、.env 和 API Key、字幕结果、视频缓存、检查点以及 Python 虚拟环境。\n\nPython、FFmpeg 和 Deno 不会被删除。" buttons {"取消", "移到废纸篓"} default button "取消" cancel button "取消" with icon caution)
  ' 2>/dev/null
) || exit 0

if [[ "$confirmation" != "移到废纸篓" ]]; then
  exit 0
fi

if command -v lsof >/dev/null 2>&1; then
  for pid in $(lsof -tiTCP:8765 -sTCP:LISTEN 2>/dev/null); do
    process_dir="$(lsof -a -p "$pid" -d cwd -Fn 2>/dev/null | sed -n 's/^n//p')"
    if [[ "$process_dir" == "$PROJECT_DIR" ]]; then
      kill -TERM "$pid" 2>/dev/null || true
    fi
  done
fi

PARENT_DIR="$(dirname "$PROJECT_DIR")"
escaped_project="${PROJECT_DIR//\\/\\\\}"
escaped_project="${escaped_project//\"/\\\"}"
cd "$PARENT_DIR" || show_error "无法进入项目的上级目录，未执行删除。"

if ! /usr/bin/osascript -e "tell application \"Finder\" to delete POSIX file \"$escaped_project\"" >/dev/null 2>&1; then
  show_error "无法将项目移到废纸篓，请关闭正在使用项目文件的程序后重试。"
fi

/usr/bin/osascript -e 'display notification "SubMD 已移到废纸篓" with title "SubMD"' >/dev/null 2>&1 || true
