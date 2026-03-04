"""
External application integration helpers.

The UI cannot replace a professional DCC tool. When deeper playback/editing is
needed, this module launches external software such as Blender.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple


def resolve_blender_executable(explicit_path: Optional[str] = None) -> Optional[str]:
    """Resolve a Blender executable from an explicit path, env vars, or common paths."""
    candidates = []
    if explicit_path:
        candidates.append(explicit_path)
    for env_name in ("BLENDER_EXE", "BLENDER_BIN"):
        if os.environ.get(env_name):
            candidates.append(os.environ[env_name])

    which_path = shutil.which("blender")
    if which_path:
        candidates.append(which_path)

    windows_candidates = [
        "/mnt/c/Program Files/Blender Foundation/Blender 4.2/blender.exe",
        "/mnt/c/Program Files/Blender Foundation/Blender 4.1/blender.exe",
        "/mnt/c/Program Files/Blender Foundation/Blender 4.0/blender.exe",
        "/mnt/c/Program Files/Blender Foundation/Blender 3.6/blender.exe",
    ]
    candidates.extend(windows_candidates)

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def _wsl_to_windows_path(path: str) -> Optional[str]:
    try:
        result = subprocess.run(
            ["wslpath", "-w", path],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def launch_blender_with_file(file_path: str, blender_executable: Optional[str] = None) -> Tuple[bool, str]:
    """
    Launch Blender with a BVH file.

    On WSL, a Windows Blender executable is opened through `cmd.exe /C start`.
    """
    target = Path(file_path)
    if not target.exists():
        return False, f"文件不存在: {file_path}"

    blender_path = resolve_blender_executable(blender_executable)
    if not blender_path:
        return False, "未找到 Blender。请安装 Blender，或提供其可执行文件路径。"

    try:
        if blender_path.lower().endswith(".exe") and os.name != "nt":
            exe_win = _wsl_to_windows_path(blender_path)
            file_win = _wsl_to_windows_path(str(target))
            if not exe_win or not file_win:
                return False, "WSL 路径转换失败，无法调用 Windows Blender。"
            command = f"start \"\" \"{exe_win}\" \"{file_win}\""
            subprocess.Popen(["cmd.exe", "/C", command])
        else:
            subprocess.Popen([blender_path, str(target)])
        return True, f"已调用 Blender 打开: {target.name}"
    except Exception as exc:
        return False, f"Blender 启动失败: {exc}"
