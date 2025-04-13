"""
Helper tools for getting Qward version information
"""

import os
import subprocess

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def _minimal_ext_cmd(cmd):
    # construct minimal environment
    env = {}
    for k in ["SYSTEMROOT", "PATH"]:
        v = os.environ.get(k)
        if v is not None:
            env[k] = v
    # LANGUAGE is used on win32
    env["LANGUAGE"] = "C"
    env["LANG"] = "C"
    env["LC_ALL"] = "C"
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        cwd=os.path.join(os.path.dirname(ROOT_DIR)),
        encoding="utf-8",
    ) as proc:
        out = proc.communicate()[0]
        if proc.returncode > 0:
            raise OSError
        return out


def git_version():
    """Get the current git head sha1."""
    try:
        out = _minimal_ext_cmd(["git", "rev-parse", "HEAD"])
        git_revision = out.strip()
    except OSError:
        git_revision = "Unknown"
    return git_revision


with open(os.path.join(ROOT_DIR, "VERSION.txt"), "r", encoding="utf-8") as version_file:
    VERSION = version_file.read().strip()


def get_version_info():
    """Get the full version string."""
    full_version = VERSION

    if not os.path.exists(os.path.join(os.path.dirname(ROOT_DIR), ".git")):
        return full_version

    try:
        release = _minimal_ext_cmd(["git", "tag", "-l", "--points-at", "HEAD"])
    except Exception:  # pylint: disable=broad-except
        return full_version

    if not release:
        git_revision = git_version()
        full_version += ".dev0+" + git_revision[:7]

    return full_version


__version__ = get_version_info()
