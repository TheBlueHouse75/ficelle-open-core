"""Reject unsafe working directories before starting Python helpers."""
import os
import stat
from types import SimpleNamespace

import pytest

from ficelle import runtime_paths


@pytest.mark.parametrize("mode", [0o700, 0o755])
def test_runtime_home_accepts_private_or_readable_directory(tmp_path, mode):
    home = tmp_path / "home"
    home.mkdir(mode=mode)
    home.chmod(mode)
    assert runtime_paths.validated_runtime_home(home) == home.absolute()
    assert stat.S_IMODE(home.stat().st_mode) == mode


def test_runtime_home_creates_missing_directory_privately(tmp_path):
    home = tmp_path / "new"
    assert runtime_paths.validated_runtime_home(home) == home
    assert stat.S_IMODE(home.stat().st_mode) == 0o700


def test_runtime_home_replaces_filesystem_diagnostics_with_a_path_free_message(tmp_path):
    parent = tmp_path / "file"
    parent.touch()

    with pytest.raises(runtime_paths.RuntimeHomeError, match="Check its permissions") as refused:
        runtime_paths.validated_runtime_home(parent / "home")

    assert str(parent) not in str(refused.value)


@pytest.mark.parametrize("kind", ["group-writable", "other-writable", "symlink", "file", "foreign-owner"])
def test_runtime_home_refuses_unsafe_existing_directory(tmp_path, monkeypatch, kind):
    home = tmp_path / "home"
    home.mkdir()
    if kind == "symlink":
        link = tmp_path / "link"
        link.symlink_to(home, target_is_directory=True)
        home = link
    elif kind == "file":
        home = tmp_path / "file"
        home.touch()
    elif kind == "foreign-owner":
        info = home.lstat()
        monkeypatch.setattr(runtime_paths, "os", SimpleNamespace(
            path=os.path, lstat=lambda path: SimpleNamespace(st_mode=info.st_mode, st_uid=123),
            getuid=lambda: 456,
        ))
    else:
        home.chmod(0o775 if kind == "group-writable" else 0o757)
    with pytest.raises(OSError, match="Ficelle home must"):
        runtime_paths.validated_runtime_home(home)


def test_runtime_home_without_getuid_skips_owner_and_mode_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime_paths, "os", SimpleNamespace(
        path=os.path, lstat=lambda path: SimpleNamespace(st_mode=stat.S_IFDIR | 0o777),
    ))
    assert runtime_paths.validated_runtime_home(tmp_path) == tmp_path
    monkeypatch.setattr(runtime_paths, "os", SimpleNamespace(
        path=os.path, lstat=lambda path: SimpleNamespace(st_mode=stat.S_IFLNK | 0o777),
    ))
    with pytest.raises(OSError, match="symbolic link"):
        runtime_paths.validated_runtime_home(tmp_path)


def test_relative_ficelle_home_resolves_to_an_absolute_path(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    paths = runtime_paths.RuntimePaths.from_env(environ={"FICELLE_HOME": "relhome"})

    assert paths.ficelle_home == tmp_path / "relhome"
    assert paths.config_path == tmp_path / "relhome" / "config.json"
