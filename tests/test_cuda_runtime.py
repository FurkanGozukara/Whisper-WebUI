import types
from pathlib import Path

from modules.utils import cuda_runtime


def test_discover_cuda_runtime_dirs_finds_windows_namespace_package_dirs(tmp_path, monkeypatch):
    cublas_bin = tmp_path / "nvidia" / "cublas" / "bin"
    cublas_bin.mkdir(parents=True)

    fake_module = types.SimpleNamespace(
        __file__=None,
        __path__=[str(cublas_bin)],
        __spec__=types.SimpleNamespace(submodule_search_locations=[str(cublas_bin)]),
    )

    def fake_import(name):
        if name == "nvidia.cublas.bin":
            return fake_module
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(cuda_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setattr(cuda_runtime.importlib, "import_module", fake_import)
    monkeypatch.setattr(cuda_runtime.site, "getsitepackages", lambda: [])
    monkeypatch.setattr(cuda_runtime.site, "getusersitepackages", lambda: str(tmp_path / "missing"))
    monkeypatch.setattr(cuda_runtime.sys, "path", [])
    monkeypatch.setattr(cuda_runtime, "_discover_dirs_from_cuda_env", lambda: [])

    assert cuda_runtime.discover_cuda_runtime_dirs() == [str(cublas_bin)]


def test_enable_cuda_runtime_autodiscovery_updates_windows_path_and_adds_dll_dirs(tmp_path, monkeypatch):
    cuda_root = tmp_path / "cuda"
    cuda_bin = cuda_root / "bin"
    cuda_bin.mkdir(parents=True)
    (cuda_bin / "cublas64_12.dll").write_bytes(b"")

    monkeypatch.setattr(cuda_runtime.platform, "system", lambda: "Windows")
    monkeypatch.setenv("PATH", "C:\\existing")
    monkeypatch.setattr(cuda_runtime, "_discover_dirs_from_nvidia_modules", lambda: [])
    monkeypatch.setattr(cuda_runtime, "_discover_dirs_from_current_env_site_packages", lambda: [str(cuda_bin)])
    monkeypatch.setattr(cuda_runtime, "_discover_dirs_from_cuda_env", lambda: [])

    added_dirs = []
    loaded_libraries = []

    monkeypatch.setattr(cuda_runtime.os, "add_dll_directory", lambda path: added_dirs.append(path) or path)
    monkeypatch.setattr(cuda_runtime.ctypes, "WinDLL", lambda path: loaded_libraries.append(path) or path)
    monkeypatch.setattr(cuda_runtime, "_CUDA_RUNTIME_CONFIGURED", False)
    cuda_runtime._DLL_DIRECTORY_HANDLES.clear()
    cuda_runtime._PRELOADED_LIBRARY_HANDLES.clear()

    directories = cuda_runtime.enable_cuda_runtime_autodiscovery()

    assert directories == [str(cuda_bin)]
    assert os_path_startswith(cuda_runtime.os.environ["PATH"], str(cuda_bin))
    assert added_dirs == [str(cuda_bin)]
    assert loaded_libraries == [str((cuda_bin / "cublas64_12.dll").resolve())]


def test_enable_cuda_runtime_autodiscovery_updates_linux_loader_paths(tmp_path, monkeypatch):
    cublas_lib = tmp_path / "nvidia" / "cublas" / "lib"
    cublas_lib.mkdir(parents=True)
    (cublas_lib / "libcublas.so.12").write_bytes(b"")

    monkeypatch.setattr(cuda_runtime.platform, "system", lambda: "Linux")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/existing")
    monkeypatch.setattr(cuda_runtime, "discover_cuda_runtime_dirs", lambda: [str(cublas_lib)])

    loaded_libraries = []

    monkeypatch.setattr(cuda_runtime.ctypes, "CDLL", lambda path, mode=0: loaded_libraries.append((path, mode)) or path)
    monkeypatch.setattr(cuda_runtime, "_CUDA_RUNTIME_CONFIGURED", False)
    cuda_runtime._PRELOADED_LIBRARY_HANDLES.clear()

    directories = cuda_runtime.enable_cuda_runtime_autodiscovery()

    assert directories == [str(cublas_lib)]
    assert cuda_runtime.os.environ["LD_LIBRARY_PATH"].startswith(str(cublas_lib))
    assert loaded_libraries[0][0] == str((cublas_lib / "libcublas.so.12").resolve())


def os_path_startswith(value, expected, separator=";"):
    assert value
    return value.split(separator)[0] == expected
