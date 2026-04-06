import ctypes
import importlib
import logging
import os
import platform
import site
import sys
from pathlib import Path


logger = logging.getLogger(__name__)

_DLL_DIRECTORY_HANDLES = []
_PRELOADED_LIBRARY_HANDLES = []
_CUDA_RUNTIME_CONFIGURED = False

_NVIDIA_RUNTIME_MODULES = (
    "nvidia.cublas.bin",
    "nvidia.cublas.lib",
    "nvidia.cudnn.bin",
    "nvidia.cudnn.lib",
    "nvidia.cuda_runtime.bin",
    "nvidia.cuda_runtime.lib",
)

_WINDOWS_LIBRARY_PATTERNS = (
    "cudart64_*.dll",
    "cublasLt64_*.dll",
    "cublas64_*.dll",
    "cudnn*.dll",
    "nvblas64_*.dll",
)

_LINUX_LIBRARY_PATTERNS = (
    "libcudart.so*",
    "libcublasLt.so*",
    "libcublas.so*",
    "libcudnn*.so*",
)


def _dedupe_existing_dirs(paths):
    seen = set()
    result = []

    for path in paths:
        if not path:
            continue

        normalized = os.path.normpath(str(path))
        if not os.path.isdir(normalized):
            continue

        key = normalized.lower() if os.name == "nt" else normalized
        if key in seen:
            continue

        seen.add(key)
        result.append(normalized)

    return result


def _module_search_dirs(module):
    candidates = []

    module_file = getattr(module, "__file__", None)
    if module_file:
        module_path = Path(module_file).resolve()
        candidates.append(str(module_path if module_path.is_dir() else module_path.parent))

    module_path_attr = getattr(module, "__path__", None)
    if module_path_attr:
        candidates.extend(str(Path(path).resolve()) for path in module_path_attr)

    module_spec = getattr(module, "__spec__", None)
    search_locations = getattr(module_spec, "submodule_search_locations", None)
    if search_locations:
        candidates.extend(str(Path(path).resolve()) for path in search_locations)

    return _dedupe_existing_dirs(candidates)


def _discover_dirs_from_nvidia_modules():
    candidates = []

    for module_name in _NVIDIA_RUNTIME_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue

        candidates.extend(_module_search_dirs(module))

    return _dedupe_existing_dirs(candidates)


def _candidate_site_package_roots():
    candidates = []

    try:
        candidates.extend(site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(user_site)
    except Exception:
        pass

    candidates.extend(path for path in sys.path if "site-packages" in str(path) or "dist-packages" in str(path))
    return _dedupe_existing_dirs(candidates)


def _is_within_current_env(path):
    try:
        return Path(path).resolve().is_relative_to(Path(sys.prefix).resolve())
    except Exception:
        return False


def _discover_dirs_from_site_packages(search_roots=None):
    candidates = []

    for root in search_roots or _candidate_site_package_roots():
        nvidia_root = Path(root) / "nvidia"
        if not nvidia_root.is_dir():
            continue

        for package_name in ("cublas", "cudnn", "cuda_runtime"):
            package_root = nvidia_root / package_name
            for subdir in ("bin", "lib"):
                candidate = package_root / subdir
                if candidate.is_dir():
                    candidates.append(str(candidate.resolve()))

    return _dedupe_existing_dirs(candidates)


def _discover_dirs_from_current_env_site_packages():
    current_env_roots = [root for root in _candidate_site_package_roots() if _is_within_current_env(root)]
    return _discover_dirs_from_site_packages(search_roots=current_env_roots)


def _discover_dirs_from_cuda_env():
    candidates = []
    system = platform.system()

    for env_name, env_value in os.environ.items():
        if not env_value:
            continue

        if env_name in {"CUDA_PATH", "CUDA_HOME", "CUDNN_HOME"} or env_name.startswith("CUDA_PATH_V"):
            base_path = Path(env_value)
            if system == "Windows":
                candidates.extend([base_path / "bin", base_path / "lib" / "x64"])
            elif system == "Linux":
                candidates.extend([base_path / "lib64", base_path / "lib"])

    return _dedupe_existing_dirs(candidates)


def discover_cuda_runtime_dirs():
    if platform.system() not in {"Linux", "Windows"}:
        return []

    discovered = _dedupe_existing_dirs(
        _discover_dirs_from_nvidia_modules() + _discover_dirs_from_current_env_site_packages()
    )
    if discovered:
        return discovered

    discovered = []
    discovered.extend(_discover_dirs_from_cuda_env())
    discovered.extend(_discover_dirs_from_site_packages())
    return _dedupe_existing_dirs(discovered)


def _prepend_env_dirs(variable_name, directories):
    separator = ";" if platform.system() == "Windows" else ":"
    existing = os.environ.get(variable_name, "")
    merged = list(directories)

    if existing:
        merged.extend(part for part in existing.split(separator) if part)

    os.environ[variable_name] = separator.join(_dedupe_existing_dirs(merged))


def _register_windows_dll_directories(directories):
    if platform.system() != "Windows" or not hasattr(os, "add_dll_directory"):
        return

    for directory in directories:
        try:
            _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(directory))
        except OSError:
            continue


def _iter_library_files(directories):
    patterns = _WINDOWS_LIBRARY_PATTERNS if platform.system() == "Windows" else _LINUX_LIBRARY_PATTERNS
    seen = set()

    for pattern in patterns:
        for directory in directories:
            for candidate in sorted(Path(directory).glob(pattern), key=lambda path: (len(path.name), path.name.lower())):
                if not candidate.is_file():
                    continue

                resolved = str(candidate.resolve())
                key = resolved.lower() if os.name == "nt" else resolved
                if key in seen:
                    continue

                seen.add(key)
                yield resolved


def _preload_libraries(directories):
    loader = ctypes.WinDLL if platform.system() == "Windows" else ctypes.CDLL
    mode = getattr(ctypes, "RTLD_GLOBAL", 0)

    for library_path in _iter_library_files(directories):
        try:
            if platform.system() == "Windows":
                _PRELOADED_LIBRARY_HANDLES.append(loader(library_path))
            else:
                _PRELOADED_LIBRARY_HANDLES.append(loader(library_path, mode=mode))
        except OSError:
            continue


def enable_cuda_runtime_autodiscovery():
    global _CUDA_RUNTIME_CONFIGURED

    if _CUDA_RUNTIME_CONFIGURED:
        return discover_cuda_runtime_dirs()

    directories = discover_cuda_runtime_dirs()
    if not directories:
        return []

    if platform.system() == "Windows":
        _prepend_env_dirs("PATH", directories)
        _register_windows_dll_directories(directories)
    elif platform.system() == "Linux":
        _prepend_env_dirs("LD_LIBRARY_PATH", directories)

    _preload_libraries(directories)
    _CUDA_RUNTIME_CONFIGURED = True

    logger.info("CUDA runtime library paths configured: %s", ", ".join(directories))
    return directories
