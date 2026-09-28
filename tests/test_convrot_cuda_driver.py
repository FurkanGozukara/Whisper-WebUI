"""Driver loading conventions can be checked without a CUDA context or Windows host."""

import ctypes
from types import SimpleNamespace

import pytest

from modules.whisper.convrot import cuda_graph


@pytest.mark.parametrize("platform,expected_loader,expected_name", [
    ("win32", "WinDLL", "nvcuda.dll"),
    ("linux", "CDLL", "libcuda.so.1"),
])
def test_cuda_driver_uses_platform_library_and_pointer_signatures(monkeypatch, platform, expected_loader, expected_name):
    calls = []

    def load(loader, name):
        calls.append((loader, name))
        return SimpleNamespace(cuStreamCreate=lambda *_args: 0, cuStreamDestroy_v2=lambda *_args: 0)

    fake_ctypes = SimpleNamespace(
        WinDLL=lambda name: load("WinDLL", name),
        CDLL=lambda name: load("CDLL", name),
        POINTER=ctypes.POINTER,
        c_void_p=ctypes.c_void_p,
        c_uint=ctypes.c_uint,
        c_int=ctypes.c_int,
    )
    monkeypatch.setattr(cuda_graph, "sys", SimpleNamespace(platform=platform))
    monkeypatch.setattr(cuda_graph, "ctypes", fake_ctypes)
    cuda_graph._cuda_driver.cache_clear()
    try:
        driver = cuda_graph._cuda_driver()
        assert calls == [(expected_loader, expected_name)]
        assert cuda_graph._cuda_driver() is driver
        assert driver.cuStreamCreate.argtypes == [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
        assert driver.cuStreamCreate.restype is ctypes.c_int
        assert driver.cuStreamDestroy_v2.argtypes == [ctypes.c_void_p]
        assert driver.cuStreamDestroy_v2.restype is ctypes.c_int
    finally:
        # A simulated library must never escape into a later real GPU test.
        cuda_graph._cuda_driver.cache_clear()
