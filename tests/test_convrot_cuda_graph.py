"""Independent CUDA graph ownership, including cache-eviction regressions."""

import gc

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_graph_streams_are_unique_and_survive_other_graph_eviction():
    from modules.whisper.convrot.cuda_graph import CudaGraph

    # More streams than PyTorch's usual 32-stream round-robin pool.
    graphs = [CudaGraph() for _ in range(40)]
    assert len({g.stream.cuda_stream for g in graphs}) == len(graphs)
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b = torch.randn_like(a)
    expected = a @ b
    with graphs[0].capture():
        actual = a @ b
    with graphs[1].capture():
        other = a @ b
    graphs[1].replay()
    finished = torch.cuda.Event()
    finished.record()
    graphs[1].close()
    assert finished.query()
    del other
    gc.collect()
    torch.cuda.empty_cache()
    graphs[0].replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, expected)
    for graph in graphs:
        graph.close()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_graph_closes_on_owning_device_and_restores_callers_device():
    from modules.whisper.convrot.cuda_graph import CudaGraph

    with torch.cuda.device(0):
        source = torch.ones(256, device="cuda")
        graph = CudaGraph()
        with graph.capture():
            output = source + 3
        graph.replay()
    with torch.cuda.device(1):
        graph.close()
        assert torch.cuda.current_device() == 1
        assert graph._handle is None
    torch.testing.assert_close(output, torch.full_like(output, 4))
