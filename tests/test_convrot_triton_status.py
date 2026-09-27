"""Reports of the ConvRot engine's Triton tuning (no GPU needed)."""

from modules.whisper.convrot import triton_status


class FakeAutotuner:
    """The two Triton Autotuner methods the reports wrap, with a switchable disk cache."""

    def __init__(self):
        self.disk = set()
        self.benchmarked = []

    def _bench(self, *args, config, **meta):
        self.benchmarked.append(config)
        return [1.0, 1.0, 1.0]

    def check_disk_cache(self, tuning_key, configs, bench_fn):
        if tuning_key in self.disk:
            return True
        bench_fn()
        self.disk.add(tuning_key)
        return False

    def run(self, tuning_key, configs):
        def benchmark():
            for config in configs:
                self._bench(config=config)

        return self.check_disk_cache(tuning_key, configs, benchmark)


def test_tuning_and_reuse_are_reported_and_counted():
    tuner = triton_status.instrument_autotuner(FakeAutotuner(), "INT8 GEMM")
    messages = []
    before = triton_status.counts()

    with triton_status.report_to(messages.append):
        assert tuner.run((4096, 5120, 1280, True), ["a", "b", "c"]) is False
        assert tuner.run((4096, 5120, 1280, True), ["a", "b", "c"]) is True

    assert tuner.benchmarked == ["a", "b", "c"]
    assert any("tuning #" in message and "INT8 GEMM for M<=4096 N=5120 K=1280" in message and "3 configurations" in message
               for message in messages)
    assert any("tuned in" in message for message in messages)
    summary = triton_status.summary_since(before)
    assert "1 shape(s) tuned now" in summary and "1 reused" in summary


def test_reused_only_summary_and_no_listener_after_block():
    tuner = FakeAutotuner()
    tuner.disk.add((16, 1280, 1280))
    triton_status.instrument_autotuner(tuner, "weight-only INT8 GEMM")
    messages = []
    before = triton_status.counts()

    with triton_status.report_to(messages.append):
        assert tuner.run((16, 1280, 1280), ["a"]) is True
    tuner.run((64, 1280, 1280), ["a"])

    assert messages == []
    assert triton_status.summary_since(before).startswith("Triton kernel cache: 1 shape(s) tuned now")
    assert triton_status.summary_since(triton_status.counts()) is None


def test_instrumenting_twice_wraps_once():
    tuner = FakeAutotuner()
    triton_status.instrument_autotuner(tuner, "INT8 GEMM")
    wrapped = tuner.check_disk_cache
    triton_status.instrument_autotuner(tuner, "INT8 GEMM")
    assert tuner.check_disk_cache is wrapped
