from app.utils.metrics import MetricsCollector


def test_metrics_collector_records_stage_and_details():
    collector = MetricsCollector()

    with collector.stage("analysis", candidates=4) as details:
        details["selected"] = 2

    metrics = collector.snapshot(source_duration_seconds=30)

    assert metrics.total_elapsed_seconds >= 0
    assert metrics.source_duration_seconds == 30
    assert metrics.speed_vs_realtime is not None
    assert metrics.resource_sampling_enabled is False
    assert len(metrics.stages) == 1
    assert metrics.stages[0].name == "analysis"
    assert metrics.stages[0].peak_memory_mb > 0
    assert metrics.stages[0].details == {
        "candidates": 4,
        "selected": 2,
    }
