from datetime import datetime

from astree_eta.board import parse_analysis_statuses, parse_server_snapshot


def test_parse_server_snapshot():
    rows = [
        {
            "processing_count": "3",
            "queued_count": "5",
            "total_mem_used_gb": "120.5",
            "free_mem_gb": "32.0",
            "timestamp": "2024-01-01T10:00:00",
        }
    ]
    snapshot = parse_server_snapshot(rows)
    assert snapshot is not None
    assert snapshot.processing_count == 3
    assert snapshot.queued_count == 5
    assert snapshot.total_mem_used_gb == 120.5
    assert snapshot.free_mem_gb == 32.0
    assert snapshot.timestamp == datetime(2024, 1, 1, 10, 0, 0)


def test_parse_analysis_statuses():
    rows = [
        {
            "build_number": "101",
            "analysis_name": "analysis-a",
            "used_memory_gb": "12.5",
            "duration_hours": "1.25",
        },
        {
            "build_number": "102",
            "analysis_name": "analysis-b",
            "queued_timestamp": "2024-01-01T09:30:00",
        },
    ]
    processing = parse_analysis_statuses(rows[:1], status="PROCESSING")
    queued = parse_analysis_statuses(rows[1:], status="QUEUED")
    assert processing[0].build_number == "101"
    assert processing[0].analysis_name == "analysis-a"
    assert processing[0].used_memory_gb == 12.5
    assert processing[0].duration_hours == 1.25
    assert queued[0].build_number == "102"
    assert queued[0].analysis_name == "analysis-b"
    assert queued[0].queued_timestamp == datetime(2024, 1, 1, 9, 30, 0)
