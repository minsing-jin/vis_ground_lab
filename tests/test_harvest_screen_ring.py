"""Tests for data_harvest.recorder.screen_ring monitor selection behavior."""

from __future__ import annotations

from data_harvest.recorder.screen_ring import ScreenRingBuffer


class TestScreenRingMonitorSelection:
    def test_multi_monitor_index_zero_selects_first_physical(self):
        monitors = [
            {"id": "all"},
            {"id": "built_in"},
            {"id": "external"},
        ]

        selected = ScreenRingBuffer._select_monitor(monitors, monitor_index=0)
        assert selected["id"] == "built_in"

    def test_multi_monitor_out_of_range_selects_first_physical(self):
        monitors = [
            {"id": "all"},
            {"id": "built_in"},
            {"id": "external"},
        ]

        selected = ScreenRingBuffer._select_monitor(monitors, monitor_index=99)
        assert selected["id"] == "built_in"

    def test_multi_monitor_valid_index_selects_requested_physical(self):
        monitors = [
            {"id": "all"},
            {"id": "built_in"},
            {"id": "external"},
        ]

        selected = ScreenRingBuffer._select_monitor(monitors, monitor_index=2)
        assert selected["id"] == "external"

    def test_single_monitor_uses_only_entry(self):
        monitors = [{"id": "single"}]

        selected = ScreenRingBuffer._select_monitor(monitors, monitor_index=0)
        assert selected["id"] == "single"
