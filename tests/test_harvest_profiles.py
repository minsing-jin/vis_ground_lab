"""Tests for data_harvest profiles module."""

from __future__ import annotations

from data_harvest.profiles.registry import discover_profiles, list_profiles, get_profile


class TestProfiles:
    def test_discover_and_list(self):
        discover_profiles()
        names = list_profiles()
        assert "civ6" in names

    def test_get_civ6(self):
        discover_profiles()
        p = get_profile("civ6")
        assert p.display_name == "Civilization VI"
        assert "btn_end_turn" in p.semantic_dict
        assert "main_map" in p.screen_types
        assert "minimap" in p.roi_hints

    def test_unknown_profile_raises(self):
        import pytest

        with pytest.raises(KeyError):
            get_profile("nonexistent_game_xyz")
