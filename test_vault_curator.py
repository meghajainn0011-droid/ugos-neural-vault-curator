#!/usr/bin/env python3
"""
Unit tests for vault_curator.py
"""
import pytest
from vault_curator import NeuralVaultCurator


def test_indexing_and_search():
    curator = NeuralVaultCurator(n_components=4)
    items = [
        {
            "id": "photo_01",
            "title": "Sunset at Puri Beach",
            "description": "Golden hour sea waves with orange sky and clouds",
            "tags": ["beach", "sunset", "ocean", "vacation"]
        },
        {
            "id": "photo_02",
            "title": "Family Diwali Celebration in Kolkata",
            "description": "Lamps, fireworks, and evening sweets with parents",
            "tags": ["festival", "family", "diwali", "kolkata"]
        },
        {
            "id": "photo_03",
            "title": "Coding Machine Learning Agent",
            "description": "Python neural network training on terminal dashboard",
            "tags": ["tech", "coding", "python", "ai"]
        }
    ]

    res = curator.index_corpus(items)
    assert res["status"] == "indexed"
    assert res["count"] == 3

    # Search query
    beach_search = curator.search("ocean waves sunset", top_k=1)
    assert len(beach_search) == 1
    assert beach_search[0]["item"]["id"] == "photo_01"

    tech_search = curator.search("python machine learning", top_k=1)
    assert len(tech_search) == 1
    assert tech_search[0]["item"]["id"] == "photo_03"


def test_motion_continuity():
    smooth_flows = [0.8, 0.85, 0.9, 0.75, 0.82]
    eval_smooth = NeuralVaultCurator.evaluate_motion_continuity(smooth_flows)
    assert eval_smooth["eligible"] is True
    assert eval_smooth["recommendation"] == "suitable_for_reel"

    jumpy_flows = [0.1, 25.0, 0.2, 30.0]
    eval_jumpy = NeuralVaultCurator.evaluate_motion_continuity(jumpy_flows)
    assert eval_jumpy["eligible"] is False


if __name__ == "__main__":
    test_indexing_and_search()
    test_motion_continuity()
    print("All unit tests passed successfully!")
