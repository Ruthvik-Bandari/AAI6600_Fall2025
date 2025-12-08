import csv
from pathlib import Path

import pytest

from chatbot_pipeline import fast_search_scored_csv


def _write_sample_csv(path: Path):
    fieldnames = [
        'name', 'city', 'state', 'overall_care_needs_score', 'affordability_score'
    ]
    rows = [
        {'name': 'Community Center A', 'city': 'Springfield', 'state': 'IL', 'overall_care_needs_score': '8.5', 'affordability_score': '9.0'},
        {'name': 'Private Clinic B', 'city': 'Springfield', 'state': 'MO', 'overall_care_needs_score': '6.0', 'affordability_score': '4.0'},
        {'name': 'County Health C', 'city': 'Hartford', 'state': 'CT', 'overall_care_needs_score': '7.5', 'affordability_score': '8.0'},
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def test_fast_search_no_filters(tmp_path):
    csv_path = tmp_path / "sample_scored.csv"
    _write_sample_csv(csv_path)

    results = fast_search_scored_csv(str(csv_path), city=None, state=None, top_n=5)
    assert isinstance(results, list)
    assert len(results) == 3
    # First result should be highest overall score (8.5)
    assert results[0]['name'] == 'Community Center A'


def test_fast_search_state_filter(tmp_path):
    csv_path = tmp_path / "sample_scored.csv"
    _write_sample_csv(csv_path)

    results = fast_search_scored_csv(str(csv_path), city=None, state='CT', top_n=5)
    assert isinstance(results, list)
    assert len(results) == 1
    assert results[0]['state'] == 'CT'
    assert results[0]['name'] == 'County Health C'


def test_fast_search_city_filter(tmp_path):
    csv_path = tmp_path / "sample_scored.csv"
    _write_sample_csv(csv_path)

    results = fast_search_scored_csv(str(csv_path), city='springfield', state=None, top_n=5)
    assert isinstance(results, list)
    # two Springfield entries across states
    assert len(results) == 2
    names = {r['name'] for r in results}
    assert 'Community Center A' in names and 'Private Clinic B' in names
