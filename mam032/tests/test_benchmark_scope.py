"""Benchmark context must use verified metadata, not guessed sample sizes."""

import hashlib
import json

import pytest

from src.artifacts import ArtifactReleaseError
from src.views import load_test_origins


def test_verified_metadata_supplies_real_sample_count_and_rejects_replacement(tmp_path):
    path = tmp_path / 'training_metadata.json'
    path.write_text(json.dumps({'origins': {'test': 12}}))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert load_test_origins(path, 'release-a', digest) == 12
    path.write_text(json.dumps({'origins': {'test': 195}}))
    with pytest.raises(ArtifactReleaseError):
        load_test_origins(path, 'release-b', digest)


@pytest.mark.parametrize('count', [True, 0, -1, 12.5, '12'])
def test_invalid_sample_count_is_not_rendered_as_benchmark_evidence(tmp_path, count):
    path = tmp_path / 'training_metadata.json'
    path.write_text(json.dumps({'origins': {'test': count}}))
    with pytest.raises(ValueError, match='positive integer'):
        load_test_origins(path, repr(count), None)
