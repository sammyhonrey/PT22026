"""Phase 12.1 identity tests: corrected Yemeni mapping, preservation, regressions."""

import json

import pandas as pd
import pytest

import app
import arabic_catalogue as ac
import previews

WRONG_SEEDS = {"Fouad Al-Kibsi", "Ali bin Ali Al-Ansi", "Faisal Al-Radhi"}
WRONG_MBIDS = {
    "b74db3e5-dcb8-4832-8e51-aa6c36502bd2",   # Fouad Abdulwahed
    "05d55604-1d31-4482-abc6-bbca6301f2d0",   # Ali bin Mohammed
    "94f5b2bb-a50d-43f1-bdf6-8db205550812",   # Faisal Alawi
}
VERIFIED = {
    "Abu Bakr Salem": ("b0b889b4-e694-41ec-9092-e510edd51bc9", "ابو بكر سالم بلفقيه"),
    "Mohammed Morshed Naji": ("a4bfc9b8-087b-4756-9c25-2cbc1d0cfc5a", "محمد مرشد ناجي"),
    "Mohammed Al-Harithi": ("4bd7a78f-b2de-4ba8-ae73-d532f852fc7e", "Muhammad Hamud Al‐Harithi"),
}

requires_corrected = pytest.mark.skipif(
    not ac.CORRECTED_YEMENI_PATH.is_file(), reason="corrected Yemeni catalogue not built"
)


@pytest.fixture(scope="module")
def corrected() -> pd.DataFrame:
    return pd.read_csv(ac.CORRECTED_YEMENI_PATH, encoding="utf-8")


@pytest.fixture(scope="module")
def corrected_prov() -> dict:
    return json.loads(ac.CORRECTED_YEMENI_PROVENANCE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# Identity verification
# --------------------------------------------------------------------------- #


@requires_corrected
def test_each_verified_seed_has_the_expected_mbid_and_name(corrected_prov):
    verified = {entry["label"]: entry for entry in corrected_prov["verified_seeds"]}
    assert set(verified) == set(VERIFIED)
    for label, (mbid, name) in VERIFIED.items():
        assert verified[label]["mbid"] == mbid
        assert verified[label]["mb_name"] == name
        assert verified[label]["area"] == "Yemen"


@requires_corrected
def test_the_three_wrong_identities_are_recorded_as_unresolved(corrected_prov):
    unresolved = {entry["label"] for entry in corrected_prov["unresolved_seeds"]}
    assert WRONG_SEEDS <= unresolved
    for entry in corrected_prov["unresolved_seeds"]:
        assert entry["reason"]


@requires_corrected
def test_wrong_mbids_and_seed_labels_cannot_return(corrected):
    assert not (set(corrected["mb_artist_id"]) & WRONG_MBIDS)
    assert not (set(corrected["artist_latin"]) & WRONG_SEEDS)


@requires_corrected
def test_similar_names_are_not_merged(corrected):
    """Ali bin Mohammed, Fouad Abdulwahed and Faisal Alawi must not appear."""
    forbidden_names = {"علي بن محمد", "فؤاد عبدالواحد", "فيصل علوي"}
    artists = set(corrected["artist"].astype(str))
    assert not (artists & forbidden_names)


@requires_corrected
def test_every_row_carries_identity_evidence(corrected):
    assert corrected["identity_verification"].str.contains("MBID").all()
    assert corrected["mb_artist_id"].isin([v[0] for v in VERIFIED.values()]).all()


# --------------------------------------------------------------------------- #
# Multi-artist credits and duplicates
# --------------------------------------------------------------------------- #


@requires_corrected
def test_multi_artist_credits_are_separated(corrected):
    for credit in corrected["artist"].astype(str):
        if ";" in credit:
            parts = [part.strip() for part in credit.split(";")]
            assert all(parts), f"empty credit segment in {credit!r}"
            assert len(parts) >= 2
    # the Phase 12 glued credits must not exist in the corrected file
    glued = ["ابو بكر سالم بلفقيهحسين الجسمي", "فؤاد عبدالواحدابو بكر سالم بلفقيه"]
    for bad in glued:
        assert bad not in set(corrected["artist"].astype(str))


@requires_corrected
def test_corrected_catalogue_has_no_duplicate_normalized_tracks(corrected):
    keys = list(
        zip(
            corrected["title"].map(ac.normalize_text),
            corrected["artist"].map(ac.normalize_text),
        )
    )
    assert len(keys) == len(set(keys))


# --------------------------------------------------------------------------- #
# Preservation of the Phase 12 originals
# --------------------------------------------------------------------------- #


def test_original_yemeni_files_are_still_present_and_unchanged():
    original = ac.DATA_DIR / "arabic_catalogue_yemeni.csv"
    provenance = ac.DATA_DIR / "arabic_catalogue_yemeni_provenance.json"
    assert original.is_file() and provenance.is_file()

    frame = pd.read_csv(original, encoding="utf-8")
    # the Phase 12 file still contains its (now superseded) rows
    assert len(frame) == 286
    assert WRONG_SEEDS & set(frame["artist_latin"])

    payload = json.loads(provenance.read_text(encoding="utf-8"))
    assert len(payload["artist_details"]) == 9      # unchanged Phase 12 provenance


# --------------------------------------------------------------------------- #
# Application now uses the corrected data
# --------------------------------------------------------------------------- #


@requires_corrected
def test_default_load_prefers_the_corrected_yemeni_file():
    merged = ac.load_arabic_catalogue().tracks
    yemen = merged[merged["region"] == "Yemen"]

    assert not (set(yemen["mb_artist_id"]) & WRONG_MBIDS)
    assert not (set(yemen["artist_latin"]) & WRONG_SEEDS)
    # 15 legacy base rows + corrected rows, no Phase 12 rows
    assert len(yemen) > 15
    assert set(yemen["genre"]) >= {"Yemeni", "Yemeni traditional", "Yemeni Sanaani"}


def test_explicit_path_load_is_unchanged_and_bypasses_correction():
    base = ac.load_arabic_catalogue(ac.DEFAULT_CATALOGUE_PATH).tracks
    assert (base["region"] == "Yemen").sum() == 15


def test_international_preview_path_is_still_resolve_many(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "resolve_many", lambda items: calls.append(list(items)) or {})
    app.cached_previews.clear()
    assert app.cached_previews((("id", "Unholy", "Sam Smith"),)) == {}
    assert calls == [[("id", "Unholy", "Sam Smith")]]


def test_previews_module_is_not_changed_in_this_phase():
    """Phase 12.1 explicitly does not touch the preview system."""
    import hashlib

    digest = hashlib.sha256(open(previews.__file__, "rb").read()).hexdigest()
    assert digest == "169e9c04c684707e4705f5c171635b5bc36da07f5e5aac7aef2c4b013fa4d251"
