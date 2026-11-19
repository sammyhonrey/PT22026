"""Tests for the Arabic metadata catalogue (arabic_catalogue.py).

These verify the metadata-only contract: Arabic/Latin search, Unicode
normalisation, duplicate handling, validation, and transparent metadata-based
scoring that is never presented as audio similarity.
"""

import pandas as pd
import pytest

import arabic_catalogue as ac

requires_catalogue = pytest.mark.skipif(
    not ac.DEFAULT_CATALOGUE_PATH.is_file(),
    reason="data/arabic_catalogue.csv not built (run scripts/fetch_arabic_metadata.py)",
)

COLUMNS = [
    "track_id",
    "title",
    "artist",
    "artist_latin",
    "album",
    "genre",
    "region",
    "source",
    "source_url",
]


def make_frame(rows):
    return pd.DataFrame(rows, columns=COLUMNS)


@pytest.fixture()
def frame():
    return make_frame(
        [
            ("arb001", "أرجعي يا ألف ليلة", "فيروز", "Fairuz", "Bent El Hares", "Levantine", "Lebanon", "MusicBrainz", "u1"),
            ("arb002", "Aal Zayno", "فيروز", "Fairuz", "Bent El Hares", "Levantine", "Lebanon", "MusicBrainz", "u2"),
            ("arb003", "أما براوة", "فيروز", "Fairuz", "Single", "Levantine", "Lebanon", "MusicBrainz", "u3"),
            ("arb004", "تملي معاك", "عمرو دياب", "Amr Diab", "Tamally Maak", "Arabic pop", "Egypt", "MusicBrainz", "u4"),
            ("arb005", "نور العين", "عمرو دياب", "Amr Diab", "Nour El Ain", "Arabic pop", "Egypt", "MusicBrainz", "u5"),
            ("arb006", "Ya Rayah", "الشاب خالد", "Cheb Khaled", "Ya Rayah", "Rai", "Algeria", "MusicBrainz", "u6"),
            ("arb007", "Aal Zayno", "فيروز", "Fairuz", "Compilation", "Levantine", "Lebanon", "MusicBrainz", "u7"),
        ]
    )


@pytest.fixture()
def catalogue(frame):
    return ac.ArabicCatalogue(frame)


# --------------------------------------------------------------------------- #
# Unicode normalisation
# --------------------------------------------------------------------------- #


def test_alef_variants_fold_together():
    assert ac.normalize_text("أم كلثوم") == ac.normalize_text("ام كلثوم") == ac.normalize_text("آم كلثوم")


def test_diacritics_and_tatweel_are_removed():
    assert ac.normalize_text("مُحَمَّد عَبْدُو") == ac.normalize_text("محمد عبدو")
    assert ac.normalize_text("فـــيروز") == "فيروز"


def test_ta_marbuta_and_alef_maqsura_fold():
    # alef maqsura folds to ya, ta marbuta folds to ha
    assert ac.normalize_text("ليلى") == ac.normalize_text("ليلي") == "ليلي"
    assert ac.normalize_text("أغنية") == ac.normalize_text("اغنيه") == "اغنيه"
    assert ac.normalize_text("ليلى") != ac.normalize_text("ليلا")   # different spelling, kept


def test_latin_case_and_punctuation():
    assert ac.normalize_text("  Amr   Diab! ") == "amr diab"


def test_is_arabic_script():
    assert ac.is_arabic_script("فيروز")
    assert not ac.is_arabic_script("Fairuz")


# --------------------------------------------------------------------------- #
# Loading and validation
# --------------------------------------------------------------------------- #


def test_missing_file_raises(tmp_path):
    with pytest.raises(ac.ArabicCatalogueError) as excinfo:
        ac.load_arabic_catalogue(tmp_path / "nope.csv")
    assert "not found" in str(excinfo.value)


def test_missing_columns_raise():
    with pytest.raises(ac.ArabicCatalogueError):
        ac.ArabicCatalogue(pd.DataFrame([{"title": "x"}]))


def test_duplicate_title_and_artist_are_collapsed(catalogue):
    # arb007 duplicates arb002's title+artist and must be dropped
    assert "arb007" not in set(catalogue.tracks["track_id"])
    assert len(catalogue.tracks) == 6


def test_stats(catalogue):
    stats = catalogue.stats()
    assert stats.tracks == 6
    assert stats.artists == 3
    assert stats.genres == 3
    assert stats.regions == 3
    assert stats.arabic_titles >= 3


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #


def test_search_arabic(catalogue):
    hits = catalogue.search("فيروز")
    assert len(hits) == 3
    assert set(hits["artist"]) == {"فيروز"}


def test_search_english_transliteration(catalogue):
    hits = catalogue.search("Fairuz")
    assert len(hits) == 3


def test_search_partial_word(catalogue):
    assert len(catalogue.search("عمرو")) == 2
    assert len(catalogue.search("diab")) == 2      # latin artist name


def test_search_arabic_without_diacritics_matches(catalogue):
    assert len(catalogue.search("ام كلثوم")) == 0   # not in this fixture
    assert len(catalogue.search("فيروز")) == 3      # plain form matches


def test_search_no_match_is_empty(catalogue):
    assert catalogue.search("zzzz nothing").empty


def test_search_empty_query_returns_rows(catalogue):
    assert len(catalogue.search("")) == 6


# --------------------------------------------------------------------------- #
# Metadata-based recommendation
# --------------------------------------------------------------------------- #


def test_same_artist_scores_full_metadata_match(catalogue):
    results = catalogue.recommend("arb001", limit=5)
    assert not results.empty
    assert set(results["artist"]) == {"فيروز"}
    assert (results["metadata_score"] == 1.0).all()
    assert results["explanation"].str.contains("same artist").all()


def test_same_genre_only_scores_point_three(catalogue):
    # arb006 is the only Rai track; nothing else shares that genre/region/artist
    results = catalogue.recommend("arb006", limit=5)
    assert results.empty

    # a Levantine track from another artist would score 0.3 for genre + 0.1 region
    frame = make_frame(
        [
            ("a", "Song One", "Artist A", "A", "Album", "Levantine", "Lebanon", "MusicBrainz", "u"),
            ("b", "Song Two", "Artist B", "B", "Album", "Levantine", "Lebanon", "MusicBrainz", "u"),
            ("c", "Song Three", "Artist C", "C", "Album", "Rai", "Algeria", "MusicBrainz", "u"),
        ]
    )
    cat = ac.ArabicCatalogue(frame)
    results = cat.recommend("a", limit=5)
    row = results[results["track_id"] == "b"].iloc[0]
    assert row["metadata_score"] == pytest.approx(0.4)           # genre 0.3 + region 0.1
    assert "same genre" in row["explanation"]
    assert "same region" in row["explanation"]
    assert "same artist" not in row["explanation"]


def test_reference_is_never_returned(catalogue):
    results = catalogue.recommend("arb001", limit=10)
    assert "arb001" not in set(results["track_id"])


def test_limit_and_determinism(catalogue):
    first = catalogue.recommend("arb001", limit=2)
    second = catalogue.recommend("arb001", limit=2)
    assert len(first) == 2
    pd.testing.assert_frame_equal(first, second)
    assert list(first["rank"]) == [1, 2]


def test_genre_filter_restricts_candidates(catalogue):
    assert catalogue.recommend("arb001", limit=5, genres=("Rai",)).empty
    assert not catalogue.recommend("arb001", limit=5, genres=("Levantine",)).empty


def test_unknown_track_raises(catalogue):
    with pytest.raises(KeyError):
        catalogue.recommend("missing-id")


def test_results_carry_no_audio_similarity_columns(catalogue):
    results = catalogue.recommend("arb001", limit=3)
    for forbidden in ("score", "numeric_score", "genre_score", "similarity"):
        assert forbidden not in results.columns
    assert "metadata_score" in results.columns


# --------------------------------------------------------------------------- #
# Real catalogue (built from MusicBrainz)
# --------------------------------------------------------------------------- #


@requires_catalogue
def test_real_catalogue_contract():
    cat = ac.load_arabic_catalogue()
    stats = cat.stats()

    assert stats.tracks > 1_000
    assert stats.artists > 20
    assert stats.regions >= 5
    assert stats.arabic_titles > 0
    assert stats.arabic_artists > 0

    hits = cat.search("فيروز")
    assert not hits.empty

    reference = hits.iloc[0]
    recs = cat.recommend(str(reference["track_id"]), limit=5)
    assert not recs.empty
    assert recs["metadata_score"].between(0, 1).all()
    assert recs["track_id"].nunique() == len(recs)
    assert "score" not in recs.columns


@requires_catalogue
def test_real_catalogue_provenance_is_recorded():
    prov = ac.provenance()
    assert prov.get("source") == "MusicBrainz"
    assert "CC0" in prov.get("licence", "")
    assert prov.get("tracks", 0) > 0
