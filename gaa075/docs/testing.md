# Testing and acceptance-criteria traceability

Run the whole suite:

```powershell
.venv\Scripts\python.exe -m pytest tests -q
```

Current state: **102 tests, all passing, ~21 s** on the development machine
(Python 3.12.10). The core tests use small synthetic DataFrames and need no
dataset; the real-data tests skip automatically when `data/dataset.csv` is absent,
so a fresh clone stays green.

## 1. Test files

| File | Tests | Purpose |
|---|---|---|
| `tests/conftest.py` | — | puts the project root on `sys.path` so `import preprocessing` works |
| `tests/test_preprocessing.py` | **20** | cleaning rules, dedup, multi-label genres, scaling, matrix shape, determinism, raw-data immutability, type annotations |
| `tests/test_recommender.py` | **53** | similarity, ranking, duplicate signature (all six required cases), determinism, errors, preferences, explanations, warning |
| `tests/test_evaluation.py` | **15** | pure metrics, structural invariants on real data, preference comparison, guard against invented relevance metrics |
| `tests/test_integration.py` | **14** | preprocessing → recommender end-to-end, error handling, edge cases |

## 2. Acceptance-criteria traceability (task 7.6)

| Acceptance criterion (tasks.md) | Proving test(s) |
|---|---|
| 4.1 cleaning: input → output rows counted, every removal justified | `test_junk_row_is_dropped`, `test_zero_coded_mask_flags_only_the_coded_row`, `test_counts_are_reported_honestly`, `test_zero_coded_track_is_excluded_but_others_remain` |
| 4.2 track identity = `track_id`, dedup keeps first occurrence | `test_duplicate_keeps_first_occurrence`, `test_matrix_rows_align_with_catalogue` |
| 4.3 scaling fitted on the catalogue, verified bounds | `test_scaler_is_fitted_on_the_catalogue`, `test_constant_features_do_not_produce_nan`, `test_every_row_has_unit_norm` |
| 4.4 block sizes and weights documented; 9 + 114 columns | `test_matrix_dimensions`, `test_multi_genre_track_has_two_nonzero_genre_entries`, `test_block_weights_change_the_matrix` |
| 4.5 one call yields matrix + aligned catalogue + identity lookup | `test_reusing_scaler_and_vocabulary_reproduces_the_matrix`, `test_matrix_rows_align_with_catalogue` |
| 4.6 same code path for app and evaluation, no refit drift | `test_pipeline_is_reproducible`, `test_real_dataset_pipeline` |
| 5.1 cosine scores in range | `test_reference_is_never_recommended`, `test_real_data_recommendations` |
| 5.2 ranking and top-k | `test_k_is_respected`, `test_k_larger_than_available_returns_all_candidates`, `test_k_one_returns_the_single_best_candidate` |
| 5.3 reference excluded, duplicates suppressed, fewer-than-k handled | `test_duplicate_of_the_reference_recording_is_not_returned`, `test_identical_recording_is_returned_once`, `test_different_audio_features_keep_both_variants`, `test_different_duration_keeps_both_variants`, `test_different_artists_are_not_suppressed`, `test_cross_id_duplicate_recordings_are_suppressed_end_to_end` |
| 5.4 preferences change ranking; no preferences changes nothing | `test_no_preferences_preserves_ranking_and_scores`, `test_genre_preference_changes_the_ranking`, `test_artist_only_preference_works`, `test_combined_preferences_follow_the_formula`, `test_genre_component_is_the_share_of_requested_genres` |
| 5.5 explanations traceable and deterministic | `test_explanation_mentions_shared_genre`, `test_explanation_mentions_audio_traits`, `test_explanation_mentions_preferred_artist`, `test_explanations_are_deterministic`, `test_every_explanation_is_non_empty` |
| 5.6 caveat exists once and travels with results | `test_warning_constant_exists_and_is_attached`, `test_warning_is_attached_to_empty_results` |
| 5.7 determinism and edge cases | `test_ordering_is_deterministic`, `test_ties_break_on_track_id`, `test_empty_result_frame_when_no_candidates`, `test_unknown_track_raises`, `test_invalid_track_id_raises`, `test_invalid_k_raises` |
| 6.3 structural correctness on real data | `test_evaluate_all_reports_structural_invariants`, `test_real_dataset_recommendations` |
| 6.1/6.5 no invented relevance metrics | `test_no_relevance_metrics_are_defined` (5 parametrised cases) |
| Error handling: missing/malformed data | `test_loader_raises_actionable_error_for_missing_file`, `test_malformed_csv_file_raises_actionable_error`, `test_missing_required_column_raises_value_error`, `test_empty_dataframe_is_rejected` |
| Integration: excluded tracks unreachable, catalogue bounds | `test_zero_coded_track_is_not_recommendable`, `test_junk_row_is_not_recommendable`, `test_all_returned_tracks_come_from_the_catalogue` |
| Graceful degradation with impossible preferences | `test_unmatched_artist_preference_preserves_order` |

Every "Done when" in Phases 4–6 maps to at least one of the tests above.

## 3. Test design notes

* **Synthetic-first.** Cleaning, dedup, scaling, similarity, suppression and
  preference logic are proven on hand-built frames of 3–11 rows. This keeps the
  suite fast and lets each rule be tested in isolation.
* **Deterministic.** No randomness, no seeds needed; ordering ties are broken on
  `track_id` and asserted.
* **Real-data integration is optional.** `@requires_dataset` skips tests when the
  CSV is missing; those tests assert the documented counts (114,000 → 89,583
  tracks, 9 + 114 features, 0 self-matches).
* **Honesty guards.** `test_no_relevance_metrics_are_defined` fails if anyone adds
  `precision`/`recall`/`accuracy`/`ndcg`/`f1_score` to `evaluation.py`, because the
  dataset has no labels to compute them from.
* **Broken-implementation check (7.3).** The suite has demonstrably caught real
  defects during this project, which is the honest version of that requirement:
  (a) `test_reference_is_never_recommended` failed on an unclipped score of
  `1.0000000000000002`, which is why score clipping exists;
  (b) `test_k_is_respected[10]` failed when `k` exceeded the number of distinct
  candidates, exposing an incorrect test assumption;
  (c) a reference reappearing under a different `track_id` was caught by manual
  inspection of sample output, and is now locked down by
  `test_duplicate_of_the_reference_recording_is_not_returned` — reverting that fix
  fails the test.
  A missing self-exclusion would fail `test_reference_is_never_recommended` and the
  integration test `test_all_returned_tracks_come_from_the_catalogue`.

## 4. Not covered (honest gaps)

* **No UI tests yet** — Phase 8 is not implemented at the time of writing.
* **No musical-quality judgement** — no automated test can assert that a
  recommendation is musically sensible; that remains a human review.
* **No relevance metrics** — impossible without labels (see §3).
* **Platform-specific paths** — tests assume the project layout, which is what
  the README documents.
