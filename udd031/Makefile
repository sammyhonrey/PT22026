# Multimodal Mental State Analyser — task runner
# Author: UDDIN MEJBAH (20243120031)
#
# Every target is a thin wrapper around a command documented in README.md.
# `make help` lists them.

PY ?= python
PORT ?= 7860

.PHONY: help setup models samples run test lint figures report docx probe calibrate clean

help:
	@echo "Multimodal Mental State Analyser — available targets"
	@echo ""
	@echo "  make setup      install runtime dependencies"
	@echo "  make models     pre-download and verify Hugging Face weights"
	@echo "  make samples    generate the six synthetic demo clips"
	@echo "  make clips      rebuild the six RAVDESS demo excerpts (held-out speakers)"
	@echo "  make run        serve the interface on port $(PORT)"
	@echo "  make test       run the test suite"
	@echo "  make probe      fit and measure the six-class acoustic head"
	@echo "  make calibrate  measure the stress distribution and rewrite its thresholds"
	@echo "  make figures    render the report figures"
	@echo "  make report     build report/report.pdf (needs playwright)"
	@echo "  make docx       build the Word report and project handbook"
	@echo "  make clean      remove caches and generated artefacts"
	@echo ""
	@echo "  Evaluation (requires labelled RAVDESS audio):"
	@echo "    $(PY) scripts/fetch_ravdess.py --print-links"
	@echo "    $(PY) scripts/evaluate_zeroshot.py && $(PY) scripts/evaluate_fusion.py"
	@echo "    $(PY) scripts/evaluate_fusion_ablation.py"

setup:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -r requirements.txt

models:
	$(PY) scripts/download_models.py

samples:
	$(PY) scripts/prepare_demo_samples.py

clips:
	$(PY) scripts/build_demo_clips.py --force

probe:
	$(PY) scripts/train_acoustic_probe.py

calibrate:
	$(PY) scripts/calibrate_stress.py --write

run:
	$(PY) app.py --port $(PORT)

test:
	$(PY) -m pytest -q

figures:
	$(PY) scripts/make_figures.py --from-results

# Both targets read the HTML sources in report/, which are local working copies and
# not tracked: the PDF and the two DOCX that ship in the repository are the artefacts.
report:
	$(PY) scripts/build_report.py

docx:
	$(PY) scripts/html_to_docx.py report/report.html \
		"report/Multimodal_Mental_State_Analyser_Report_20243120031.docx" \
		--title "Multimodal Mental State Analyser" \
		--subtitle "Emotion and Stress Prediction by Fusing Speech Audio and Transcribed Text using Hugging Face Transformers" \
		--running-head "Multimodal Mental State Analyser · UDDIN MEJBAH · 20243120031"
	$(PY) scripts/html_to_docx.py report/project_guide.html \
		"report/Project_Guide_and_Viva_Handbook_20243120031.docx" \
		--title "Project Guide and Viva Handbook" \
		--subtitle "How the Multimodal Mental State Analyser was built, why each decision was made, and how to defend it" \
		--running-head "Project handbook · UDDIN MEJBAH · 20243120031"

clean:
	rm -rf .pytest_cache results report/figures report/report.pdf
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
