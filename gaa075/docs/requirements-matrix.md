# Requirements traceability

Assignment requirements (project brief §12) mapped to the artifact that satisfies
each one and the task in [`tasks.md`](../tasks.md) that produces it. This table is
the audit list used in task 13.1: a requirement is only "Done" when the artifact
exists and works.

Legend — **Planned**: not started. **In progress**: partially built. **Done**: exists and verified.

## Implementation requirements

| ID | Requirement | Satisfied by | Task(s) | Status |
|---|---|---|---|---|
| I1 | Python is the primary language | all `.py` modules | 0.3 | Done |
| I2 | Interactive application demonstrating an ML task | `app.py` + `recommender.py` | 8.1–8.8, 5.1–5.7 | Planned |
| I3 | Accepts user input and displays model output | song selector, preference controls, results list | 8.3–8.5 | Planned |
| I4 | Readable, modular, appropriately commented code | one responsibility per module; docstrings pass | 9.6 | In progress |
| I5 | Appropriate error handling | six handled cases in `app.py` | 8.6 | Planned |
| I6 | Necessary dependencies and configuration | `requirements.txt`, `.gitignore` | 0.4, 0.6 | In progress |
| I7 | Algorithm identified, with operation, capabilities, limitations explained | README algorithm section, About page, report §5 | 9.4, 8.8, 11.5 | Planned |

## Required project files

| ID | Requirement | Satisfied by | Task(s) | Status |
|---|---|---|---|---|
| F1 | Complete application source code | `app.py`, `recommender.py`, `preprocessing.py`, `evaluation.py` | Phases 4–8 | In progress |
| F2 | Training / fine-tuning scripts, if applicable | **Not applicable** — content-based method trains no model; the fitted scaler/encoder are built by `preprocessing.py`. Stated in README §method. | 9.4 | Planned |
| F3 | Supporting files and configurations | `.gitignore`, `requirements.txt`, `docs/` | 0.4, 0.6, 9.3 | In progress |
| F4 | `requirements.txt` with dependencies | `requirements.txt` | 0.6 | In progress |
| F5 | `README.md` with installation, setup and usage | `README.md` | 9.1, 9.2 | In progress |
| F6 | Dataset or clear instructions for obtaining it | `docs/dataset.md` + README dataset section | 2.1–2.7, 9.3 | Planned |

## GitHub submission requirements

| ID | Requirement | Satisfied by | Task(s) | Status |
|---|---|---|---|---|
| G1 | Use Git throughout with meaningful commits | one commit per task on branch `gaa075` | 0.5–13.5 | In progress |
| G2 | Appropriate `.gitignore` | `.gitignore` | 0.4 | Done |
| G3 | No large datasets, weights, checkpoints or binaries | ignore rules + history audit | 0.4, 10.3 | In progress |
| G4 | Do not modify or delete other students' files | commits touch only `gaa075/` | 10.2, 10.3 | Planned |
| G5 | Submit through a Pull Request | PR from `gaa075` branch | 10.4 | Planned — **blocked: no push access (Q2)** |
| G6 | Do not modify the `main` branch | work stays on `gaa075` | 0.5, 10.1 | Done |

## Report requirements (PDF, at least 10 pages)

| ID | Requirement | Satisfied by | Task(s) | Status |
|---|---|---|---|---|
| R1 | Title page | report §1 | 11.3 | Planned |
| R2 | Abstract | report §2 | 11.3 | Planned |
| R3 | Introduction | report §3 | 11.4 | Planned |
| R4 | Related work / literature review | report §4 | 11.1, 11.4 | Planned |
| R5 | System design and methodology | report §5 | 11.5 | Planned |
| R6 | Implementation | report §6 | 11.6 | Planned |
| R7 | Results and discussion | report §7 | 11.7 | Planned |
| R8 | Conclusion and future work | report §8 | 11.8 | Planned |
| R9 | References | report §9 | 11.1, 11.9 | Planned |
| R10 | At least 10 pages, actual results only | exported PDF | 11.10, 11.11 | Planned |

## Demonstration requirements (3–5 minutes)

| ID | Requirement | Satisfied by | Task(s) | Status |
|---|---|---|---|---|
| D1 | Introduce the problem | demo script §1 | 12.1 | Planned |
| D2 | Explain the selected algorithm | demo script §2 | 12.1, 12.4 | Planned |
| D3 | Demonstrate the working application | live run | 12.2, 12.3 | Planned |
| D4 | Show input and corresponding recommendations | live selection + results | 12.3 | Planned |
| D5 | Explain implementation and results | demo script §4 | 12.1 | Planned |
| D6 | Able to answer technical questions | Q&A preparation | 12.4 | Planned |

## Original-work requirements

| ID | Requirement | Satisfied by | Task(s) | Status |
|---|---|---|---|---|
| O1 | Original individual work | all code written in this repo; no copied code | 10.3, 11.1 | In progress |
| O2 | Acknowledgement of external code, datasets, models, resources | acknowledgements section + report references | 9.7, 11.1 | Planned |

## Mark distribution → where the effort goes

Brief §12 awards marks in five groups. This table ties each group to the tasks that
actually earn it, so effort is not spread evenly by accident.

| Marks | What is graded | Plan tasks that earn it | Evidence the grader sees |
|---|---|---|---|
| **30** | Application functionality and technical explanation | 4.1–4.8 (features), 5.1–5.7 (algorithm), 8.1–8.8 (UI), 8.6 (errors) | a working app that recommends from real features, plus the About page explaining the algorithm |
| **20** | Code quality and GitHub repository | 4.6 (single code path), 7.1–7.6 (tests), 9.6 (docstrings), 10.3 (history), 10.5 (self-review) | modular readable modules, a real test suite, one commit per task, clean PR diff |
| **30** | Report content, structure, clarity, references | 11.1–11.11 | a ≥10-page PDF that describes this implementation and these numbers, with real citations |
| **10** | Knowledge, depth of understanding, creativity | 6.1 (honest evaluation design), 6.7 (limitations), 12.4 (Q&A) | justified design choices and an honest discussion of limitations instead of invented metrics |
| **10** | Demonstration | 12.1–12.5 | a 3–5 minute rehearsed run that shows input → recommendations and survives a failure |

**Implication for ordering:** the 30-mark functionality block and the 20-mark code/repo
block worth 50 marks together depend entirely on Phases 4–8 being genuinely finished,
while the report (30 marks) can only be written *after* those results exist. The plan
therefore front-loads implementation and evidence, and never trades them for
documentation volume.

