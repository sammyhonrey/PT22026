# VoiceForge — 3–5 minute demonstration plan

Goal: show a complete, working product without depending on a download or
an unreliable network. Total time: about 4 minutes.

## Before the session (5 minutes earlier)

1. Start the application: `python app.py` (from the project folder, venv
   active). Wait for `VoiceForge is running at http://127.0.0.1:7860`.
2. Open the page once so the Kokoro model is loaded into memory (the first
   request is the slowest).
3. Keep two backup WAV files ready in `outputs/demo/` in case audio cannot
   be played live; say clearly that they were generated beforehand.
4. Have a small PDF or DOCX ready (e.g. `demo.docx` with 3 short
   paragraphs).
5. Have one pronunciation rule and one preset already possible to create
   in under 20 seconds.

## Script

### 0:00 — Introduction (20 s)

> "VoiceForge is a local text-to-speech application. It runs entirely on
> this machine, uses the pretrained Kokoro-82M model as its main engine and
> an experimental Tacotron-lite model I trained myself. Everything you see
> is served by a small Python standard-library server with a vanilla
> JavaScript frontend."

Show the six navigation items.

### 0:20 — Speech Studio (60 s)

1. Speech Studio → type: *"Hello, welcome to VoiceForge."*
2. Point out: engine selector (Kokoro / custom), voice selector, speed slider.
3. Click **Generate speech**. Audio appears with metadata (engine, voice,
   speed, duration, sample rate).
4. Move the speed slider to **1.5×**, generate again, mention that Kokoro
   applies speed per request without reloading the model.
5. Press **Enter** in the text box to show the keyboard shortcut; then
   **Shift+Enter** to show a newline. Mention duplicate-submit protection.
6. Download the WAV and click **MP3** (or the library MP3 link).

### 1:20 — Pronunciation + preset (50 s)

1. Open **Pronunciation** → add `Abdulrahman` → `Ab-dool-rah-man` → Add.
2. Back to Speech Studio, type *"Hello, my name is Abdulrahman."* with the
   pronunciation toggle enabled, generate.
3. Explain honestly: the rule substitutes the respelling before synthesis;
   the model's phoneme string changes, which we verified in tests — it does
   not guarantee perfect pronunciation.
4. Open **Voice Presets** → create `Fast Narration` (Kokoro, a male voice,
   1.25×). Go back to the Studio and apply it from the preset dropdown;
   note that applying does not start generation.

### 2:10 — Document to Speech (60 s)

1. Open **Document to Speech**, choose the prepared PDF/DOCX.
2. Show the extracted text in the editable box, page/character count, and
   mention the scanned-PDF/OCR limitation.
3. Edit one sentence, choose the preset, click **Generate from document**.
4. Show progress (queued → processing → completed) and the result player,
   then mention it is saved in the library automatically.

### 3:10 — Audio Library (40 s)

1. Open **Audio Library**: the recordings from the demo are listed with
   engine, voice, speed, duration and size.
2. Search for "welcome", sort by name, filter by engine — all server-side.
3. Play one recording, download WAV and MP3.
4. Rename one record and delete a test record (confirm dialog).

### 3:50 — Wrap-up (20 s)

> "All of this is tested: 200 Python tests, 100 JavaScript tests and 16
> real-model tests, all passing. Data is stored in SQLite and survives a
> restart. Known limitations are OCR, respelling-based pronunciation and
> the experimental quality of my own model."

## Contingencies

| Problem | Response |
|---|---|
| Audio does not play | Use the pre-generated backup WAV and explain it was generated beforehand. |
| Model slow to load | Show the already-open page; if needed, narrate the architecture while waiting. |
| Upload fails | Explain the 10 MB limit and use the smaller backup document. |
| Browser download blocked | Show the file in the library and open it from `outputs/library/`. |
| Time overrun | Skip the preset demo; keep document → library → MP3. |

## Commands cheat sheet

```powershell
# activate the environment
.\.venv\Scripts\Activate.ps1
# run the application
python app.py
# tests (optional to show)
.\.venv\Scripts\python.exe -m pytest tests -q
node --test frontend/tests
```
