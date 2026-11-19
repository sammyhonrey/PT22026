/**
 * Output panel: playback, metadata and WAV download.
 *
 * The panel keeps its DOM nodes and only updates them, so re-generating does
 * not rebuild the audio element (which would break playback state).
 */

function metaItem(key) {
  const item = document.createElement("div");
  item.className = "meta-item";

  const keyEl = document.createElement("span");
  keyEl.className = "meta-key";
  keyEl.textContent = key;

  const valueEl = document.createElement("span");
  valueEl.className = "meta-value";
  valueEl.textContent = "\u2014";

  item.append(keyEl, valueEl);

  return { item, valueEl };
}

export function createAudioPanel() {
  const panel = document.createElement("section");
  panel.className = "panel";

  const head = document.createElement("div");
  head.className = "panel-head";

  const title = document.createElement("h2");
  title.className = "panel-title";
  title.textContent = "Output";

  const hint = document.createElement("span");
  hint.className = "panel-hint";
  hint.textContent = "WAV \u00b7 16-bit PCM";

  head.append(title, hint);

  const body = document.createElement("div");
  body.className = "panel-body";

  const empty = document.createElement("div");
  empty.className = "output-empty";
  empty.innerHTML = `
    <span class="output-empty-icon" aria-hidden="true">\u25b6</span>
    <p>Your generated speech will appear here.</p>
    <p class="meta-key">Enter a sentence and press Generate</p>
  `;

  const output = document.createElement("div");
  output.hidden = true;

  const audio = document.createElement("audio");
  audio.className = "audio-player";
  audio.id = "player";
  audio.controls = true;
  audio.preload = "metadata";

  const grid = document.createElement("div");
  grid.className = "meta-grid";

  const meta = {
    engine: metaItem("Engine"),
    voice: metaItem("Voice"),
    speed: metaItem("Speed"),
    duration: metaItem("Duration"),
    sampleRate: metaItem("Sample rate"),
  };

  grid.append(
    meta.engine.item,
    meta.voice.item,
    meta.speed.item,
    meta.duration.item,
    meta.sampleRate.item,
  );

  const actions = document.createElement("div");
  actions.className = "output-actions";

  const download = document.createElement("a");
  download.className = "button button-ghost";
  download.id = "download";
  download.textContent = "\u2b07 Download WAV";
  download.setAttribute("download", "");

  actions.append(download);
  output.append(audio, grid, actions);
  body.append(empty, output);
  panel.append(head, body);

  function setMeta(entry, value) {
    entry.valueEl.textContent = value || "\u2014";
  }

  return {
    element: panel,

    /**
     * @param {{audio_url: string, filename: string, duration_seconds: number,
     *          sample_rate: number, engine: string}} result
     * @param {{engineLabel: string, voiceLabel: string, speedLabel: string}} labels
     */
    show(result, labels) {
      audio.src = result.audio_url;
      download.href = result.audio_url;
      download.setAttribute("download", result.filename);

      setMeta(meta.engine, labels.engineLabel);
      setMeta(meta.voice, labels.voiceLabel);
      setMeta(meta.speed, labels.speedLabel);
      setMeta(meta.duration, `${result.duration_seconds.toFixed(2)} s`);
      setMeta(meta.sampleRate, `${result.sample_rate.toLocaleString()} Hz`);

      empty.hidden = true;
      output.hidden = false;

      audio.load();
    },

    clear() {
      audio.pause();
      audio.removeAttribute("src");
      download.removeAttribute("href");

      empty.hidden = false;
      output.hidden = true;
    },
  };
}
