/**
 * Pronunciation Dictionary page: list, search, add, edit, delete rules.
 */

import { ApiError } from "../api.js";
import {
  createPronunciation,
  deletePronunciation,
  listPronunciation,
  updatePronunciation,
} from "../advanced-api.js";
import { validatePronunciationDraft } from "../advanced-state.js";

export function createPronunciationPage({ notifications }) {
  const panel = document.createElement("section");
  panel.className = "panel";

  const head = document.createElement("div");
  head.className = "panel-head";
  head.innerHTML =
    '<h2 class="panel-title">Pronunciation rules</h2><span class="panel-hint">applied before synthesis</span>';

  const body = document.createElement("div");
  body.className = "panel-body stack";

  // --- Form -------------------------------------------------------------
  const form = document.createElement("form");
  form.className = "dictionary-form";

  const word = document.createElement("input");
  word.className = "input";
  word.placeholder = "Word or phrase";
  word.maxLength = 80;
  word.setAttribute("aria-label", "Word or phrase");

  const pronunciation = document.createElement("input");
  pronunciation.className = "input";
  pronunciation.placeholder = "Pronunciation, e.g. Ab-dool-rah-man";
  pronunciation.maxLength = 200;
  pronunciation.setAttribute("aria-label", "Pronunciation");

  const description = document.createElement("input");
  description.className = "input";
  description.placeholder = "Description (optional)";
  description.maxLength = 200;
  description.setAttribute("aria-label", "Description");

  const addButton = document.createElement("button");
  addButton.type = "submit";
  addButton.className = "button button-primary";
  addButton.textContent = "Add rule";

  form.append(word, pronunciation, description, addButton);

  // --- Search -----------------------------------------------------------
  const search = document.createElement("input");
  search.type = "search";
  search.className = "input";
  search.placeholder = "Search rules…";
  search.setAttribute("aria-label", "Search rules");

  const list = document.createElement("div");
  list.className = "dictionary-list";

  body.append(form, search, list);
  panel.append(head, body);

  const wrapper = document.createElement("div");
  wrapper.append(panel);

  let entries = [];
  let editingId = null;

  function renderRow(entry) {
    const row = document.createElement("div");
    row.className = "dictionary-row";

    if (editingId === entry.id) {
      const editForm = document.createElement("form");
      editForm.className = "dictionary-form";

      const editWord = document.createElement("input");
      editWord.className = "input";
      editWord.value = entry.word;
      editWord.maxLength = 80;

      const editPronunciation = document.createElement("input");
      editPronunciation.className = "input";
      editPronunciation.value = entry.pronunciation;
      editPronunciation.maxLength = 200;

      const editDescription = document.createElement("input");
      editDescription.className = "input";
      editDescription.value = entry.description || "";
      editDescription.maxLength = 200;

      const save = document.createElement("button");
      save.type = "submit";
      save.className = "button button-primary button-small";
      save.textContent = "Save";

      const cancel = document.createElement("button");
      cancel.type = "button";
      cancel.className = "button button-ghost button-small";
      cancel.textContent = "Cancel";
      cancel.addEventListener("click", () => {
        editingId = null;
        render();
      });

      editForm.append(editWord, editPronunciation, editDescription, save, cancel);

      editForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const validation = validatePronunciationDraft({
          word: editWord.value,
          pronunciation: editPronunciation.value,
        });

        if (!validation.ok) {
          notifications.error(validation.error);
          return;
        }

        try {
          await updatePronunciation(entry.id, {
            word: validation.word,
            pronunciation: validation.pronunciation,
            description: editDescription.value,
          });
          notifications.success("Rule updated.");
          editingId = null;
          await load();
        } catch (error) {
          notifications.error(
            error instanceof ApiError ? error.message : "Update failed.",
          );
        }
      });

      row.append(editForm);
      return row;
    }

    const text = document.createElement("div");
    text.className = "dictionary-text";

    const title = document.createElement("span");
    title.className = "dictionary-word";
    title.textContent = entry.word;

    const arrow = document.createElement("span");
    arrow.className = "dictionary-arrow";
    arrow.textContent = "→";

    const value = document.createElement("span");
    value.className = "dictionary-pronunciation";
    value.textContent = entry.pronunciation;

    text.append(title, arrow, value);

    if (entry.description) {
      const note = document.createElement("span");
      note.className = "dictionary-note";
      note.textContent = entry.description;
      text.append(note);
    }

    const actions = document.createElement("div");
    actions.className = "library-card-actions";

    const edit = document.createElement("button");
    edit.type = "button";
    edit.className = "button button-ghost button-small";
    edit.textContent = "Edit";
    edit.addEventListener("click", () => {
      editingId = entry.id;
      render();
    });

    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "button button-ghost button-small is-danger";
    remove.textContent = "Delete";
    remove.addEventListener("click", async () => {
      if (!window.confirm(`Delete the rule for "${entry.word}"?`)) {
        return;
      }

      try {
        await deletePronunciation(entry.id);
        notifications.success("Rule deleted.");
        await load();
      } catch (error) {
        notifications.error(
          error instanceof ApiError ? error.message : "Delete failed.",
        );
      }
    });

    actions.append(edit, remove);
    row.append(text, actions);

    return row;
  }

  function render() {
    list.replaceChildren();

    if (entries.length === 0) {
      const empty = document.createElement("div");
      empty.className = "output-empty";
      empty.innerHTML =
        '<p>No pronunciation rules yet.</p><p class="meta-key">Add a word and how it should sound</p>';
      list.append(empty);
      return;
    }

    list.append(...entries.map(renderRow));
  }

  async function load() {
    try {
      entries = await listPronunciation(search.value.trim());
      render();
    } catch (error) {
      notifications.error(
        error instanceof ApiError ? error.message : "Could not load rules.",
      );
    }
  }

  form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const validation = validatePronunciationDraft({
      word: word.value,
      pronunciation: pronunciation.value,
    });

    if (!validation.ok) {
      notifications.error(validation.error);
      return;
    }

    try {
      await createPronunciation({
        word: validation.word,
        pronunciation: validation.pronunciation,
        description: description.value,
      });
      notifications.success("Rule added.");
      form.reset();
      await load();
    } catch (error) {
      notifications.error(
        error instanceof ApiError ? error.message : "Could not add the rule.",
      );
    }
  });

  search.addEventListener("input", () => {
    window.clearTimeout(search._timer);
    search._timer = window.setTimeout(load, 300);
  });

  load();

  return { element: wrapper, refresh: load };
}
