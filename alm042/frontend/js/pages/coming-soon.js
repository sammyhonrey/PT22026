/**
 * Placeholder page for navigation entries that are not implemented yet.
 * It deliberately offers no controls: nothing here pretends to work.
 */
export function createComingSoonPage({ title, description }) {
  const wrapper = document.createElement("div");
  wrapper.className = "coming-soon";

  const badge = document.createElement("span");
  badge.className = "badge badge-accent";
  badge.textContent = "Coming soon";

  const heading = document.createElement("h2");
  heading.textContent = title;

  const text = document.createElement("p");
  text.textContent = description;

  const note = document.createElement("p");
  note.className = "meta-key";
  note.textContent = "This page is part of the next development phase.";

  wrapper.append(badge, heading, text, note);

  return { element: wrapper };
}
