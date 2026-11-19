
# Split document pages into overlapping chunks for retrieval.


def chunk_pages(pages, chunk_size=120, overlap=30):
    # Turn (page_number, text) pairs into overlapping chunks.

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size - 1."
        )

    step = chunk_size - overlap
    chunks = []

    for page_number, text in pages:
        words = text.split()

        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]

            if not piece:
                break

            chunks.append({
                "text": " ".join(piece),
                "page": page_number,
            })

            if start + chunk_size >= len(words):
                break

    return chunks


def chunk_layout_items(items, chunk_size=120, overlap=30):
    # Preserve table context and split normal paragraphs.

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be between 0 and chunk_size - 1."
        )

    step = chunk_size - overlap
    chunks = []

    current_page = None
    table_context = ""

    for item in items:
        page_number = item["page"]
        text = item["text"].strip()
        item_type = item.get("type", "paragraph")

        if not text:
            continue

        # Reset table context when the page changes.
        if page_number != current_page:
            current_page = page_number
            table_context = ""

        if item_type == "table":

            # Identify section headings for the following table rows.
            if (
                "characteristic:" in text.lower()
                and "atb moderate case" in text.lower()
            ):
                table_context = text

                chunks.append({
                    "text": text,
                    "page": page_number,
                    "type": "table",
                })

                continue

            # Attach the current table heading to each table row.
            if table_context:
                contextual_text = (
                    f"{table_context} | {text}"
                )
            else:
                contextual_text = text

            chunks.append({
                "text": contextual_text,
                "page": page_number,
                "type": "table",
            })

            continue

        # Split normal paragraphs into overlapping chunks.
        words = text.split()

        for start in range(0, len(words), step):
            piece = words[start:start + chunk_size]

            if not piece:
                break

            chunks.append({
                "text": " ".join(piece),
                "page": page_number,
                "type": "paragraph",
            })

            if start + chunk_size >= len(words):
                break

    return chunks
