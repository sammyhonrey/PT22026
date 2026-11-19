
# Hybrid retrieval using FAISS semantic search and keyword matching.
# Multi-answer table questions retrieve all rows from the requested section.

import re
import faiss


class Retriever:

    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.chunks = []

        self.stop_words = {
            "what", "is", "are", "the", "a", "an",
            "of", "for", "in", "on", "to", "and",
            "does", "do", "how", "which", "who"
        }

    def _normalize_words(self, text):
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())

        normalized = []

        for word in words:
            if word in self.stop_words:
                continue

            if word.endswith("ies") and len(word) > 4:
                word = word[:-3] + "y"
            elif word.endswith("s") and len(word) > 3:
                word = word[:-1]

            normalized.append(word)

        return normalized

    def _keyword_score(self, question, chunk_text):
        query_words = set(self._normalize_words(question))
        chunk_words = set(self._normalize_words(chunk_text))

        if not query_words or not chunk_words:
            return 0.0

        matched_words = query_words.intersection(chunk_words)

        return len(matched_words) / len(query_words)

    def _content_intent_bonus(self, question, chunk_text):
        """Reward chunks containing evidence relevant to material questions."""

        q = question.lower()

        material_question = bool(
            re.search(
                r"\b(made of|built from|constructed from|composed of|consists? of|material)\b",
                q
            )
        )

        if not material_question:
            return 0.0

        text = chunk_text.lower()

        material_terms = (
            "iron", "steel", "aluminium", "aluminum", "copper",
            "bronze", "wood", "timber", "concrete", "stone",
            "brick", "glass", "plastic", "carbon fiber",
            "fiberglass", "fibreglass"
        )

        has_material = any(
            re.search(r"\b" + re.escape(term) + r"\b", text)
            for term in material_terms
        )

        return 0.25 if has_material else 0.0

    def _get_table_section(self, question):
        q = question.lower()

        multi_patterns = [
            r"\bwhat are\b",
            r"\blist\b",
            r"\bcharacteristics\b",
            r"\benumerate\b",
            r"\bfeatures\b",
        ]

        is_multi = any(
            re.search(pattern, q)
            for pattern in multi_patterns
        )

        if not is_multi:
            return None

        if "solar pv" in q or "pv" in q:
            return "solar pv characteristic"

        if "wind turbine" in q or "wind" in q:
            return "wind turbine characteristic"

        return None

    def build(self, chunks):
        if not chunks:
            raise ValueError("There are no chunks to index.")

        texts = [chunk["text"] for chunk in chunks]

        vectors = self.embedder.encode(texts)

        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        self.chunks = chunks

    def search(self, question, top_k=3):

        if self.index is None:
            raise RuntimeError(
                "Build the index with a document before searching."
            )

        if not question or not question.strip():
            raise ValueError("The question is empty.")

        if not self.chunks:
            return []

        top_k = max(1, min(top_k, len(self.chunks)))

        is_location_question = bool(
            re.search(r"\b(where|located|location)\b", question.lower())
        )

        if is_location_question:
            search_question = (
                question + " city country geographical position"
            )
            query_vectors = self.embedder.encode(
                [question, search_question]
            )
            scores, positions = self.index.search(
                query_vectors, len(self.chunks)
            )

            original_scores = {
                int(pos): float(score)
                for score, pos in zip(scores[0], positions[0])
                if pos >= 0
            }
            expanded_scores = {
                int(pos): float(score)
                for score, pos in zip(scores[1], positions[1])
                if pos >= 0
            }
        else:
            query_vector = self.embedder.encode([question])
            scores, positions = self.index.search(
                query_vector, len(self.chunks)
            )
            original_scores = {
                int(pos): float(score)
                for score, pos in zip(scores[0], positions[0])
                if pos >= 0
            }
            expanded_scores = {}

        results = []

        for position, original_chunk in enumerate(self.chunks):

            chunk = dict(original_chunk)

            semantic_score = original_scores.get(position, 0.0)

            if is_location_question:
                expanded_score = expanded_scores.get(position, 0.0)
                semantic_score = (
                    0.80 * semantic_score
                    + 0.20 * expanded_score
                )

            keyword_score = self._keyword_score(
                question,
                chunk["text"]
            )

            table_bonus = (
                0.03
                if chunk.get("type") == "table"
                else 0.0
            )

            hybrid_score = (
                0.55 * semantic_score
                + 0.45 * keyword_score
                + table_bonus
            )

            hybrid_score += self._content_intent_bonus(
                question,
                chunk["text"]
            )

            if is_location_question:
                text_lower = chunk["text"].lower()

                location_cue = bool(
                    re.search(
                        r"\b(located|situated|stands|lies)\s+(in|on|at)\b",
                        text_lower
                    )
                    or re.search(
                        r"\b(in|on|at)\s+(?:the\s+)?[A-Z][A-Za-zÀ-ÿ-]+",
                        chunk["text"]
                    )
                )

                if location_cue:
                    hybrid_score += 0.12

                reference_signals = sum([
                    bool(re.search(
                        r"\b(retrieved|accessed|doi|isbn)\b",
                        text_lower
                    )),
                    bool(re.search(
                        r"^\s*(references|bibliography|footnotes)\s*$",
                        text_lower,
                        re.MULTILINE
                    )),
                ])

                if reference_signals >= 1:
                    hybrid_score -= 0.20

            chunk["semantic_score"] = semantic_score
            chunk["keyword_score"] = keyword_score
            chunk["score"] = hybrid_score
            chunk["_position"] = int(position)

            results.append(chunk)

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        # Special handling for Solar PV and Wind Turbine
        # multi-answer table questions.
        required_section = self._get_table_section(question)

        if required_section:

            table_rows = [
                item
                for item in results
                if (
                    item.get("type") == "table"
                    and required_section in item["text"].lower()
                    and "|" in item["text"]
                )
            ]

            if table_rows:

                # Restore the original document order.
                table_rows.sort(
                    key=lambda item: item["_position"]
                )

                for item in table_rows:
                    item.pop("_position", None)

                return table_rows

        # Normal questions continue to use Top-K retrieval.
        results = results[:top_k]

        for item in results:
            item.pop("_position", None)

        return results
