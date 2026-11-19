"""
feedback.py - Turns the AI's correction into something a learner can learn from.

The AI model (see grammar_model.py) only gives us a corrected sentence.
This file compares the original and corrected sentences word by word and:
  1. finds every change (an "edit"),
  2. guesses what kind of mistake it was (verb tense, article, spelling, ...),
  3. attaches a short grammar lesson,
  4. draws the changes in colour (red = removed, green = added).

This part uses simple rules, NOT machine learning, so it is fast and easy to
understand. Because they are rules, the mistake labels are a best guess - this
is discussed as a limitation in the README and report.
"""

import difflib
import html
import re
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# Small word lists used by the rules
# ---------------------------------------------------------------------------
ARTICLES = {"a", "an", "the"}

PREPOSITIONS = {
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about", "into",
    "onto", "over", "under", "between", "among", "through", "during", "since",
    "until", "after", "before", "around", "across", "toward", "towards", "upon",
}

PRONOUNS = {
    "i", "me", "my", "mine", "you", "your", "yours", "he", "him", "his", "she",
    "her", "hers", "it", "its", "we", "us", "our", "ours", "they", "them",
    "their", "theirs", "this", "that", "these", "those", "who", "whom", "which",
}

SUBJECT_PRONOUNS = {"i", "you", "he", "she", "it", "we", "they", "there", "this", "that"}

# Pairs that are about subject-verb agreement (he IS / they ARE).
AGREEMENT_PAIRS = {
    frozenset(p) for p in [("is", "are"), ("was", "were"), ("has", "have"),
                           ("does", "do"), ("am", "is"), ("am", "are"),
                           ("doesn't", "don't"), ("isn't", "aren't"),
                           ("wasn't", "weren't"), ("hasn't", "haven't")]
}

# Present <-> past forms of "be" and common helpers (is -> was = tense change).
TENSE_PAIRS = {
    frozenset(p) for p in [("is", "was"), ("are", "were"), ("am", "was"),
                           ("is", "were"), ("are", "was"), ("has", "had"),
                           ("have", "had"), ("do", "did"), ("does", "did"),
                           ("will", "would"), ("can", "could"), ("will", "did"),
                           ("shall", "should"), ("may", "might")]
}

# Common irregular verbs: base, past, past participle.
IRREGULAR_VERBS = [
    ("be", "was", "been"), ("become", "became", "become"), ("begin", "began", "begun"),
    ("break", "broke", "broken"), ("bring", "brought", "brought"), ("build", "built", "built"),
    ("buy", "bought", "bought"), ("catch", "caught", "caught"), ("choose", "chose", "chosen"),
    ("come", "came", "come"), ("do", "did", "done"), ("drink", "drank", "drunk"),
    ("drive", "drove", "driven"), ("eat", "ate", "eaten"), ("fall", "fell", "fallen"),
    ("feel", "felt", "felt"), ("find", "found", "found"), ("fly", "flew", "flown"),
    ("forget", "forgot", "forgotten"), ("get", "got", "gotten"), ("give", "gave", "given"),
    ("go", "went", "gone"), ("grow", "grew", "grown"), ("have", "had", "had"),
    ("hear", "heard", "heard"), ("keep", "kept", "kept"), ("know", "knew", "known"),
    ("leave", "left", "left"), ("lose", "lost", "lost"), ("make", "made", "made"),
    ("meet", "met", "met"), ("pay", "paid", "paid"), ("read", "read", "read"),
    ("run", "ran", "run"), ("say", "said", "said"), ("see", "saw", "seen"),
    ("sell", "sold", "sold"), ("send", "sent", "sent"), ("sing", "sang", "sung"),
    ("sit", "sat", "sat"), ("sleep", "slept", "slept"), ("speak", "spoke", "spoken"),
    ("spend", "spent", "spent"), ("stand", "stood", "stood"), ("swim", "swam", "swum"),
    ("take", "took", "taken"), ("teach", "taught", "taught"), ("tell", "told", "told"),
    ("think", "thought", "thought"), ("understand", "understood", "understood"),
    ("wake", "woke", "woken"), ("wear", "wore", "worn"), ("win", "won", "won"),
    ("write", "wrote", "written"),
]
VERB_FORMS = {}  # word -> set of all forms of the same verb
for forms in IRREGULAR_VERBS:
    all_forms = set(forms) | {forms[0] + "s", forms[0] + "es", forms[0] + "ing"}
    for form in forms:
        VERB_FORMS.setdefault(form, set()).update(all_forms)

# ---------------------------------------------------------------------------
# Mini grammar lessons, one per mistake category
# ---------------------------------------------------------------------------
LESSONS = {
    "Verb tense": (
        "Match the verb to WHEN the action happens. Finished actions use the past "
        "tense: 'Yesterday I <b>went</b> home', not 'Yesterday I go home'."
    ),
    "Subject-verb agreement": (
        "The verb must agree with its subject. Singular subjects (he, she, it, one "
        "person) take 'is / has / goes'; plural subjects take 'are / have / go'."
    ),
    "Articles (a / an / the)": (
        "Use 'a/an' for one non-specific thing ('a book'), 'the' for a specific or "
        "already-mentioned thing ('the book I bought'). Use 'an' before a vowel sound."
    ),
    "Singular / plural": (
        "Countable nouns need -s when there is more than one: 'two <b>books</b>', "
        "'many <b>students</b>'."
    ),
    "Prepositions": (
        "Prepositions often follow fixed patterns: 'interested <b>in</b>', 'arrive "
        "<b>at</b> school', 'on Monday', 'in May'. These are best learned in phrases."
    ),
    "Spelling": "This word was misspelled. Check the spelling carefully or read it aloud.",
    "Capitalization": (
        "Start every sentence with a capital letter. Names, countries, languages, days "
        "and 'I' are always capitalized."
    ),
    "Punctuation": (
        "Punctuation shows where ideas start and stop. End sentences with . ? or ! "
        "and use commas to separate parts of a sentence."
    ),
    "Pronouns": (
        "Pronouns must match the noun they replace (he/she/it/they) and their role "
        "in the sentence ('<b>I</b> saw him' vs 'he saw <b>me</b>')."
    ),
    "Word order": (
        "English usually follows Subject - Verb - Object order: '<b>I really like</b> "
        "music', not 'I like really music'."
    ),
    "Missing word": "A necessary word was missing, so it was added to complete the sentence.",
    "Extra word": "This word was not needed, so it was removed to make the sentence correct and clear.",
    "Word form": (
        "Use the right form of the word. After verbs like 'enjoy' and 'like' we often use "
        "-ing ('I like <b>playing</b>'). '-ed' adjectives describe how you feel ('I am "
        "<b>interested</b>'); '-ing' adjectives describe the thing ('the film is <b>interesting</b>')."
    ),
    "Commonly confused words": (
        "These words sound alike but mean different things: their (belongs to them) / there "
        "(a place) / they're (they are); to / too / two; your / you're; its / it's; then / than."
    ),
    "Word choice": (
        "A different word fits better here, either for grammar or for natural-sounding English."
    ),
}

# Groups of words that learners often mix up because they sound similar.
CONFUSED_GROUPS = [
    {"their", "there", "they're"}, {"to", "too", "two"}, {"your", "you're"},
    {"its", "it's"}, {"then", "than"}, {"affect", "effect"}, {"lose", "loose"},
    {"accept", "except"}, {"whose", "who's"}, {"were", "where", "we're"},
    {"advice", "advise"}, {"quiet", "quite"},
]

CATEGORY_COLORS = {  # used for the little label tags in the app
    "Verb tense": "#7c3aed", "Subject-verb agreement": "#9333ea",
    "Articles (a / an / the)": "#2563eb", "Singular / plural": "#0891b2",
    "Prepositions": "#0d9488", "Spelling": "#dc2626", "Capitalization": "#ea580c",
    "Punctuation": "#64748b", "Pronouns": "#db2777", "Word order": "#ca8a04",
    "Word form": "#4f46e5", "Commonly confused words": "#be185d",
    "Missing word": "#16a34a", "Extra word": "#b91c1c", "Word choice": "#475569",
}


@dataclass
class Edit:
    """One change between the original and corrected text."""
    removed: list      # words taken out (may be empty)
    added: list        # words put in (may be empty)
    category: str
    lesson: str

    def describe(self) -> str:
        before, after = " ".join(self.removed), " ".join(self.added)
        if self.category == "Word order":
            return f'moved "{before}" to a new position'
        if before and after:
            return f'"{before}" → "{after}"'
        if after:
            return f'added "{after}"'
        return f'removed "{before}"'


# ---------------------------------------------------------------------------
# Step 1: split text into tokens (words and punctuation marks)
# ---------------------------------------------------------------------------
TOKEN_PATTERN = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]")


def tokenize(text: str) -> list:
    """'I goed home.' -> ['I', 'goed', 'home', '.']"""
    return TOKEN_PATTERN.findall(text or "")


def is_punct(token: str) -> bool:
    return not any(ch.isalnum() for ch in token)


def join_tokens(tokens: list) -> str:
    """Join tokens back into readable text (no space before . , ! ? etc.)."""
    text = ""
    for tok in tokens:
        if text and not (is_punct(tok) and tok not in "([{\"'") and not text.endswith(("(", "[", "{")):
            text += " "
        text += tok
    return text


# ---------------------------------------------------------------------------
# Step 2: decide what kind of mistake an edit is
# ---------------------------------------------------------------------------
def _similar(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _is_regular_tense_change(a: str, b: str) -> bool:
    """walk -> walked, study -> studied, stop -> stopped (and the reverse)."""
    for x, y in ((a, b), (b, a)):
        if y.endswith("ed") and (y[:-2] == x or y[:-1] == x or y[:-3] + "y" == x
                                 or (len(y) > 4 and y[:-3] == x and y[-3] == y[-4])):
            return True
    return False


def _is_plural_change(a: str, b: str) -> bool:
    """book -> books, box -> boxes, city -> cities (and the reverse)."""
    for x, y in ((a, b), (b, a)):
        if y in (x + "s", x + "es") or (x.endswith("y") and y == x[:-1] + "ies"):
            return True
    return False


def _strip_apostrophes(word: str) -> str:
    return word.replace("'", "").replace("’", "")


def _is_word_form_change(a: str, b: str) -> bool:
    """play -> playing, interesting -> interested, quick -> quickly (same root, new ending)."""
    endings = ("ing", "ed", "d", "ly", "er", "est", "tion", "ment", "ness", "ful")
    def root(w):
        for end in endings:
            if w.endswith(end) and len(w) - len(end) >= 3:
                return w[: -len(end)].rstrip("e")
        return w.rstrip("e")
    ra, rb = root(a), root(b)
    return a != b and (ra == rb or ra.startswith(rb) or rb.startswith(ra)) and min(len(ra), len(rb)) >= 3


def categorize(removed: list, added: list, previous_word: str = "") -> str:
    """Return the mistake category for one edit using simple grammar rules."""
    old = [w.lower() for w in removed]
    new = [w.lower() for w in added]
    words = old + new

    # Only punctuation changed
    if words and all(is_punct(w) for w in words):
        return "Punctuation"

    # Same words, different capital letters (i -> I, monday -> Monday)
    if removed and len(removed) == len(added) and old == new:
        return "Capitalization"

    # Same words in a different order
    if len(old) > 1 and sorted(old) == sorted(new):
        return "Word order"

    # Articles added, removed or swapped (a / an / the)
    if words and all(w in ARTICLES for w in words):
        return "Articles (a / an / the)"

    # Single word replaced by a single word
    if len(old) == 1 and len(new) == 1:
        a, b = old[0], new[0]
        if _strip_apostrophes(a) == _strip_apostrophes(b):
            return "Punctuation"                      # dont -> don't, its -> it's
        if any(a in group and b in group for group in CONFUSED_GROUPS):
            return "Commonly confused words"
        na, nb = _strip_apostrophes(a), _strip_apostrophes(b)
        if frozenset((na, nb)) in {frozenset(map(_strip_apostrophes, p)) for p in AGREEMENT_PAIRS}:
            return "Subject-verb agreement"           # dont -> doesn't
        if frozenset((a, b)) in AGREEMENT_PAIRS:
            return "Subject-verb agreement"
        if frozenset((a, b)) in TENSE_PAIRS or _is_regular_tense_change(a, b):
            return "Verb tense"
        for wrong, right in ((a, b), (b, a)):   # buyed -> bought, goed -> went
            if wrong.endswith("ed") and right in VERB_FORMS and \
               (wrong[:-2] in VERB_FORMS[right] or wrong[:-1] in VERB_FORMS[right]):
                return "Verb tense"
        if a in VERB_FORMS and b in VERB_FORMS[a]:
            # go -> goes after "he" is agreement; go -> went is tense
            if previous_word.lower() in SUBJECT_PRONOUNS and _is_plural_change(a, b):
                return "Subject-verb agreement"
            return "Verb tense"
        if _is_plural_change(a, b):
            if previous_word.lower() in SUBJECT_PRONOUNS:
                return "Subject-verb agreement"
            return "Singular / plural"
        if a in PREPOSITIONS and b in PREPOSITIONS:
            return "Prepositions"
        if a in PRONOUNS and b in PRONOUNS:
            return "Pronouns"
        if _is_word_form_change(a, b):
            return "Word form"
        if _similar(a, b) >= 0.75:
            return "Spelling"
        return "Word choice"

    # Prepositions added / removed
    if words and all(w in PREPOSITIONS for w in words):
        return "Prepositions"

    # Words only added or only removed
    if not old:
        return "Missing word"
    if not new:
        return "Extra word"

    # Article swapped together with another word, e.g. "a apple" -> "an apple"
    if any(w in ARTICLES for w in words):
        return "Articles (a / an / the)"
    return "Word choice"


# ---------------------------------------------------------------------------
# Step 3: compare two texts and list every edit
# ---------------------------------------------------------------------------
def find_edits(original: str, corrected: str) -> list:
    """Return a list of Edit objects describing how `original` became `corrected`."""
    src, tgt = tokenize(original), tokenize(corrected)
    # Compare in lower case so that a capitalization fix is still an "edit"
    # but does not confuse the alignment of the other words.
    matcher = difflib.SequenceMatcher(None, [t.lower() for t in src],
                                      [t.lower() for t in tgt], autojunk=False)
    edits = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            # Words match ignoring case - check for capitalization changes
            for s, t in zip(src[i1:i2], tgt[j1:j2]):
                if s != t:
                    edits.append(Edit([s], [t], "Capitalization", LESSONS["Capitalization"]))
            continue
        previous_word = src[i1 - 1] if i1 > 0 else ""
        for removed, added in _split_block(src[i1:i2], tgt[j1:j2]):
            category = categorize(removed, added, previous_word)
            edits.append(Edit(removed, added, category, LESSONS[category]))
            previous_word = removed[-1] if removed else previous_word
    return _merge_moved_words(edits)


def _split_block(removed: list, added: list) -> list:
    """Break one big change into smaller ones so each gets its own lesson.

    "buyed a" -> "bought an"  becomes  "buyed"->"bought"  and  "a"->"an".
    "arts"    -> "art ."      becomes  "arts"->"art"      and  added ".".
    """
    words_old = [t for t in removed if not is_punct(t)]
    words_new = [t for t in added if not is_punct(t)]
    punct_old = [t for t in removed if is_punct(t)]
    punct_new = [t for t in added if is_punct(t)]

    pieces = []
    if (punct_old or punct_new) and (words_old or words_new) and punct_old != punct_new:
        pieces.append((punct_old, punct_new))
        removed, added = words_old, words_new

    if len(removed) == len(added) > 1:
        for r, a in zip(removed, added):
            if r != a:
                pieces.append(([r], [a]))
    elif removed or added:
        pieces.append((removed, added))
    return pieces


def _merge_moved_words(edits: list) -> list:
    """If a word was removed in one place and added in another, call it 'Word order'."""
    result = list(edits)
    for i, first in enumerate(edits):
        for second in edits[i + 1:]:
            if first not in result or second not in result:
                continue
            pair = {first.category, second.category}
            if pair == {"Missing word", "Extra word"} and \
               [w.lower() for w in first.removed + first.added] == \
               [w.lower() for w in second.removed + second.added]:
                moved = " ".join(first.removed + first.added)
                merged = Edit([moved], [moved], "Word order", LESSONS["Word order"])
                result[result.index(first)] = merged
                result.remove(second)
    return result


def correctness_score(original: str, corrected: str) -> int:
    """Percentage of the original words the AI kept unchanged (100 = no mistakes found)."""
    src = tokenize(original)
    if not src:
        return 100
    tgt = tokenize(corrected)
    matcher = difflib.SequenceMatcher(None, src, tgt, autojunk=False)
    kept = sum(block.size for block in matcher.get_matching_blocks())
    return round(100 * kept / max(len(src), len(tgt)))


def similarity(a: str, b: str) -> float:
    """How similar two sentences are, 0.0 to 1.0, ignoring case."""
    ta = [t.lower() for t in tokenize(a)]
    tb = [t.lower() for t in tokenize(b)]
    if not ta and not tb:
        return 1.0
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


# ---------------------------------------------------------------------------
# Step 4: draw the changes as coloured HTML for the app
# ---------------------------------------------------------------------------
def diff_html(original: str, corrected: str) -> str:
    """Show the corrected text with removed words in red and added words in green."""
    src, tgt = tokenize(original), tokenize(corrected)
    matcher = difflib.SequenceMatcher(None, [t.lower() for t in src],
                                      [t.lower() for t in tgt], autojunk=False)
    parts = []  # (first_token, html_piece)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for s, t in zip(src[i1:i2], tgt[j1:j2]):
                if s == t:
                    parts.append((t, html.escape(t)))
                else:
                    parts.append((s, f'<del class="gc-del">{html.escape(s)}</del>'
                                     f'<ins class="gc-ins">{html.escape(t)}</ins>'))
            continue
        if i2 > i1:
            parts.append((src[i1], f'<del class="gc-del">{html.escape(join_tokens(src[i1:i2]))}</del>'))
        if j2 > j1:
            parts.append((tgt[j1], f'<ins class="gc-ins">{html.escape(join_tokens(tgt[j1:j2]))}</ins>'))

    # Join the pieces with spaces, but no space before punctuation like . , ! ?
    out = ""
    for first_token, piece in parts:
        if out and not (is_punct(first_token) and first_token not in "([{\"'"):
            out += " "
        out += piece
    return '<div class="gc-diff">' + out + "</div>"


def lessons_html(edits: list) -> str:
    """Render one card per edit with its category tag and mini-lesson."""
    if not edits:
        return ('<div class="gc-card gc-good">🎉 <b>No mistakes found!</b> '
                'Your writing looks grammatically correct.</div>')
    cards = []
    for n, edit in enumerate(edits, start=1):
        color = CATEGORY_COLORS.get(edit.category, "#475569")
        cards.append(
            f'<div class="gc-card">'
            f'<span class="gc-tag" style="background:{color}">{edit.category}</span>'
            f' <b>Fix {n}:</b> {html.escape(edit.describe())}'
            f'<div class="gc-lesson">💡 {edit.lesson}</div></div>'
        )
    return "".join(cards)
