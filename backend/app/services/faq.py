"""Small, local FAQ shortcut. No model, embeddings, or network required."""
import json
import re
import unicodedata
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

FAQ_PATH = Path(__file__).with_name("faq_answers.json")
STOP_WORDS = set("a an the please can could you me tell explain what is are does do how i to about".split())


def tokens(text):
    text = "".join(char for char in unicodedata.normalize("NFKD", text)
                   if not unicodedata.combining(char))
    return sorted(word for word in re.findall(r"[a-z0-9]+", text.lower())
                  if word not in STOP_WORDS)


@lru_cache(maxsize=1)
def entries():
    return json.loads(FAQ_PATH.read_text(encoding="utf-8"))


def match_faq(question, language="en", has_context=False):
    # Follow-ups, images and selected map points may change the question's meaning.
    if has_context or language.lower() not in {"en", "en-us", "en-gb"}:
        return None
    query = tokens(question)
    if not query:
        return None
    candidates = []
    for entry in entries():
        best = 0.0
        for variant in entry["questions"]:
            expected = tokens(variant)
            # Extra qualifiers (locations, dates, negation, additional questions)
            # must not disappear in an approximate match.
            if len(query) != len(expected):
                continue
            scores = [SequenceMatcher(None, a, b).ratio()
                      for a, b in zip(query, expected)]
            if all(score >= 0.85 for score in scores):
                best = max(best, sum(scores) / len(scores))
        if best >= 0.94:
            candidates.append((best, entry))
    candidates.sort(key=lambda item: item[0], reverse=True)
    if not candidates or (len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.05):
        return None
    return candidates[0][1]["answer"]
