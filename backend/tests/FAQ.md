# Local FAQ answers

Edit `app/services/faq_answers.json` to add or revise approved answers. Each entry
has a stable `id`, a list of example `questions`, and one pre-written `answer`.
Restart the backend after edits: the file is cached in memory.

The chat route checks these answers before retrieval, forecast downloads or
Gemma. Matching ignores capitalization, punctuation and a small set of filler
words, and allows minor spelling differences. Add explicit question variants
for paraphrases; this is deliberately not unrestricted semantic matching.
Ambiguous matches and extra qualifiers fall through to the existing pipeline.

This initial FAQ is English-only and skips requests with conversation history,
an image, or populated map context, to avoid replacing context-sensitive replies.
Answers describe app functionality, not medical advice or changing forecast values.
Keep coverage and forecast-window answers synchronized with lookup configuration.

Run `venv/bin/python -m unittest discover -s tests` from the backend directory.
The route test verifies that a matching FAQ does not call retrieval, NOAA lookup,
or Gemma. These tests are local and incur no model API charges.
