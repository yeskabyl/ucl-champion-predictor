# Test fixtures

`cl_2024-25.txt` and `cl_2025-26.txt` are unmodified copies of files from https://github.com/openfootball/champions-league
(CC0 1.0 public domain). They are real match results, used so the parser and the bracket logic are tested offline
against real data. Unit tests that need *invented* data build it in code (see `tests/helpers.py`).
