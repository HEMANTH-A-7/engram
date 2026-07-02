"""Eval set: synthetic facts + queries for recall@k and conflict-resolution
measurement.

`EVAL_CASES` (Bucket 2) covers stable facts for recall@k. `CONFLICT_CASES`
(Bucket 3) covers facts that get contradicted/updated within a session, for
the conflict-resolution-accuracy metric. Forgetting cases will be added once
consolidation exists (Bucket 4) — each addition extends this eval set rather
than replacing it, so recall@k stays comparable across buckets.

Each `EvalCase` pairs one or more source facts with a query and the
substrings that must appear in a retrieved hit for that query to count as
"recalled". Each `ConflictCase` is an ordered sequence of updates to the same
(subject, relation); `current_substrings` must appear in top-k for a
correctly-resolved query, while `stale_substrings` (the superseded value) are
what a naive, non-resolving baseline is expected to sometimes leak instead.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    facts: tuple[str, ...]
    query: str
    expected_substrings: tuple[str, ...]


@dataclass(frozen=True)
class ConflictCase:
    id: str
    updates: tuple[str, ...]
    query: str
    current_substrings: tuple[str, ...]
    stale_substrings: tuple[str, ...]


EVAL_CASES: tuple[EvalCase, ...] = (
    EvalCase(
        id="lovelace-algorithm",
        facts=("Ada Lovelace wrote the first algorithm intended for a machine, Babbage's Analytical Engine.",),
        query="Who wrote the first algorithm for a mechanical computer?",
        expected_substrings=("Ada", "Lovelace"),
    ),
    EvalCase(
        id="turing-test",
        facts=("Alan Turing proposed the Turing test in his 1950 paper 'Computing Machinery and Intelligence'.",),
        query="Who proposed the Turing test?",
        expected_substrings=("Turing",),
    ),
    EvalCase(
        id="hopper-compiler",
        facts=("Grace Hopper developed the first compiler and popularized the term 'debugging'.",),
        query="Who developed the first compiler?",
        expected_substrings=("Grace", "Hopper"),
    ),
    EvalCase(
        id="python-creator",
        facts=("The Python programming language was created by Guido van Rossum and first released in 1991.",),
        query="Who created Python?",
        expected_substrings=("Guido", "Rossum"),
    ),
    EvalCase(
        id="linux-creator",
        facts=("Linus Torvalds created the Linux kernel in 1991 and later created Git in 2005.",),
        query="Who created the Linux kernel?",
        expected_substrings=("Linus", "Torvalds"),
    ),
    EvalCase(
        id="web-inventor",
        facts=("Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN.",),
        query="Who invented the World Wide Web?",
        expected_substrings=("Tim", "Berners"),
    ),
    EvalCase(
        id="hamilton-apollo",
        facts=("Margaret Hamilton led the team that developed the onboard flight software for NASA's Apollo program.",),
        query="Who led development of the Apollo program's flight software?",
        expected_substrings=("Margaret", "Hamilton"),
    ),
    EvalCase(
        id="mccarthy-ai",
        facts=("John McCarthy coined the term 'Artificial Intelligence' in 1955 and later created the Lisp programming language.",),
        query="Who coined the term Artificial Intelligence?",
        expected_substrings=("McCarthy",),
    ),
)


CONFLICT_CASES: tuple[ConflictCase, ...] = (
    ConflictCase(
        id="alice-city",
        updates=(
            "Alice lives in Boston.",
            "Alice now lives in Seattle.",
        ),
        query="Where does Alice currently live?",
        current_substrings=("Seattle",),
        stale_substrings=("Boston",),
    ),
    ConflictCase(
        id="atlas-project-lead",
        updates=(
            "The project lead for Atlas is Raj.",
            "The project lead for Atlas is now Priya.",
        ),
        query="Who is the current project lead for Atlas?",
        current_substrings=("Priya",),
        stale_substrings=("Raj",),
    ),
    ConflictCase(
        id="acme-ceo",
        updates=(
            "The CEO of Acme Corp is Diana.",
            "The CEO of Acme Corp is now Marcus.",
        ),
        query="Who is the current CEO of Acme Corp?",
        current_substrings=("Marcus",),
        stale_substrings=("Diana",),
    ),
)
