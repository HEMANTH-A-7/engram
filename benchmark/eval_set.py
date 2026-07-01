"""Bucket 2 eval set: synthetic stable facts + queries for recall@k measurement.

v1 only covers (a) stable facts. Contradiction/update chains are added once the
conflict resolver exists (Bucket 3) and forgetting cases once consolidation
exists (Bucket 4) — each extends this eval set rather than replacing it, so
recall@k stays comparable across buckets.

Each case pairs one or more source facts with a query and the substrings that
must appear in a retrieved hit for that query to count as "recalled".
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EvalCase:
    id: str
    facts: tuple[str, ...]
    query: str
    expected_substrings: tuple[str, ...]


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
