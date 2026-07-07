---
name: root-cause-debugging
description: "How to find and fix bugs at the root \u2014 reproduce first, distinguish failure modes, suspect the test as much as the code, and prove the fix. Use this whenever something is broken, flaky, slow, or weird: failing tests, intermittent errors, \"it keeps timing out\", \"it worked yesterday\", crashes, wrong output, or a bug hunt across a codebase. Also use it BEFORE blaming an external component (model, library, network) \u2014 the methodology exists precisely because the obvious suspect is usually innocent."
---

# Root-cause debugging

A bug isn't fixed when the symptom disappears. It's fixed when you can state the mechanism — *this* caused *that* because *why* — and a regression test pins it. Everything below serves getting to that sentence.

## 1. Reproduce before you theorize

Get the failure happening on demand, in the smallest setup that still fails. Until then you have an anecdote, not a bug. If it won't reproduce, that fact *is* the lead: what differs between the failing environment and yours? (Fresh vs. warm state, real data file vs. tmp fixture, first run vs. cached, one process vs. two.) Some of the best bugs only exist in one of those worlds — a schema migration bug can hide forever because unit tests always use fresh databases and only the real, aged data file has the old shape.

## 2. Read the actual error, all of it

The bottom line of a traceback is where it *died*, not where it went *wrong*. Read upward until you find the first frame in code you control. Read the exception message word by word — "unable to open database file" is about paths and lifetimes, not permissions or schemas. Resist pattern-matching to the last bug you saw.

## 3. Distinguish failure modes before diagnosing

Three modes that get confused constantly, with opposite remedies:

- **Slow** — the operation is healthy but takes longer than someone's patience. Check: does it succeed with a generous budget? A watchdog killing healthy 25-minute runs at 7 minutes looks exactly like "the process hangs". Diagnose wall-time before blaming the component.
- **Flaky** — intermittent, load- or timing- or randomness-dependent. Check: run it N times; what's the actual rate? A 2/3 success rate is a different beast from 0/50.
- **Systematic** — fails the same way every time. Check: is the "randomness" actually deterministic? Six identical wrong outputs across six retries is not bad luck — it's a structural mismatch (e.g. a model echoing the schema wrapper it was shown), and blind retrying will *never* recover it. Systematic failures deserve structural fixes: repair the malformed-but-recoverable shape, change the prompt/input, fix the contract — don't burn retries.

## 4. Suspect the test, the harness, and the eval data too

The bug is in your assertion, fixture, or eval set as often as in the code. Classic cases: a test asserting the wrong expected outcome (the formula was right; the expectation was wrong), and eval data that poisons its own measurement (an "update" sentence that restates the stale value it replaces, so any leak-check false-positives even under perfect behavior). When a failure makes no sense, re-derive what *should* happen from first principles before touching the implementation.

## 5. Localize by bisection, not intuition

Cut the space in half repeatedly: which layer, which component, which commit, which input? Prefer cheap experiments that eliminate half the hypotheses over expensive ones that confirm a favorite. Instrument the boundary between "definitely fine" and "definitely broken" and move it inward. Keep a scratch note of hypotheses eliminated — under fatigue you *will* re-test the same guess twice.

## 6. Fix the mechanism, not the symptom

Once the mechanism is stated, choose the fix at the right depth: a null-guard where a null is legitimate is a fix; a null-guard hiding an upstream contract violation is a landmine. Fixes that widen tolerance (lenient parsing, retries, defaults) are legitimate *when the intolerance was the bug* — e.g. a required field that a flaky producer omits, where the field is never actually read downstream. Say which kind you're shipping.

## 7. Prove the fix — red, then green

Run the failing case before and after. If you can, revert the fix and watch it fail again. When the original bug no longer reproduces even *without* the fix (environments drift, dependencies fix things upstream), don't claim red→green — label the test a **guard by construction** and keep the fix as defense-in-depth. That honesty matters: a test that "proves" something it never demonstrated is documentation that lies.

Then write the regression test at the boundary where the bug lived — if it corrupted a wire protocol, the test drives a real client over that wire, not the function in-process.

## 8. Log the autopsy

Every bug that cost over ten minutes earns a bullet in the project's hard-won-facts log: symptom, mechanism, fix, and the generalizable lesson ("an external timeout must budget for the workload's real wall-time", "tests pass ≠ works over the real transport"). This log is how a codebase gets easier to debug over time instead of harder.

## Bug-hunt mode (finding bugs nobody reported)

When asked to "find all the bugs", don't just read linearly. Hunt where bugs cluster:

- **Boundaries**: every trust/serialization/process boundary — what happens to malformed input, empty input, the wrong type, concurrent access?
- **State lifecycle**: creation, migration, deletion. What happens to *old* data when the schema changes? What's the behavior on an empty store, on double-init, on a half-written file?
- **Error paths**: read every `except`/`catch` — what do they swallow? Fail-open where it must fail closed (or vice versa)?
- **Unreachable-by-design code**: thresholds nothing can cross, flags no caller sets, ids no tool exposes. Reachability analysis by hand — "can any real caller actually trigger this?" — finds bugs no test does, because nobody wrote a test for a path they didn't realize was dead.
- **Math edge cases**: normalizations when the denominator is one item, floors that equal thresholds (making a condition permanently unreachable), decays as t→∞.
- Verify each candidate with a concrete reproduction before reporting it; a static "unused/suspicious" list is mostly false positives until proven otherwise.
