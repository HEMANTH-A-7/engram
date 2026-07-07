---
name: task-execution
description: "How to execute a planned task so that progress is real, verified, and survivable across sessions \u2014 incremental buckets, before/after numbers, honest progress logs, and disciplined deviation handling. Use this whenever carrying out any multi-step piece of work: implementing a feature, working through a plan, a long refactor, a migration, or any session expected to span more than a couple of tool calls. Use it even for solo work with no reviewer \u2014 the discipline is for your future self and whichever model continues the work."
---

# Executing tasks

Execution quality is not about typing speed. It's about making sure every increment is *verified* before the next one starts, and that the state of the work is *recoverable* by someone (or some model) with zero memory of this session.

## 1. Work in buckets with checkpoints

Break the plan into ordered buckets, each independently verifiable and each ending in a checkpoint: tests green, a number measured, a demo run. Keep a live task list — one bucket in progress at a time, marked complete only when *fully* done. A bucket that's 90% done is 0% done for checkpoint purposes: failing tests, partial implementations, and unresolved errors keep it open. The discipline point is refusing to self-report progress that verification hasn't confirmed.

Order buckets so measurement comes early: baseline first, then the benchmark/harness that measures it, and only then the clever components. Every component built after the harness gets clean before/after numbers for free. Built the other way around, you'll be retro-fitting measurements onto claims you already made.

## 2. Verify at the boundary the user actually touches

Unit tests passing is necessary, never sufficient. Each bucket's checkpoint should exercise the realest available interface: the real protocol client instead of imported functions, the real subprocess instead of in-process calls, the real data file instead of a fixture. In-process tests can pass for months while the real transport is broken — chatter on stdout that corrupts a JSON-RPC wire is invisible to every test except one that drives an actual client. "Tests pass" and "works" are different claims; know which one you're making.

When you measure, isolate: time the new path against a temp/scratch environment, never against live user data, and never claim a speedup you measured once — verify it twice, note the conditions.

## 3. Keep a progress log with hard-won facts

Maintain a running log (PROGRESS.md or equivalent) with two distinct sections:

- **Session log** — what was done, what passed, what's uncommitted, what's next. Written so a fresh session gets full context without replaying history.
- **Hard-won facts** — every config trap, API quirk, and environmental gotcha that cost more than ten minutes to learn, written as one dense bullet each with the *why*. Label it "don't relearn these". This section compounds: it saves hours per future session, which is the difference between a project that accelerates and one that re-fights old battles.

Convert relative dates to absolute ones. Record negative results too ("X doesn't work because Y") — they're the most expensive knowledge to regenerate.

## 4. One logical change per commit, and only when asked

Each commit is one coherent change with a message stating what and why. Batch-committing a week of mixed work destroys bisectability. If the user has a "never commit/push without an explicit ask" rule — or any standing rule — treat it as an invariant, not a suggestion; note uncommitted work in the progress log with a suggested commit split so the eventual commit is easy.

## 5. Handle deviations honestly

Plans meet reality. When you deviate from an approved plan — dropping a feature, changing a mechanism — do three things: say so explicitly (don't let the user discover it), get the deviation ratified if it changes anything they decided, and record it in the log with the reason. A deviation ratified in one sentence is a decision; a silent one is a betrayal of the plan's whole purpose.

When genuinely blocked, don't thrash between approaches. Diagnose first (see the root-cause-debugging skill if available), create a task naming the blocker, and either resolve it or surface it. Three failed attempts at the same approach is data; ten is negligence.

## 6. Respect invariants like they're load-bearing — because they are

Projects accumulate invariants: "nothing may print to stdout in the server process", "this dataset is real user data, never reset it", "hooks must fail open". Before touching a subsystem, re-read its invariants. Most catastrophic regressions are an invariant violated by someone who didn't know it existed — which is why they belong in a file every session reads first (CLAUDE.md, a README header), not in one person's memory.

## 7. End-of-session hygiene

Before stopping: update the progress log, list uncommitted changes with a suggested commit structure, state what's verified vs. merely written, and name the next action. The test of good execution hygiene: could a different model, tomorrow, continue this work from the written state alone? If not, the session isn't finished — the write-up is part of the work.

## Anti-patterns to catch in yourself

- Marking a task complete because the code is written rather than because verification passed.
- Measuring your improvement against a guess instead of a recorded baseline.
- Rediscovering the same environment quirk twice (it wasn't logged the first time).
- Ten changes in flight simultaneously — nothing attributable, nothing bisectable.
- Progress reports that describe effort ("worked on the parser") instead of state ("parser handles nested case; fuzz test still failing on empty input").
