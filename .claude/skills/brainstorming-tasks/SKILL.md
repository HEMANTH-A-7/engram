---
name: brainstorming-tasks
description: "How to think through a task before building anything \u2014 turn a vague ask into ratified design decisions and a verifiable plan. Use this at the START of any non-trivial task: new features, projects, refactors, pivots, \"how should we approach X\", \"I want to build Y\", or any request where more than one reasonable design exists. Use it even when the user sounds certain \u2014 especially then, because a confident ask often hides an unstated assumption. Skip it only for trivial one-step chores."
---

# Brainstorming a task

The goal of this phase is to reach a small set of **locked, written-down decisions** before any code or content exists. Almost every expensive failure traces back to skipping this: building the assumed solution instead of the actual requirement, or discovering a load-bearing contradiction three days in.

## 1. Restate the goal in your own words

Write one or two sentences: what does the user actually get when this is done? Not the mechanism — the outcome. If you can't state it without using the user's own phrasing, you don't understand it yet.

Then separate two things people always fuse together:

- **The requirement** — the property the user needs (e.g. "my data never leaves my machine").
- **The assumed solution** — the mechanism they reached for (e.g. "so I need a local LLM").

Interrogate the link between them. The single highest-value brainstorming question is: *"is the assumed mechanism actually load-bearing for the requirement?"* Often it isn't — local storage was the real requirement, the local LLM was never load-bearing for it, and dropping it removed the slowest, flakiest component of the whole system. Ask "why do I even need X?" about the most expensive component in the plan, on purpose, every time.

## 2. Surface tensions early — out loud

Scan the constraints for pairs that fight each other: offline vs. cloud-priced cost metrics, speed vs. accuracy, "no new dependencies" vs. a feature that begs for a library, seamlessness vs. explicit user control. Don't resolve these silently. Name each tension to the user and propose a resolution (e.g. "compute dollar-equivalent cost from a published price table so metrics stay meaningful while offline"). A tension resolved explicitly becomes a design principle; one resolved silently becomes a bug report later.

## 3. Enumerate genuinely different options

For any real design fork, list 2–3 approaches that differ in *kind*, not degree. For each: what it costs, what it risks, what it makes easy later. If all your options are variations of one idea, you haven't brainstormed — you've decorated a foregone conclusion.

Beware the naive split. Empirically test assumptions that sound obvious ("small model = fast but weak, big model = slow but strong") before baking them into architecture — a 10-sample manual test can reverse the assumption entirely and change the design from a static split to a cascade.

## 4. Sort the decisions: yours vs. the user's

Two piles:

- **Conventional-default decisions** — naming, file layout, idiomatic library use. Pick the obvious option, state it, move on. Asking about these wastes the user's attention.
- **The user's decisions** — anything that trades off things only they can weigh: scope cuts, cost vs. quality, what data is disposable, pivots. Present these as concrete options with tradeoffs and get an explicit answer *before* building. One ratified sentence ("user picked: keep local, fix write speed") prevents a rebuilt week.

## 5. Lock decisions in writing

Keep a numbered decision table (D1, D2, …) with the choice, the reason, and the date, in a file that survives the session (PROGRESS.md, a design doc, the repo). Format matters less than existence. This is what lets any future session — or any other model — pick up without replaying history, and what makes "wait, why did we do it this way?" answerable in one lookup.

## 6. Define "done" and "verified" before starting

Before executing, write down:

- What observable behavior proves this works? (a passing test, a measured number, a live demo over the real interface)
- What's the baseline it will be compared against? Build the measurement **before** the improvement — a benchmark harness right after the baseline exists, before the clever component does, so every later change gets clean before/after numbers.
- What's explicitly out of scope? Say it now, honestly, or scope will be negotiated implicitly through fatigue later.

## 7. Spike the risky unknown first

If the plan hinges on an unproven assumption ("the retrieval still works if we drop pipeline stage 3"), spend an hour on a throwaway spike that proves or kills it before any production code changes. The spike's only deliverable is a yes/no. Gate the plan on it, and throw the spike code away without guilt.

## Anti-patterns to catch in yourself

- Jumping to the first workable design and back-justifying it.
- Asking the user questions the codebase or a sensible default already answers.
- Treating the user's proposed mechanism as sacred instead of interrogating it.
- Planning the happy path only — where does this fail, and is failing open or closed the right call there?
- A plan with no numbers in it: no budget, no latency target, no size estimate. If nothing is quantified, nothing can be verified.

## Output of this phase

A short written plan: goal restated, tensions named and resolved, locked decision table, done/verified criteria, out-of-scope list, and the first risky spike if one exists. If the task is big, order the build so that measurement infrastructure comes right after the baseline. Then move to execution (see the task-execution skill if available).
