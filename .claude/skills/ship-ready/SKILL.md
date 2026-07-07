---
name: ship-ready
description: "How to take working code to genuinely shipping-ready \u2014 a verification matrix, metrics that tell the truth, docs a stranger could start from, and a final pass from a clean environment. Use this whenever the user wants to ship, launch, publish, demo, hand off, or \"make it production ready / portfolio-grade\"; before opening a repo to the public; before a release or a deadline; or when asked \"is this done?\" about any product, feature, or project."
---

# Making it shipping-ready

"It works on my machine, the tests pass" is the *starting* condition. Shipping-ready means a stranger — user, reviewer, hiring manager, another model — can run it, verify its claims, and not be lied to by its own metrics. Work through the five gates in order.

## Gate 1 — The verification matrix

Run and record all of it, freshly, not from memory:

- **Fast suite** green, with the count stated (112/112, not "tests pass").
- **Slow/integration suite** green end-to-end, with real dependencies up, budgeted for its true wall-time (a healthy 25-minute eval killed by a 7-minute watchdog looks like a failure and isn't).
- **A live demo over the real interface** — the actual protocol client, the actual CLI, the actual browser; not imported functions. In-process green and broken-over-the-wire is a real and common combination.
- **The empty/fresh states**: first run on a clean machine, empty database, no config. Products break most at zero.
- Record environment caveats honestly ("2 failures are sandbox-only: blocked tokenizer download; green on the target machine").

## Gate 2 — Metrics that tell the truth

Every number the product shows or the README claims must survive hostile scrutiny:

- **Estimates labeled as estimates**, in the data itself, not just in your head — a `chars/4` token count says so in its own `note` field; a dollar figure computed from reference prices says "reference pricing, real spend $0" *inside the report it appears in*. The reader of the number may never read the docs.
- **Every displayed metric has real backing.** Building a dashboard is a forcing function: if a card has no measured data behind it, you either build the measurement or drop the card — never ship a placeholder chart. "Live" data must actually be live, and canned data must be labeled as canned.
- **By-construction values explained**: a 100% rate that is the *designed maximum* of a synthetic eval must say so in fine print, or it reads as either a triumph or a disaster — both wrong.
- **Small n disclosed** ("n=3; results move in thirds; run-to-run variance is real"). Cherry-picked or single-run numbers are debt with interest.
- Keep the honesty symmetric: report the metric that went the *wrong* way too (a digest that costs slightly more than raw at tiny scale is a finding, not an embarrassment — explain when the crossover comes).

## Gate 3 — No silent stubs, no invisible failure

- A parameter that's accepted but not yet honored is **documented as accepted-but-ignored** — a visible stub beats an invisible no-op, which is a lie in API form.
- Every failure mode at a boundary chose **fail-open or fail-closed deliberately**, and the choice is written down (hooks that must never block a session fail open and exit 0; a destructive path with a lock contention fails closed). Unchosen failure behavior is the bug you'll ship.
- Degradation is *reported*: if the system silently falls back to a weaker mode (semantic → keyword search), the response says which mode actually served the request. Silent downgrades destroy user trust precisely when the system is limping.

## Gate 4 — Docs a stranger could start from

- **README**: why it exists, what it actually does (not aspirationally), quickstart that has been *executed as written* from a clean clone, an architecture map, and an honest design-notes/limitations section. The limitations section is what separates portfolio-grade from toy — it proves you know where the edges are.
- **Progress/decision log** current: locked decisions, hard-won facts, session log, uncommitted-work note.
- **SECURITY.md** if anything is stored or served (see the security-audit skill if available).
- Wording precision matters at the boundary of what the product does: "persists and compacts context across sessions (measured 29.6% token savings)" is shippable; "increases the context window" is not, even if it feels adjacent. Users forgive missing features; they don't forgive discovered exaggerations.

## Gate 5 — The clean-room pass

Before calling it done: from a fresh clone/environment on the target platform, follow your own quickstart verbatim. Check the repo for leftovers (debug prints, dead flags, unused imports — verify each by hand, not by trusting a linter's list), confirm nothing secret or personal is committed, confirm the gitignore matches reality (artifacts that should live on disk only, do). If publishing is deferred, say so as a decision with its trigger ("publish is a one-time step when X"), not as an ambiguity.

## The shipping-readiness verdict

End with one scoped paragraph: ready **for what**, verified **how**, accepted risks **which**, and what's deliberately out. If any gate above was skipped, the verdict names it as a known gap instead of hoping nobody asks. A product whose own claims are all verifiable is shipping-ready; everything else is a demo.
