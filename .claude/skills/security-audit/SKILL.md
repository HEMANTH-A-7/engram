---
name: security-audit
description: "How to run a security and production-readiness audit that produces a defensible verdict \u2014 trust model first, full coverage, dependency audit, and honestly documented accepted risks. Use this whenever asked to audit code or a project for security, check whether something is \"safe to ship/expose/publish\", do a hardening pass, review before open-sourcing, or when the user mentions vulnerabilities, secrets, permissions, injection, or \"is this production ready\" in a security sense."
---

# Security audit

An audit's product is not a list of scary findings — it's a **verdict scoped to a trust model**, plus fixes for what's real and honest documentation for what's accepted. "Secure" is not a property a system has; "production-ready for its stated scope" is.

## 1. Write the trust model first

Before reading a line of code, state: who can reach this system, with what privileges, and what is *by design*? (e.g. "local, single-user, no auth — by design; the boundary is the machine's user account.") Every finding gets judged against this. Without it you'll report non-findings ("no authentication!" on a deliberately local tool) and miss real ones (a loopback-bound server that a rebinding webpage can still reach *through the victim's own browser* — inside the trust boundary, and genuinely exploitable). If the project has no written trust model, writing one is finding #1 — put it in SECURITY.md.

## 2. Review everything, and say what "everything" was

Sampled audits produce false confidence. Enumerate the file list, read every source file, and state the coverage in the report ("reviewed all N project files"). While reading, check each of these classes — walk the list explicitly rather than trusting ambient vigilance:

- **Injection**: every SQL statement parameterized (no string-built queries anywhere, including test helpers); every shell invocation without `shell=True`/string concat; every HTML sink using safe assignment (`textContent`, not `innerHTML`) — grep for the dangerous forms, don't skim for them.
- **Deserialization & dynamic code**: no `eval`/`exec`/`pickle.loads` on data that crosses any boundary.
- **Secrets**: nothing real in the repo or history; `.env` contents actually dummy (verify the values, don't trust the filename); keys never logged.
- **Data at rest**: permissions on stores and directories (a 644 database of personal memory is world-readable on a shared machine — chmod 600/700, enforced idempotently in code, not in a README).
- **Network surface**: bind addresses, Host-header validation (DNS-rebinding guard on anything serving loopback HTTP), what's reachable pre-auth.
- **Input validation at trust boundaries**: validate *before* expensive or destructive operations (a timestamp stored verbatim into ordering-sensitive columns silently corrupts; reject at the boundary), clamp numeric params, bound sizes.
- **Concurrency on shared state**: two processes touching one data root need a real cross-process lock (flock), not an in-process one; test contention with a genuine second process.
- **Licenses**: dependencies compatible with the project's intended distribution.

## 3. Audit the dependency tree with a tool, then triage by hand

Run the ecosystem's audit tool (`pip-audit`, `npm audit`, `cargo audit`) over the *locked* tree. For each hit, answer: is a fixed release available, is the vulnerable path actually reachable from this code, and what does exploitation require? A transitive CVE with **no fixed release** whose exploitation requires same-user filesystem access is not a fire drill for a single-user local tool — it's an **accepted risk**: document it with the CVE id, why it's acceptable *under this trust model*, and the concrete condition for re-checking ("when the parent dependency bumps"). Silently ignoring it and loudly panicking are both wrong.

## 4. Prompt-injection and data-poisoning (for AI-adjacent systems)

If the system stores content that later re-enters a model's context (memories, notes, RAG chunks), assume some of it is adversarial. Ask what the blast radius is: can injected content trigger tool calls, or is it bounded to "the model believes wrong things"? Schema-constrained outputs and narrow tool contracts shrink the blast radius; document the residual as an accepted risk with its bound stated.

## 5. Verify fixes like you verify bugs

Every hardening change gets a test that exercises the *attack*, not the code: send the evil Host header and expect 400, probe the flock from a real second process, check the on-disk permission bits after init. Re-run the full suite after hardening — security fixes regress functionality embarrassingly often (e.g. host allowlists breaking test clients whose default Host isn't localhost — that's the guard *working*; fix the test's base URL, not the guard).

## 6. Write the two documents

- **SECURITY.md**: trust model, hardenings table (what/why), accepted risks with re-check conditions, and what the system deliberately does *not* defend against. This file is the difference between "we thought about it" and folklore.
- **The verdict**: one paragraph, scoped. "Production-ready for its stated scope (X), with accepted risks A, B" — plus honest caveats about what the system actually does versus what marketing language might imply. A verdict that can't name its scope isn't a verdict.

## Anti-patterns to catch in yourself

- Findings ranked by scariness instead of by reachability under the trust model.
- "No fixed release exists" reported as if the project did something wrong — triage it, document it, set the re-check trigger.
- Trusting a static analyzer's list (of unused code, of vulnerabilities) without hand-verifying — such lists are mostly false positives and burying real findings in them destroys the report's credibility.
- Auditing only the app code and skipping test helpers, scripts, and CI — injection in a test helper is still injection.
- A "passed the audit" summary with no record of what was checked; six months later nobody can tell what the audit covered.
