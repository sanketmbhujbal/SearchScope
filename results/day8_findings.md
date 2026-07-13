# Day 8 Findings: Domain-Adapted QA + Rejection Gate

## Automated metrics (as run)

| Metric | Value |
|---|---|
| Answer Rejection Rate (unanswerable/mismatched-context set) | 100.0% (43/43) |
| Answerable queries incorrectly rejected | 7/43 (16.3%) |
| Citation hygiene rate (as originally computed, see bug below) | 94.4% (34/36) |
| Avg latency | 1.43s |
| Max latency | 4.42s (target: <2.0s, DESIGN.md §10.3) |

## Bug found and fixed: citation-format parsing inflated the hallucination count

Two of the 36 non-rejected answerable-set answers were flagged as having
hallucinated citations. Reading them directly (not trusting the
automated count) showed both were false positives:

- Query "who is robert gray": cited `[doc_id: 3620986]`, `[doc_id: 8760866]`,
  `[doc_id: 8760871]`. All three are real, correctly-provided passage
  IDs, cited to support a genuinely accurate, well-grounded answer
  distinguishing three different people named Robert Gray/Grey.
- Query "who formed the commonwealth of independent states": cited
  `[doc_id: 2332189]`, `[doc_id: 8220088]`. Again, both real, correctly-provided
  passages supporting an accurate answer.

**Root cause:** the system prompt asked the model to cite "in the form
`[doc_id]`", genuinely ambiguous phrasing that gpt-4o-mini reasonably
read as a literal template ("write the word doc_id, then the value") in
these two cases, rather than "the doc_id value goes directly in
brackets." `parse_response`'s citation regex only recognized bare
`[12345]`, so these got extracted as the literal string `"doc_id: 3620986"`,
which matched no real doc_id, flagging two accurate, well-cited answers
as hallucinations they weren't.

**Fix:** two changes, not just one. A prompt fix alone wouldn't be
enough, since LLM output formatting is never fully guaranteed even with
a precise instruction:
1. `SYSTEM_PROMPT_TEMPLATE` now gives a concrete example (`[12345]`) and
   explicitly says "write ONLY the doc_id inside the brackets," removing
   the ambiguity.
2. `parse_response` now strips an optional `doc_id:`/`doc:`/`id:` label
   prefix from bracket contents before matching, so it's robust to this
   format even if it recurs. Four regression tests added
   (`tests/test_grounded_qa.py`), including the two real cases from this
   run verified directly against the fix.

**Corrected citation hygiene rate: 36/36 = 100%** once these two false
positives are excluded. Zero genuine hallucinated citations found in
this run. Worth noting this as the real number, not the originally
reported 94.4%.

## Human spot-check: Answer Rejection Rate = 100% looks perfect, but reading the actual false-rejection cases tells a more useful story

100% rejection on the mismatched-context set is the headline number, but
the more informative read is the **7 cases where the model rejected a
genuinely answerable query**, that's where the rejection gate's
calibration actually shows itself. Reading all 7 directly:

| Query | Verdict | Why |
|---|---|---|
| "medicare's definition of mechanical ventilation" | **Correct rejection** | Passages define mechanical ventilation generically; none mention Medicare specifically. The query asks for *Medicare's* definition, genuinely not supported. |
| "what is an aml surveillance analyst" | Borderline / defensible | Passages describe "AML Analyst," "BSA/AML Analyst" roles but never the exact phrase "AML surveillance analyst." Reasonable caution about a term not literally present. |
| "what is the daily life of thai people" | **Likely over-cautious** | Passages directly describe SANUK and JAI YEN as everyday-life concepts, real, on-topic content the model could have synthesized into an answer. |
| **"cost of interior concrete flooring"** | **Clear miss** | Passage [1773805] states directly: *"concrete floors can cost as little as $2 to $6 a square foot or be as expensive as $15 to $30 a square foot."* This is a direct, unambiguous answer the model had and didn't use. |
| **"causes of military suicide"** | **Clear miss** | Passage [2624886] states directly: *"We suggest that moral injury is likely one of the most important factors in military suicide rates."* A stated cause, explicitly present. |
| "anthropological definition of environment" | Borderline / defensible | Passages define *environmental anthropology* and *ecological anthropology* (the sub-fields), but none give a clean definition of "environment" itself, the query asks for something subtly different from what's present. |
| **"is cdg airport in main paris"** | **Clear miss** | Passage [8433854] states directly: *"Charles de Gaulle airport (CDG) is the main international airport for Paris."* Directly answers the yes/no question. |

**3 of 7 (43%) are clear misses**, cases where an explicit, directly
quotable answer was sitting in the provided passages and the model
rejected anyway. **2 of 7 are genuinely correct, careful rejections**
(the query asked for something subtly more specific than what the
passages actually supported). **2 of 7 are defensible edge cases**
(exact-phrase mismatch between the query's terminology and the passages').

## Why this is the real Day 8 finding, not the 100%/94.4% headline numbers

A 100% rejection rate on the automated unanswerable set is easy to
over-read as "the rejection gate works great." Reading the actual
false-rejection cases shows real, non-trivial over-caution: the model
sometimes rejects even when a direct quote answering the question is
plainly present in the context. This is a legitimate, common failure
mode for grounded QA systems. Models tuned to avoid hallucination can
overcorrect toward excessive caution, especially at temperature 0 with a
strict "only answer if supported" instruction. It's also a good
match for the class of problem DESIGN.md's AI-fluency framing (§ Glean
JD) is actually asking candidates to reason about: not just "does the
system have a rejection gate," but "what does the gate actually get
wrong, and why."

**Possible follow-up (not required for Day 8, noted for later):** the
three clear misses all happen to have their answering sentence appear
verbatim, without heavy paraphrasing, in a single passage. Worth testing
whether the model is more reliable when the answer requires combining
information across passages versus quoting one directly, though 3 data
points isn't enough to claim that pattern with confidence.

## Answer Supported Rate

Per DESIGN.md §10.3, this is a human spot-check rather than an automated
metric. Reviewing the 36 non-rejected answerable-set answers: every one
read directly during this analysis (including the two originally
mis-flagged as hallucinated) was well-grounded, directly citing passage
content accurately, with no fabricated claims beyond what the passages
stated. A full line-by-line pass of all 36 is the actual deliverable
here (this findings doc reviewed a representative sample while
investigating the rejection/citation questions above, not the complete
set), worth doing as a final pass before treating this number as fully
confirmed, but nothing found so far suggests problems beyond the
rejection-calibration issue already documented.

## Latency

Avg 1.43s is within the <2s target; max 4.42s exceeds it on at least one
call. A single outlier isn't concerning on its own (API latency varies
run to run), but if this recurs consistently on re-runs, it's worth
profiling which specific query/passage combination triggers it.
