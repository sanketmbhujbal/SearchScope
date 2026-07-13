# Day 4 Findings: Role-Based Personalization

**Query:** "policy" (matching DESIGN.md's example) | **Roles:** Engineer,
Sales, HR, Legal | **Method:** TF-IDF cosine similarity between each
candidate passage and a role's seed-topic vector (`config.ROLE_AFFINITY_TOPICS`),
blended with base retrieval relevance for demo visibility (see
`eval/run_personalization_demo.py`. This blend is explicitly not the
production design; see below).

## What worked: HR shows real, plausible differentiation

HR's top-5 is genuinely distinct from the other three roles and the
hits are plausible: a Group Policy/Active Directory doc, a concussion
management policy, a benefits-coordination policy for a health plan.
These are reasonable candidates for an HR-context query about "policy."

## What didn't work: Engineer, Sales, and Legal largely collapse into the same list

Ranks 2-5 are nearly identical across Engineer, Sales, and Legal:
`1169578`, `5589306`, `5794872` all appear in every one of their top-5s.
Only rank 1 meaningfully differs between them, and even that's shaky:
Engineer's #1 result is an HIV/AIDS workplace policy document, which has
no real connection to the seed terms (`code, deployment, debugging,
infra, api, auth`).

More telling: "Group Policy", a Windows Active Directory / IT
administration term, scored the single highest affinity in the entire
demo (0.804, HR's #1). That's a red flag, not a win. It's TF-IDF
rewarding shared surface vocabulary ("policy," "group") between the doc
and the seed terms, not any real understanding that this document has
nothing to do with HR.

## Root cause

TF-IDF bag-of-words cosine similarity against a short, hand-picked seed
list is a blunt signal on a general-purpose web corpus. MS MARCO/TREC DL
wasn't built with role-segmented enterprise content in mind, so there's
little genuine engineering/legal/sales-specific vocabulary in this corpus
for the seed terms to lock onto. The scorer ends up matching on whatever
weak lexical overlap happens to exist, which is noisy by construction.

HR's seed list (`policy, benefits, onboarding, leave, payroll, review`)
likely overlaps unusually well with this particular corpus and with the
query itself ("policy" is a literal seed term), which probably explains
why HR's affinity scores run uniformly higher across the board, not
just at rank 1, than the other three roles. That looks like it's partly
a seed-vocabulary-richness artifact rather than purely better role
modeling, and is worth being upfront about rather than reading too much
into HR's stronger showing.

## Why this doesn't undermine the Day 4 deliverable: it validates the design

Two things already documented before this run explain exactly why this
happened, and point at the right fix:

1. **DESIGN.md §5 already flags this corpus as a limitation**: "Synthetic
   enterprise corpus (Slack/Confluence/GitHub mix)... deferred to v3",
   a general web corpus was never going to have strong role-segmented
   signal, and this result is direct evidence of that gap, not a surprise.
2. **This is exactly why `role_affinity.py`'s `rerank_by_role()`
   deliberately refuses to hand-tune a blend**: a fixed TF-IDF-cosine
   formula is too blunt an instrument to trust in production. The actual
   plan (DESIGN.md §9) is `role_doc_affinity` as one signal among 20 in
   the Day 5-6 LTR model, where the ranker can learn how much to trust
   it, including learning to discount it when, as here, the signal is
   noisy, rather than a fixed weight assuming it's always meaningful.

## Possible follow-up (not required for Day 4, noted for later)

Richer, more specific seed-term lists per role (beyond a handful of
single words) would likely sharpen differentiation somewhat, but
wouldn't fix the underlying limitation: this corpus just doesn't contain
much genuinely role-segmented content. The real fix is the v3 synthetic
enterprise corpus already scoped as future work, or (per DESIGN.md §15)
Glean's actual Personal Knowledge Graph in production, which uses real
interaction history rather than a fixed keyword list against a general
web corpus.

## Framing for the README / blog post

"Role-based TF-IDF affinity showed real differentiation for HR but
collapsed toward a shared ranking for Engineer/Sales/Legal, a direct,
diagnosable consequence of running a role-segmentation signal against a
general web corpus rather than role-labeled enterprise content. This is
exactly the argument for treating role affinity as a learned LTR feature
rather than a hand-tuned formula, and for scoping a synthetic enterprise
corpus as future work rather than assuming a general corpus would show
clean role signal."
