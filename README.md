# AI motor claim assessment

Photographs of a damaged car and the customer's policy PDF go in. What comes out is a
claim decision — `APPROVED`, `PARTIALLY_APPROVED`, `REJECTED` or
`NEEDS_MANUAL_REVIEW` with the policy clauses it rests on, quoted and
checked against the document they came from.


The trained detector weights are 239 MB and as they cannot be uploaded to github due to size constraints, I have attached the drive link from where they can be copied and dropped into the weights folder.

## How a claim is decided

```
policy PDF ─► extract ─► sections ─► clauses ─┐
                                              ├─► reader ─► findings ─► rules ─► decision
photos ─► parts detector + damage detector ───┘           (validated citations)
```

The model never decides anything. It reads clauses and reports what they say, each 
finding carrying a clause id and a quotation. The rules in `src/claim/decision.py` 
turn those findings into an outcome. That means any decision can be explained, and 
re-derived, without calling the model again.

The checks run in a fixed order, and the order matters:
1. **Claim level.** Was the policy in force on the incident date? Does the cover
   include own damage at all?
2. **Cause.** Is the stated cause one of the perils listed in Section I?
3. **Exclusions.** Does a clause remove cover — for the whole claim, or only for
   particular parts?
4. **Limits.** Depreciation by material, and the deductibles.

A policy that was not in force cannot be approved no matter how favourably the
clauses read, so the schedule is checked first.

## The decisions, and why I made them

### Rules decide, the model reads

The alternative was to hand the model the clauses and the claim and ask it for an
outcome. That was not chosen for the below mentioned reasons.

An outcome from a model is not reproducible and not auditable: an insurer has to be 
able to say *this claim was reduced under this clause*, and "the model concluded" is 
not that. 
Splitting it means the model does the part it is genuinely good at: reading dense,
cross-referenced legal prose and finding the relevant sentence, and the rules do the
part that needs to be deterministic. Each finding is validated before it counts: the
clause id has to exist, and the quoted words have to actually appear in that clause.
A citation that fails validation is kept and displayed, but it cannot reach the
outcome.

### Structural section selection, not similarity search

This is the choice I expected to make differently when I started. Retrieval over a
chunked policy is the obvious shape for a document-reasoning problem, and it is what
I initially reached for, but there were two major reasons not to.
The first was budget: Groq's free tier allows 8,000 tokens a minute, and the four policies
run from about 7,000 to 61,000 tokens, so sending the whole document was never possible. 
Selecting Section I, the General Exceptions, Limitations as to Use, the Driver's Clause, 
the Deductible and the No Claim Bonus brings every one of them under 2,800 tokens.

The second reason is the major one. **Exclusions live inside the
section they qualify.** A similarity retriever ranks chunks against the claim, and a
dented bumper has nothing in common with *"whilst the insured is under the influence
of intoxicating liquor"* , so the single clause that would reject the claim is
precisely the one that does not get retrieved. Structural selection carries the
exclusions along with the cover automatically, because that is how the document is
organised.

### Two trained detectors, not a vision-language model

A VLM is the cheap route, no training, no labels, no weights to host. 
I tried both and kept the detectors. 
Parts and damage are two models rather than one because they come from two
separately labelled datasets, and no image carries both kinds of label.

### Four outcomes, not two

Approve or reject would have been simpler but doesnot reflect real production cases.
So there is a fourth outcome. The system returns `NEEDS_MANUAL_REVIEW` whenever it is
not sure: a citation that failed validation, a blank cause on the claim form, damage
that does not sit on any identified part, or a repair estimate at or above 75% of the
insured value. `PARTIALLY_APPROVED` covers the common case where the policy pays but
depreciation and deductibles reduce the amount.
The rule throughout is to fail toward review.

### Synthetic schedules on real wordings

A published policy wording is a set of rules without a customer. All four say
"as specified in the Schedule" and then leave the Schedule blank, so none of them can
decide a claim on its own, there is no policy period, no IDV, no deductible, no
add-ons.

`scripts/build_sample_policies.py`
generates eight schedules whose field structure follows the IRDAI Customer
Information Sheet, which every insurer publishes blank and binds each one onto a
real wording. The rules are real, the people are invented, and the eight cases
between them exercise every branch of the claim-level gate, including the ones that
must reject.

## The code

```
app.py                   Streamlit interface
src/policy/extract.py    PDF -> clean lines (two-column, furniture, ligatures)
src/policy/sections.py   structural sections; select what can decide a claim
src/policy/clauses.py    citable clauses with stable ids, 4 numbering styles
src/policy/schedule.py   synthetic schedules; bind to a real wording; read back
src/policy/integrity.py  is this document what it claims to be?
src/claim/decision.py    citation validation, claim-level gate, the rules
src/claim/llm.py         the model reader (any OpenAI-compatible provider)
src/claim/baseline.py    keyword reader, same interface, the no-model floor
src/claim/pipeline.py    orchestration
src/vision/linking.py    damage -> part, and when a human must look (no torch)
src/vision/detector.py   the trained detectors (torch imported lazily)
scripts/                 build the samples, evaluate, regenerate the outputs
notebooks/               training and the second vision approach, outputs saved
```

