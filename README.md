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




**Download:** <!-- DRIVE LINK HERE --> *(link to be added)*

Put both files in `weights/`:

| file | what it is | size |
| --- | --- | --- |
| `damage.pt` | 6 damage classes, Faster R-CNN ResNet-50 @640, from the `final_r50_640_40ep` run | 165 MB |
| `parts.pt` | 21 part classes, Faster R-CNN MobileNetV3 @640, from the `parts_frcnn_640` run | 73 MB |

Both come out of `notebooks/train_detectors.ipynb`, where every run saves its own
`best.pt`. The backbone is inferred from the state dict rather than recorded
alongside it, so the two files cannot be swapped by mistake — a wrong pairing fails
to load rather than running quietly wrong. On CPU they load in about two seconds and
a photograph takes about two more.

Without them the app asks for the damaged parts by hand, and the policy pipeline, the
tests and the evaluation all run unchanged.

### Switching provider

`src/claim/llm.py` speaks the OpenAI chat-completions format, which is what Groq's
endpoint is. Another provider is therefore a URL and a model name, not new code:

```bash
# in .env
LLM_BASE_URL=https://api.cerebras.ai/v1
LLM_MODEL=gpt-oss-120b
LLM_API_KEY=csk_...
```

Left unset it uses Groq, and `LLM_API_KEY` falls back to `GROQ_API_KEY`.

Groq's free tier gives 8,000 tokens a minute and 200,000 a day. Measured across these
eight sample policies, one call costs about 4,000 tokens and a full 28-claim run
about 98,500 — roughly two runs a day. I added this after hitting the daily cap
mid-evaluation. The reader asks for a strict JSON schema, so a provider that supports
only `json_object` will return JSON without that guarantee, and `parse_findings` is
stricter than that.

## The interface

`python -m streamlit run app.py`. It degrades rather than failing: without the
detector weights it asks for the damaged parts by hand and says so in the sidebar,
and without an API key it falls back to the keyword baseline and says that too.

The sidebar also lets you pick the reader. With a key configured it defaults to the
model, but the keyword baseline stays selectable and costs no tokens — I added that
after spending a day's budget testing whether buttons worked. One claim through the
model is about 3,500 tokens.

If you leave "Damaged parts" empty and upload photographs, the detectors fill it in.
Typing anything in that field overrides them.

## Sample test pack

`outputs/test_pack.md` is the system's own output rather than a hand-written example.
It holds a complete worked assessment for each of the four outcomes, with the clauses
each decision rests on, plus the two claims where the baseline disagrees with my
labels. `outputs/assessments.json` has all 28 in full, findings included.

The inputs are committed: the eight policies in `data/sample_policies/` and the 28
labelled claims in `data/test_claims.json`. Each sample policy is a synthetic
schedule bound onto an insurer's published wording, reproduced here so the pack runs
straight from a clone. Regenerate the outputs with
`python -m scripts.make_test_pack`.

Five test images sit in `data/test_images/`, taken from the held-out test split
recorded in the damage run's own `splits.json`, so the detectors never trained on
them. `python -m scripts.run_test_images` runs the detectors over them and writes
`outputs/vision_report.md`, `outputs/vision_results.json` and annotated copies with
boxes and scores into `outputs/test_images/`. That report is where the threshold
finding further down comes from. The wider evaluation, on the full 81-image split, is
in the notebooks.

## Evaluation

### Reader comparison

| reader | score | fails |
|---|---|---|
| keyword baseline (no model) | 26/28 (93%) | C09, C12 |
| language model as reader | 28/28 (100%) | — |

I wrote the keyword baseline expecting it to be a straw man, and it very nearly is
not: 93% from string matching is an uncomfortable result for anyone arguing a model
is necessary. The two claims it gets wrong are the entire argument.

**C12, unlicensed driver.** Three of the four wordings never print a Driver's Clause
at all. The only route to a rejection is General Exception 3(b) — *"driven by any
person other than a Driver as stated in the Driver's Clause"* — which is a
cross-reference to a clause that is not in the document. A keyword reader cannot
follow that, so it approved the claim.

**C09, windshield.** The baseline matched a windshield to *"fibre glass components,
30%"*. Fibre glass is a body-panel material; window glass has its own rate, and that
rate is Nil. So it reported a reduction the policy does not impose.

C09 also caught an error in my own labels. I had marked it `PARTIALLY_APPROVED` on
the assumption that any depreciation clause reduces payment. A Nil rate does not.

**The caveat matters more than the table.** That 100% has not been produced by a
deployed model — it is what a model got reading these clauses by hand.
`python -m scripts.run_groq_eval` produces the real number, and whoever runs it
should name the model that produced it.

### Vision: two approaches

The trained arm is `src/vision/`: Faster R-CNN for 21 car parts and Faster R-CNN for
6 damage types, both fine-tuned from COCO weights. The off-the-shelf arm is SmolVLM
run zero-shot in `notebooks/vlm_comparison.ipynb`, on identical held-out splits read
from each training run's own `splits.json`.

A VLM returns text, not boxes, so the per-image detection metric cannot score it. The
shared task has to be the one the pipeline actually consumes: which labels are on
this photograph. I handicapped the detectors into that comparison by discarding their
boxes, and every other allowance ran in the VLM's favour, including handing it the
closed vocabulary.

| approach | damage F1 | parts F1 | boxes |
| --- | --- | --- | --- |
| trained detectors | 0.69 | **0.90** | yes |
| SmolVLM, closed vocabulary | 0.18 | 0.09 | no |
| SmolVLM, "list every one" | 0.68 | not run | no |
| SmolVLM, one yes/no per class | 0.57 | not run | no |
| constant: name the 4 commonest classes | **0.75** | 0.12 | no |

Two things came out of this that I did not expect, and both are more useful than the
ranking.

**The first VLM score was mostly my prompt.** That 0.18 came from the model answering
with exactly one label per image, against a truth average of 3.41 labels — a recall
ceiling of 0.29 before it got anything right at all. Rewording the prompt took damage
F1 to 0.68, within noise of the detector. I had written a conclusion off the 0.18 and
had to withdraw it.

**Then the damage metric proved too weak to decide anything.** `Broken part` appears
on 88% of the damage photographs and `Dent` on 83%. A constant that ignores the image
entirely and names the four commonest classes scores 0.75 — beating every approach I
had measured, including my own.

So the decision cannot rest on damage F1, and it does not. It rests on parts and on
localisation. The parts vocabulary has 21 classes and a best constant of 0.12, where
the detector scores 0.90 against the VLM's 0.09. And only the detectors produce
boxes, which `src/vision/linking.py` needs in order to attach a damage to a part.
Without boxes there is no "scratch on the front bumper", only a bag of words, and the
claim cannot be itemised or depreciated. No amount of prompting fixes that.

One further result, which is why cause never comes from the photograph. Asked whether
a car had been damaged by fire, by a flood, or broken into, SmolVLM said yes on only
4–6% of collision photographs — so it is not simply agreeable. But it answered no to
*"is there any visible damage?"* on 30% of cars that all have visible damage. Cause is
read from the policy text and the claim form.

The figures above are the notebook's saved outputs, so they can be checked by opening
it. The 0.75 constant comes from its last cell, which takes a second and needs no GPU.

## Assumptions and trade-offs

Five assumptions hold this up. Each would break it if it turned out to be wrong.

- **Policies are digital PDFs with a text layer.** All four are. A photographed or
  scanned policy would need OCR, and nothing currently detects that case to warn
  about it.
- **Policies follow the IRDAI standard form.** Section selection and the integrity
  check both depend on it. A non-standard or foreign wording fails the integrity
  check — correctly, but unhelpfully.
- **One claim covers one vehicle and one incident.** Nothing handles a second
  incident in the same policy period.
- **The incident facts come from the claimant and are taken at face value.** This
  reads a claim form; it does not investigate one.
- **The damaged-part list is what the policy stage reasons over.** If the linking
  step gets it wrong, a perfectly correct policy reading still produces a wrong claim
  decision.

What each choice bought, and what it cost:

| choice | gained | given up |
| --- | --- | --- |
| rules decide, model reads | reproducible, explainable outcomes | cannot handle wording the rules do not anticipate |
| structural selection over retrieval | exclusions can never be dropped | breaks on a non-standard wording |
| two detectors, not one | each trains on its own labelled data | the link between them has no ground truth |
| 640 px, no mask head | runs on 2 GB of VRAM, or on CPU | about 0.017 mAP, inside the noise band |
| conservative review triggers | fewer wrong denials | more claims reach a human |
| synthetic schedules | no one's personal data in the repository | the schedule parser only reads our own format |

## Where this stops: eligibility, not quantum

The system answers whether the policy pays and what reduces the payment. It does not
produce a rupee figure, and that is a decision rather than an unfinished edge.

A payable amount is:

```
payable = Σ (part cost × (1 − depreciation%)) + labour
          − compulsory deductible − voluntary deductible      capped at IDV
```

Of those terms:

| term | source | present |
| --- | --- | --- |
| compulsory and voluntary deductible | schedule | yes |
| IDV | schedule | yes |
| the depreciation clause | policy text | yes, as a validated quotation |
| depreciation as a *number* | parsing the rate out of that clause | no |
| cost of each damaged part | nothing | no |
| repair or replace | nothing | no |

**Severity is the blocker, not the arithmetic.** The detector returns a box and a
class. "Scratch on the Back-bumper" is Rs 1,500 of buffing or an Rs 8,000 replacement
panel, and nothing in the model separates those two. Printing an amount would mean
supplying a parts price list the system does not have, which would make the output
look authoritative while being invented. Every other number in this README is
measured; that one would not be.

It is also where the real process divides. The quantum of a motor claim is assessed
by a licensed surveyor, and the stage being automated here is the inspection that
comes before that.

Money does enter at exactly one point. If the schedule carries a `repair_estimate`
and it reaches 75% of the IDV, the claim is routed to review as a constructive total
loss rather than a repair claim. `src/claim/decision.py` reads that field; none of
the eight sample schedules set it.

**The honest way to extend it** would be to take the repair estimate per part from
the operator, parse the rate out of the clause the reader already cites, apply it,
subtract the deductibles, cap at the IDV, and show every line of the arithmetic. The
amount then comes from a human and the reduction comes from the policy, with nothing
invented in between. Grading severity from the photograph — the part that would make
it automatic — needs labels this dataset does not have.

## Failure analysis

Everything here was found by probing the finished pipeline, not anticipated while
building it.

| probe | was | now |
|---|---|---|
| policy PDF carries "ignore all previous instructions" | approved on the injected clause | `NEEDS_MANUAL_REVIEW`, clause named |
| claim form leaves the cause blank | `REJECTED` | `NEEDS_MANUAL_REVIEW` |
| both front tyres burst, nothing else | `APPROVED` | `REJECTED` under Section I(b) |
| zero-depreciation add-on on the schedule | depreciation still deducted | waived, customer paid in full |
| repair estimate 86% of IDV | `PARTIALLY_APPROVED` | `NEEDS_MANUAL_REVIEW`, total loss |

**Prompt injection is the serious one.** The customer supplies the policy PDF, so its
text reaches the model's prompt. The citation validator cannot catch this, and it is
worth being precise about why: the validator checks that a quote really appears in
the cited clause, and an injected sentence really does appear. It validates
perfectly. So the document itself has to be checked before its clauses are trusted,
which is what `src/policy/integrity.py` does — instruction-shaped text, and a Section
I that lists none of the standard perils. It is not a security boundary. A production
system would verify the wording against the insurer's own published document using
the UIN printed on it.

The zero-depreciation one is the quietest. Add-ons live on the schedule, and their
wording is not in the policy document at all, so the depreciation clause reads as
though it still applies. Every customer who had paid for that add-on was being
under-paid, and nothing in the output looked wrong.

### The worst bug

Clause ids collided. Section I of all four wordings contains two lists both numbered
from 1 — the insured perils, and then the depreciation rates. The second overwrote
the first, so fire, theft, riot, earthquake and flood were missing from the clause
index entirely.

Every one of those claims would have been rejected as "not an insured peril", citing
a clause id that looked perfectly valid. Fixed by suffixing duplicate ids and
asserting uniqueness across the index. The clause count went from 22 to 37, which is
how obvious it was in hindsight and how invisible it was before.

### Still failing

- **Damage that predates the policy is approved.** Nothing compares the damage to the
  incident date. This needs claim history, or photograph metadata.
- **Only one cause is considered.** "Hit a barrier, then caught fire" is read as a
  single peril. Harmless when both are insured, wrong when one is excluded.
- **Age-based depreciation is ignored.** The standard form applies a rate by vehicle
  age to "all other parts", but only to parts that are replaced — and a claim never
  says whether a part was repaired or replaced.
- **Damaged part names are not checked** against the detector's vocabulary, so a
  typed part name that no model knows passes through silently.
- **The score threshold is costing most of the damage recall.** 0.6 was chosen once
  and never swept. On the five held-out images in `data/test_images/`, the classes it
  misses are still being proposed, just under the cut: `Paint chip` peaks at
  0.35–0.48 and never clears 0.6, while `Missing part` clears it easily. Dropping to
  0.3 takes damage-type recall from 39% to 86% with no change at all in the number of
  wrongly named types. Five images is a signal and not a recalibration — it should be
  swept on the full 81-image split, and probably set per class. See
  `outputs/vision_report.md`.
- **Damage is detected on background objects.** In one test image `Missing part`
  fires at 0.72 on a different car parked behind the subject. Nothing constrains
  detections to the vehicle being claimed for.

## Tests

13 test functions, 19 cases once the parametrised ones expand, and each one backs a
claim this README makes. `python -m pytest -q`.

Expect **14 passed, 5 skipped**. The skips are not failures: five tests read an
original wording from `data/raw_policies/`, which is not committed. Add the four
PDFs and run `python -m scripts.build_sample_policies` and it becomes 18 passed.

## Where the numbers come from

Every figure quoted above sits in a saved notebook output or a generated file, so it
can be checked without re-running anything.

| claim | where |
| --- | --- |
| located 0.86, right type 0.86 at threshold 0.6 | `notebooks/train_detectors.ipynb` |
| 78% of 777 damage objects missed, 18% found and named right | same |
| mAP@50 0.130 damage, 0.830 parts | same |
| Scratch 8%, Paint chip 0%, Other damage 0% | same |
| the VLM comparison and the constant baseline | `notebooks/vlm_comparison.ipynb` |
| 26/28 baseline, and every claim's decision | `outputs/` |
| the threshold sweep on five test images | `outputs/vision_report.md` |

The one number with nothing behind it is the language model's 28/28. It has not been
produced by a deployed model; `python -m scripts.run_groq_eval` is what produces it.

## Known limits

- The 28 claim labels are hand-written from the policy text. Two of them were already
  found wrong by the comparison above. There may well be more.
- The damage-to-part link has no ground truth, because no image in the vision dataset
  carries both part and damage labels. It is the output the entire policy stage
  consumes, and it has never been scored.
- The two stronger VLM promptings were run on damage only. The 0.09 parts figure
  comes from the weakest prompting, not the VLM's ceiling on parts — and parts is
  what the conclusion leans on.
