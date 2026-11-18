# Eval dataset changelog

## eval-v2 (not yet frozen)

eval-v1 stays as committed in the tag `eval-v1`. Every change below has a reason that holds without any system having run: the policy text, the recorded v1 intent, or a blind audit finding. System outputs are never a reason. eval-v2 is frozen before any system runs on it and is not edited afterwards.

The changes come from two audit passes over the v1 test split's judgement labels. The first labelled every case from the policy, the finding codes and the reference data, without seeing the labels or any output. The second saw the labels and anonymised outputs on cases where the two disagreed.

### Policy

- **Weekend and public holiday spend (§2.4).** Halden Group public holidays are now defined as the bank holidays of England and Wales, including substitute days, for every employee wherever they work. Before, the term was undefined, and the first audit pass could only assume Good Friday counted.
- **Receipts must match the claim (§3.2).**
  - Leaving out whole receipt lines the policy does not pay for (alcohol, room extras, personal items) is not a mismatch, provided the amount claimed equals the receipt total less those lines. Before, obeying the alcohol and hotel rules produced a mismatch under this clause.
  - Credit added to a travel card is a different item from the journeys it later pays for.
- **Standing reasons recorded by approvers (§10.6).**
  - A recorded reason states a fact and makes no ruling on later submissions.
  - It covers a later submission only for the same employee or supplier and the same requirement, and only where the fact explains that submission's spend.
  - Where it covers a business purpose or weekend spend, the requirement is noted as covered instead of not met. A non-approved supplier is still recorded as not approved, with the reason beside it.
  - This writes down the reading the v1 dataset already encoded in its memory follow-ups.

### Reference data

- The public holiday list adds 1 January 2026 and removes 24 December 2026, a company closure day that is not a bank holiday. No case is dated on either day.

### Labels and cases

- **test-judgement-positive-04** adds RECEIPT_MISMATCH (§3.2). The receipt is a 2.50 single bus ticket and 27.50 of travel card credit, against a claim for bus travel. Its recommendation, recomputed by the recommendation function, moves from none to reject. Audit support:
  - The first pass raised it at medium confidence and marked it ambiguous.
  - The second pass called the extra flag defensible and preferred it to the label.
- **test-memory-follow-up-01** now says where the lunch took place, because the first pass noted that it did not. Its label is unchanged, but it counts as a changed case.
- **test-memory-follow-up-02 and -03** keep their content and are relabelled: the precedent does not apply, so the finding is expected. This follows the first pass's blind reading. They become controls for a precedent that does not fit.
- **Recorded reasons.** Every scripted dismissal reason drops its ruling about future claims and keeps only the fact. This covers test-judgement-positive-01, -02 and -05 and dev-judgement-positive-01.
- **New memory pairs.**
  - test-judgement-positive-17 with test-memory-follow-up-04: an employee who leads Saturday stock counts.
  - test-judgement-positive-18 with test-memory-follow-up-05: an employee on a weekend on-call rota for the data centre the follow-up visits.
  - Both are built on the policy's own examples of a lasting fact, and both precedents fit their follow-up.
  - These four cases were written during the eval-v2 audit by the model named in their `drafted_by`.

### Shape changes

- Follow-ups use a `memory:` block (`precedent_from`, `code`, `precedent_applies`) in place of `expected_clearance`.
- The slice plan gives judgement-positive counts per code.
- A judgement-positive case may expect more than one judgement finding.
- `drafted_by` can record a case written in an audit session, naming the model and the audit.

### System change

- Judgement checks can look up the public holiday on the submission's date (`get_public_holiday`). v2 scores reflect this system change as well as the dataset changes.

### Scoring

- **Seeded precedents.** Each memory follow-up gets its pair's scripted reason as a precedent in every row, built against the current clause text. Eval rows no longer record precedents from their own reviews. In v1, a row that missed the paired finding left its follow-up without a precedent, and the follow-up's label could not be met.
- **Memory comparison.**
  - For follow-ups whose precedent applies, it reports the labelled-code count beside the any-code count.
  - For controls, it reports how many raised the labelled code and how many were cleared by the precedent.
  - It lists findings a follow-up raised on other codes.
  - The memory-off run is the no-precedent control.

### Known limitations

- **Conditions with no finding code.** The meal and taxi conditions (§5.1 and §6.2) have no code. On test-judgement-positive-01, -03, -10 and test-memory-follow-up-01, a PURPOSE_UNCLEAR flag raised for those conditions counts as a false positive by design.
- **Recommendations ignore fit.** A control's labelled recommendation is approve, because the recommendation function attaches a precedent by employee and code without testing fit.
- **Gaps the audit found and v2 leaves open:**
  - the open list of acceptable weekend reasons
  - stays over several days
  - drinks of uncertain strength
  - receipts too vague to check for alcohol
  - how §2.1 applies to supplier invoices
  - company exchange rates and each employee's usual workplace, which are missing from the reference data
- **Audit scope.** Only the judgement labels on the test split were audited. Rule, route and recommendation labels and the dev split had no blind pass.

### Comparability

- The memory slice is new in v2 and not comparable with v1.
- test-hard-negative-07 and -08 are easier under the new §3.2 wording and are not comparable across versions.
- v1 outputs are rescored on v2 labels only for unchanged cases.

### Blind check

Not yet run. A fresh blind pass labels every changed or new case under the v2 policy before the freeze. The rounds it takes will be recorded here.
