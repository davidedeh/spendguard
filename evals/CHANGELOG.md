# Eval dataset changelog

## eval-v2 (frozen, dataset hash `968c6684f01ecf58c3ad73bca433cf2667f8891afa7e8d6b756b1592d8959827`)

eval-v1 stays as committed in the tag `eval-v1`. Every change below has a reason that holds without any system having run: the policy text, the recorded v1 intent, or a blind audit finding. System outputs are never a reason. eval-v2 is frozen before any system runs on it and is not edited afterwards.

The changes come from two audit passes over the v1 test split's judgement labels. The first labelled every case from the policy, the finding codes and the reference data, without seeing the labels or any output. The second saw the labels and anonymised outputs on cases where the two disagreed.

### Policy

- **Weekend and public holiday spend (§2.4).**
  - Halden Group public holidays are now defined as the bank holidays of England and Wales, including substitute days, for every employee wherever they work. Before, the term was undefined, and the first audit pass could only assume Good Friday counted.
  - Added after the third audit pass: "A deadline, or a date the work was needed by, is not enough on its own. The justification must also say why the work could not have been done on an earlier working day."
    - Why: the first and third passes both named the open list of weekend reasons, blind. A bare deadline does not answer the clause's existing question of why the work could not take place on a working day, so the sentence applies that test to one common case rather than adding a rule.
    - Effect: it sides with the existing labels, and it makes test-judgement-positive-01 and dev-judgement-positive-01 easier to get right.
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

- **test-judgement-positive-04** adds RECEIPT_MISMATCH (§3.2).
  - The claim is for bus travel between two depots, but the receipt lines are "Single bus ticket 2.50" and "Travel card top‑up 27.50".
  - §3.2 says "the items on the receipt must be the items claimed for".
  - Its recommendation, recomputed by the recommendation function, moves from none to reject.
  - Audit support: the first pass raised it at medium confidence and marked it ambiguous. The second pass was shown a system's flag and classed it "defensible", siding with the system. Neither pass called the old label wrong outright.
- **test-memory-follow-up-01** now says where the lunch took place, because the first pass noted that it did not. Its label is unchanged, but it counts as a changed case.
  - After the third pass, its spend changed from preparing an executive strategy session to reviewing the quarter-end results for the quarter that ended on 31 March. The third pass found the precedent covered the strategy session only on balance, because the recorded fact names quarter-end reviews. The precedent, its paired case and the label are unchanged.
- **dev-memory-follow-up-01**, after the third pass, was changed to buy binders for report packs the team delivers to the client on Monday, instead of for an internal review meeting. The third pass found its precedent, about client deliverables with hard Monday deadlines, covered the internal meeting only on balance. After the fourth pass it was dropped under the stop rule (see Blind check). dev-judgement-positive-01 loses its scripted review and stays a weekend judgement positive with the same label, so the dev split has no memory pair.
- **Why these two edits, and what they cost.** The reason is the third pass's blind finding and the second review of the proposed fixes, not any system output; v2 has not been run. The edits still fit the facts to a label, and they make the memory slice easier for every system. The slice now tests use of a clearly fitting precedent and refusal of a clearly unfitting one. It does not test borderline fit.
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

- **Seeded precedents.** Each memory follow-up gets its pair's scripted reason as a precedent in every row, built against the current clause text. Eval rows no longer record precedents from their own reviews.
  - In v1, a row that did not keep the paired finding left its follow-up without a precedent, so the follow-up's label could not be met.
  - In the v1 baseline, follow-ups ran without their precedent as follows:

    | Row | Follow-ups without a precedent | Why |
    | --- | --- | --- |
    | Claude Haiku | 1 of 3 | the paired finding was not raised |
    | Claude Sonnet | 0 of 3 | |
    | GPT OSS 120B | 1 of 3 | the paired case failed; a second follow-up failed itself |
    | Nova Pro | 3 of 3 | two paired findings were not raised; one was removed at verification for citing a precedent that had not been supplied |
    | Claude Haiku, whole policy | 1 of 3 | the paired finding was not raised |
- **Memory comparison.**
  - It runs only the follow-ups, each with its seeded precedent.
  - For follow-ups whose precedent applies, it reports the labelled-code count beside the any-code count.
  - For controls, it reports how many raised the labelled code, were cleared by the precedent or failed.
  - It lists findings a follow-up raised on other codes.
  - The memory-off run is the no-precedent control. The labels assume the precedent, so recommendations are scored with memory on only.

### Known limitations

- **Conditions with no finding code.** The meal and taxi conditions (§5.1 and §6.2) have no code. On test-judgement-positive-01, -03, -10 and test-memory-follow-up-01, a PURPOSE_UNCLEAR flag raised for those conditions counts as a false positive by design.
- **Recommendations ignore fit.** A control's labelled recommendation is approve, because the recommendation function attaches a precedent by employee and code without testing fit.
- **Gaps the audit found and v2 leaves open:**
  - the open list of acceptable weekend reasons, now partly closed: a deadline alone is not enough
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

A fresh blind pass labels every v2 case from the policy, the reference data and what the system sees, with no labels or results.

- **Round 1 (third audit pass).**
  - Its judgement findings matched the v2 labels on all 60 cases, and its reading of each follow-up's precedent matched `precedent_applies` on all six.
  - Four changed cases were only medium confidence on their labelled answer: test-judgement-positive-01, dev-judgement-positive-01, test-memory-follow-up-01 and dev-memory-follow-up-01.
  - These led to the §2.4 deadline sentence and the two follow-up edits above.
- **Round 2 (fourth audit pass).**
  - Stop rule, set before the pass: any of those four cases still at medium confidence on its labelled answer is dropped, not edited again. A case whose inputs did not change but whose confidence falls counts only if the pass ties its doubt to the new §2.4 sentence.
  - Result: its judgement findings matched the v2 labels on all 60 cases, and its reading of each follow-up's precedent matched `precedent_applies` on all six.
  - test-hard-negative-05 and dev-hard-negative-01, which mention a renewal deadline beside a client-fixed day, stayed at no findings with high confidence: the new sentence did not flip them.
  - The fourth pass gives one confidence per case, so each medium case was read for where its doubt sits.
  - **dropped: dev-memory-follow-up-01.** The doubt is on its labelled answer: the precedent explains working on Saturday, but not why the binders could not be bought on Friday. That was its second round, so the stop rule applies.
  - **kept: dev-judgement-positive-01**, at high confidence.
  - **kept: test-judgement-positive-01.** The weekend finding is at high confidence; the pass says so. The remaining doubt is whether PURPOSE_UNCLEAR also applies because the claim does not show the lunch was away from the usual workplace (§5.1). That is the meal condition with no finding code, already listed as a known limitation.
  - **kept: test-memory-follow-up-01.**
    - The precedent call ("covers") carries no doubt.
    - The medium is about whether PURPOSE_UNCLEAR applies, because the claim does not say why the review took the claimant to Henley-on-Thames (§5.1). This is the same no-code limitation, which this changelog already names for this case.
    - The third pass noted the same question without lowering its confidence. In that pass, its medium was on the precedent's fit, which the quarter-end edit addressed.
  - **kept: test-memory-follow-up-03, a control.**
    - It is not one of the four, and was first at medium in this pass. The fourth pass still agreed with its label: the precedent does not cover.
    - The doubt is that §10.6 does not say how specific a role-based fact must be.
    - Its inputs were the same in the third pass, which rated it high. Its confidence fell with no tie to the new sentence, the case the carve-out discounts.
  - Other medium cases had unchanged inputs and doubts outside their labelled answers, mostly the §5.1 and §6.2 conditions with no code: test-judgement-positive-03 and -15, test-rule-coverage-01 and -03, and test-clean-04. No label changed.
- **Frozen.** No case, policy text or reference data changes after this point.
