# Limitations and next steps

## Known limitations

- The eval is small: 21 expected judgement findings and one run per row, so it cannot separate close models or measure run-to-run variation.
- About 4 in 10 of the primary model's judgement findings are wrong, which sends more claims to people than necessary.
- Nothing checks that a precedent fits the claim it clears; the approver is the safeguard.
- The policy fits in the prompt. A policy too large for it is untested.
- The CloudFormation role the deploy uses has full administrator rights, accepted for a single-owner demo account.
- The finance system is a mock, and the demo reuses fixed claim IDs, so running it again returns the earlier submissions.

## Future improvements

- **A demo web UI.** Today claims arrive by signed webhook and approvers decide through the API with a bearer token. The main gap is an approver view: the pending queue, each finding with its quoted evidence and clause, any precedent shown as a note, and approve or reject with dismissal reasons. Receipt capture would complete it: the claimant drops in a receipt photo or PDF, a vision model extracts the merchant, date, amount and line items into a pre-filled form, and the claimant corrects it and adds the purpose before submitting. Today the system assumes receipt text arrives already transcribed. Capture would need an extraction eval against real receipts, and injection defences for text inside images.
- **Tiered auto-approve thresholds by category**, for example a higher one for travel and accommodation. Today one threshold covers every category.
- **Fit-aware recommendations.** A precedent annotates a finding by employee or supplier and code, and a finding with a precedent recommends approve, even where the recorded fact does not explain this spend. Recommending approve only where the precedent fits would need a judgement the recommendation function does not make today.
- **Checking precedent fit.** With memory on, the primary model cleared both eval controls, where the recorded fact does not explain the spend, and verification checks only that a cited precedent was supplied. Making a clearance quote the fact that explains this submission, and verifying that quote as verification already checks quotes from the submission, is worth exploring.
- **A model reflection pass in verification**, deferred in ADR 0001 because it costs an extra call and is not deterministic; worth adding only if measured precision is poor.
- **Grow the dev split** (it has 10 cases and no memory pair since eval-v2), then tune prompts on it for judgement precision and precedent fit.
