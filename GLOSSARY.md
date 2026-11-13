# Glossary

**Submission**
One expense claim or supplier invoice sent for checking and decision.
_Avoid_: request

**Expense claim**
A submission in which an employee asks to be reimbursed for money they spent.

**Supplier invoice**
A submission in which a supplier asks to be paid for goods or services.

**Requester**
The employee accountable for a submission's spend: the claimant for an expense claim; for a supplier invoice, whoever raised the purchase order, or the cost centre's owner when there is none. An approver may never decide a submission for which they are the requester.
_Avoid_: submitter

**Finding**
A specific way a submission breaks the policy, tied to one finding code, one policy clause and the evidence for it.
_Avoid_: violation, flag, issue

**Finding code**
One entry from the closed set of ways a submission can break the policy, such as `MISSING_PO`. Every finding has exactly one.

**Severity**
How serious a finding code is (`low`, `medium` or `high`). The policy fixes it for each code; it is never judged case by case.

**Rule check**
A check whose outcome follows mechanically from the submitted figures and company records, such as a budget or an approved-supplier list. It always runs when it applies to the submission type.

**Judgement check**
A check that needs a reading of free text, such as whether a business purpose is plausible or whether a receipt matches the claim. It is the only kind of check a model performs.

**Approver**
The person who decides a submission that needs human review, and who answers for that decision.

**Disposition**
An approver's verdict on one finding: upheld or dismissed, with a reason when dismissed.
_Avoid_: override

**Precedent**
A remembered dismissal: a finding code that an approver dismissed for a particular employee or supplier, with the approver's reason. Only codes whose dismissal can rest on a standing fact about that employee or supplier, and which the policy does not require to be decided case by case, can become precedents. A precedent informs later decisions and never removes or softens a rule finding.
_Avoid_: exception, whitelist

**Recommendation**
The outcome suggested to the approver (approve, reject or none), derived from the findings and their precedents. It is advice; only an approver decides.
