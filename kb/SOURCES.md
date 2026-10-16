# KB Sources (FR-A10)

SmartDesk AI is a fictional company. Its KB is original writing, with a small number of entries adapted from one public dataset. This file records every public source, its licence, and what was changed.

## Public source used

| Field | Value |
|---|---|
| Dataset | `strova-ai/hr-policies-qa-dataset` on Hugging Face (`dataset.jsonl`, 644 single-turn Q&A rows) |
| Retrieved | 2026-10-15, with the `huggingface_hub` client. Row numbers below are zero-based line numbers of `dataset.jsonl` at that date. |
| Nature | Synthetic HR-policy Q&A for one fictional employer, "Kreeda Labs". It covers conduct, gifts, whistleblowing, compensatory off and remote work. It has no IT, onboarding, payroll or tax content. |

### Licence statements (both recorded)

1. **Dataset metadata:** `license: mit`. MIT is permissive and allows adaptation.
2. **Dataset card prose:** "Open for research and educational use."

The two statements are compatible with this use, which is a course project. The MIT notice is kept here as attribution. Nothing from the dataset is copied verbatim.

## Adapted items

Each row below is a KB entry whose content was adapted from the dataset. "Adapted" means rewritten in new words for SmartDesk AI, with company name, contacts, numbers and processes replaced (FAQ Q6). No sentence is reused.

| Entry | Source | Licence | Dataset rows | Adaptation notes |
|---|---|---|---|---|
| HR-LEAVE-08 | strova-ai/hr-policies-qa-dataset | MIT | 21–29, 34–54, 47, 48, 136 | Compensatory off: approval in the leave system before use, lapse period, limit on combining leave types, discretion for extended hours. Lapse set to 45 days and combining limit to 3 leave types; portal flow rewritten for SmartDesk. |
| HR-COND-01 | strova-ai/hr-policies-qa-dataset | MIT | 168–174, 176–179 | Annual review of the code of conduct and employee acknowledgement of updates, plus feedback routes. Acknowledgement window set to 14 days in the people portal. |
| HR-COND-02 | strova-ai/hr-policies-qa-dataset | MIT | 7–13, 196–209, 252–258 | Gifts above a value threshold must be reported; unavoidable gifts go to HR or Compliance; exceptions need written approval; anti-bribery. The dataset leaves the threshold unspecified; SmartDesk sets it at ₹3,000. Conflict-of-interest content is original (the dataset has none). |
| HR-HARASS-01 | strova-ai/hr-policies-qa-dataset | MIT | 234, 239, 316 | Prevention through policy, training and encouragement to report; no retaliation. Mandatory training, committee and examples are original. |
| HR-HARASS-02 | strova-ai/hr-policies-qa-dataset | MIT | 238, 241, 275, 276, 281, 283, 286 | Good-faith protection, a preliminary review followed by a formal investigation, and consequences. Channels, timings and support are original SmartDesk details. |
| HR-WFH-01 | strova-ai/hr-policies-qa-dataset | MIT | 371–380 | Remote-work expectations: reachability in working hours, company devices only for company data, absence reporting, escalating consequences, breaks. Core days, hours and the 2-day limit are original. |

Rows deliberately **not** used: those that name real-looking approvers or job titles (for example 126 and 259), those containing contact email addresses (183 to 188), and the generic "how often should any policy be reviewed" rows, which carry no company-specific facts.

## Everything else

All other KB entries and the Q&A pairs in `kb/qa.json` are original writing for this project, drafted with the help of Claude Code and reviewed and approved by the project owner. They were not generated through the OpenAI API and were not copied from any public source. All people, portals, extensions, phone numbers, amounts and policies are fictional.
