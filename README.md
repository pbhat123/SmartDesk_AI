# SmartDesk AI

First-line IT/HR help desk agent: answers from a knowledge base, creates tickets after confirmation, and checks ticket status.

The full README (setup, run commands, example conversations, models, hardware) is written in Phase 9, T9.2. This file holds the notes already decided.

## Tickets

MCP: not used; direct GitHub REST API (allowed per FAQ Q12)

Mock tickets live in memory; a restart clears them and numbering restarts at #5. (Not shown in the app; the chat sidebar has only the mode pills.)

## Known limitations

- Moderation `illicit` calibration: the OpenAI Moderation API flags some harmless IT questions (a VPN server name, a scraping script) as `illicit` at scores around 0.28 to 0.34. The app lets a message through when `illicit` is the only flagged category and its score is below `guardrails.illicit_block_score` (0.50); clearly illicit phrases scored 0.75 to 0.95. The threshold was set from a small sample (4 illicit, 6 benign texts), so a borderline request in the 0.30 to 0.50 range now passes the input guard (the injection check, the out-of-scope check and the output guard still run). See PLAN 14.10.
