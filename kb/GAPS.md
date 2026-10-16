# Deliberate KB Gaps (FR-A2)

These topics are intentionally **not** covered anywhere in the KB. They exist to test the "I don't know" path: a question on any of them must end in an honest "I don't have that" and an offer to create a ticket (FR-A5, FR-A16), never an invented answer. `tests/test_gaps_doc.py` checks that no KB entry (title, topic or body) matches a gap's keywords.

Each section has an expected ticket category (FR-B7) and the keywords the test uses.

### Monitor flickering
Keywords: monitor, monitors, flicker, flickering, flickers
Expected category: IT
Note: sounds like IT, but the KB has no hardware-fault content. Equipment entries cover allocation, extra peripherals, and lost or stolen laptops only.

### Office parking
Keywords: parking, car park
Expected category: MISC
Note: facilities question; neither IT nor HR owns it.

### Cafeteria menu
Keywords: cafeteria, canteen, food court, menu
Expected category: MISC
Note: facilities question. The KB mentions no food or meal service.

### Desk booking
Keywords: desk booking, book a desk, hot desk, hot-desk, desk reservation, seat booking, book a seat
Expected category: MISC
Note: the hybrid schedule entry (HR-WFH-01) gives office days only, with no seating rules.

### Sabbatical leave
Keywords: sabbatical
Expected category: HR
Note: grey-zone HR question next to the leave entries; no entry offers extended or unpaid leave.

### Employee referral bonus
Keywords: referral, referral bonus, refer a friend
Expected category: HR
Note: grey-zone compensation question; HR-SAL-01 covers pay components and the annual bonus only.

### Learning and certification budget
Keywords: learning budget, learning allowance, certification, training budget, course fee
Expected category: HR
Note: grey-zone HR question; the reimbursement entries cover travel, expenses and home internet only.

### Company offsites and events
Keywords: offsite, off-site, team outing, company event, annual day
Expected category: MISC
Note: facilities and culture question, outside the KB.
