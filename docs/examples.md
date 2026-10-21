# Example conversations

Recorded 2026-10-21 with the real service (the same `ChatService` the UI uses): LLM live, tickets live. One fake email, `demo.employee@smartdesk.example`. Spend by the ledger: $0.0072 over 11 model calls. The replies are exactly as returned; the lines in italics are the agent steps from the debug panel.

Ticket created in conversation B: **#7** (https://github.com/pbhat123/SmartDesk-Tickets/issues/7).

## A. A KB question answered with citations

**Employee:**

```text
I forgot my laptop password. How can I reset it myself?
```

**SmartDesk AI:**

```text
You can reset a forgotten password yourself if multi-factor authentication is already set up on your phone. Go to https://id.smartdesk.example/reset, enter your work email address, choose **Forgot password**, then approve the sign-in prompt in the SmartDesk Authenticator app or type the 6-digit code it shows. Choose a new password that meets the password rules, then wait about 2 minutes and sign in again.

After that, update the saved password anywhere it is stored, including Outlook on your phone, the VPN client and any browser that remembered it. If your phone is no longer available, you cannot use self-service and need to call the help desk on extension 4100.

Sources: IT-PWD-01 (Reset your password yourself)
```

*Steps: start → route → search_kb → answer → reply*

## B. A gap question: ticket offer, summary, approval, a real GitHub issue

**Employee:**

```text
Several keys on my laptop keyboard stopped responding. Can IT look at it?
```

**SmartDesk AI:**

```text
To open a ticket I need your work email address. What is it?
```

*Steps: start → route → t_collect*

**Employee:**

```text
Yes please, my email is demo.employee@smartdesk.example
```

**SmartDesk AI:**

```text
Here is the ticket I'm ready to create:

Title: Laptop keyboard keys not responding
Description: Several keys on my laptop keyboard stopped responding. Can IT look at it?
Category: IT
Priority: High
Email: demo.employee@smartdesk.example

Shall I go ahead and create this?
```

*Steps: start → offer_reply:agreed → t_collect → t_draft → t_dup_check → t_present*

**Employee:**

```text
Make the priority high
```

**SmartDesk AI:**

```text
Sorry, I didn't catch that. Please reply yes to create the ticket, no to cancel, or tell me what to change.

Here is the ticket I'm ready to create:

Title: Laptop keyboard keys not responding
Description: Several keys on my laptop keyboard stopped responding. Can IT look at it?
Category: IT
Priority: High
Email: demo.employee@smartdesk.example

Shall I go ahead and create this?
```

*Steps: start → t_confirm → t_reask*

**Employee:**

```text
Yes, go ahead
```

**SmartDesk AI:**

```text
Your ticket has been created. Ticket #7 (Open): https://github.com/pbhat123/SmartDesk-Tickets/issues/7

You can ask me for updates on it anytime.
```

*Steps: start → t_confirm → t_create*

## C. A status check for the same email

**Employee:**

```text
Any update on my keyboard ticket?
```

**SmartDesk AI:**

```text
Ticket #7: Laptop keyboard keys not responding
Status: Open
https://github.com/pbhat123/SmartDesk-Tickets/issues/7

No updates from the support team yet.
```

*Steps: start → route → s_lookup*

## D. A new session with the same email recalls the ticket (long-term memory)

**Employee:**

```text
Any update on my ticket?
```

**SmartDesk AI:**

```text
To check on your tickets I need your work email address. What is it?
```

*Steps: start → route → s_lookup*

**Employee:**

```text
demo.employee@smartdesk.example
```

**SmartDesk AI:**

```text
Ticket #7: Laptop keyboard keys not responding
Status: Open
https://github.com/pbhat123/SmartDesk-Tickets/issues/7

No updates from the support team yet.
```

*Steps: start → s_email:ok → s_lookup*
