# IT Support

## Contacting the IT help desk, hours and response times

```yaml
id: IT-HELP-01
domain: IT
topic: help desk
title: Contacting the IT help desk, hours and response times
owner: IT Service Desk
last_reviewed: 2026-02-09
```

The SmartDesk AI IT help desk handles account and access problems, password and MFA issues, software requests, VPN, email and security reports. It is the first place to go for anything on that list.

- **Phone:** extension 4100 from any office phone. From outside the office, call the main switchboard on +91-80-5550-0100 and ask for extension 4100.
- **Portal:** https://helpdesk.smartdesk.example. Raising a request here is the best option for anything that is not urgent, because you get a request number and can follow progress.
- **Email:** helpdesk@smartdesk.example. Emails are turned into portal requests automatically.
- **Hours:** Monday to Friday, 08:00 to 20:00 IST. Outside these hours, only urgent security reports are picked up.

Response targets: a security incident or an outage affecting many people is acknowledged within 30 minutes. Every other request is acknowledged within 4 working hours and resolved, or given a status update, within 1 working day.

To get a fast resolution, include your employee ID, a short description, any error text, and the time it happened. For a suspected account compromise, do not wait for help desk hours: call Security on extension 4999 at any time.

## Reset your password yourself

```yaml
id: IT-PWD-01
domain: IT
topic: password
title: Reset your password yourself
owner: IT Service Desk
last_reviewed: 2026-03-02
```

You can reset a forgotten or expired password without contacting the help desk, as long as multi-factor authentication (MFA) is already set up on your phone.

1. Go to https://id.smartdesk.example/reset.
2. Enter your work email address and choose **Forgot password**.
3. Approve the sign-in prompt in the SmartDesk Authenticator app, or type the 6-digit code it shows.
4. Choose a new password that meets the rules in IT-PWD-02 (Password rules and expiry).
5. Wait about 2 minutes, then sign in again.

After a reset, update the saved password everywhere it is stored: Outlook on your phone, the VPN client and any browser that remembered it. A stale saved password is the most common reason an account locks again right after a reset, because five wrong attempts lock it.

**Common problems.** If the account is locked, it unlocks by itself after 15 minutes. If the reset page says your email is not recognised, check you typed the full work address, not a personal one. If you no longer have your phone, you cannot use self-service; call the help desk on extension 4100. IT will ask your manager to confirm your identity, then set a temporary password that you must change at first sign-in.

## Password rules and expiry

```yaml
id: IT-PWD-02
domain: IT
topic: password
title: Password rules and expiry
owner: IT Service Desk
last_reviewed: 2026-04-14
```

SmartDesk AI passwords must:

- be at least **14 characters** long (a passphrase of several unrelated words is fine and easier to remember);
- not repeat any of your last 10 passwords;
- not contain your name, employee ID or work email address;
- not appear on the blocked list of commonly used passwords, which the reset page checks automatically.

Passwords expire every **180 days**. You get reminder emails 14 days and 3 days before expiry, and a sign-in banner in the final week. You can change the password early at any time at https://id.smartdesk.example/reset. After 5 wrong attempts the account locks for 15 minutes.

**What not to do.** Do not share your password with anyone, including IT staff; the help desk will never ask for it, by phone or email. Do not reuse your work password on any other website. A password manager is recommended; the approved one is available in the Software Centre.

If you think your password has been seen by someone else, reset it immediately and tell Security on extension 4999. Do not wait for it to expire.

## Set up multi-factor authentication

```yaml
id: IT-MFA-01
domain: IT
topic: MFA
title: Set up multi-factor authentication
owner: IT Service Desk
last_reviewed: 2026-01-19
```

MFA is mandatory for all email, VPN and portal sign-ins. You set it up once, on your first day or the first time you sign in. Allow about 10 minutes.

1. Install **SmartDesk Authenticator** on your phone from the App Store or Google Play.
2. On your computer, sign in at https://id.smartdesk.example/mfa with your work email and the temporary password from HR.
3. Choose **Add device**. A QR code appears.
4. In the app, choose **Add account** and scan the QR code.
5. Type the 6-digit code from the app to confirm.
6. Save the 10 backup codes shown on screen. Store them somewhere safe that is not the same phone, for example a password manager.

Push approval is the default: you tap **Approve** when signing in. If you have no mobile signal, open the app and type the 6-digit code instead.

**Common problems.** If the QR code will not scan, choose **Enter code manually** and type the key. If the code is rejected, check the phone's date and time are set to automatic, because the codes depend on correct time. Employees who cannot carry a phone at work can ask the help desk for a hardware security key.

## New or lost phone, re-enrol MFA

```yaml
id: IT-MFA-02
domain: IT
topic: MFA
title: New or lost phone, re-enrol MFA
owner: IT Service Desk
last_reviewed: 2026-05-11
```

**Switching to a new phone while the old one still works.**

1. Sign in at https://id.smartdesk.example/mfa on your computer.
2. Choose **Move to a new device** and scan the QR code with SmartDesk Authenticator on the new phone.
3. Confirm with the 6-digit code, then remove the old device from the list.

Do this before you wipe or hand in the old phone. Moving while both phones work takes about five minutes.

**Phone lost, stolen or already reset.**

1. Sign in with one of your 10 backup codes at https://id.smartdesk.example/mfa, then add the new device. Each backup code works once.
2. If you have no backup codes, call the help desk on extension 4100. IT will contact your manager to confirm who you are, then clear your MFA enrolment. You register again from scratch, which takes about 15 minutes.
3. If the phone was stolen, also tell Security on extension 4999 straight away so that all your sessions are signed out.

After re-enrolling, generate a fresh set of backup codes, because the old set should be treated as lost together with the phone.

## Install and configure the VPN client

```yaml
id: IT-VPN-01
domain: IT
topic: VPN
title: Install and configure the VPN client
owner: IT Service Desk
last_reviewed: 2026-02-23
```

SmartDesk AI uses the **SmartDesk Connect** VPN client on company laptops. It is preinstalled; if it is missing, install it from the Software Centre.

1. Open **SmartDesk Connect**.
2. In **Server**, enter `vpn.smartdesk.example`. On managed laptops the profile "SmartDesk-Corp" is already set up and you only need to select it.
3. Sign in with your work email and password.
4. Approve the MFA prompt in SmartDesk Authenticator, or enter the 6-digit code.
5. The status turns green and shows "Connected". Internal sites such as https://people.smartdesk.example and the file shares now work.

To disconnect, choose **Disconnect** in the client; closing the window leaves the tunnel running.

The client is available for Windows, macOS and Ubuntu. Personal devices cannot be given VPN access, and you should not install the client on one even if it installs without error.

If the profile is missing after an update, choose **Add profile**, use the server name above and the profile name "SmartDesk-Corp". If the client asks for a certificate, restart the laptop once so the company certificate loads, then try again.

## Fixing common VPN connection errors

```yaml
id: IT-VPN-02
domain: IT
topic: VPN
title: Fixing common VPN connection errors
owner: IT Service Desk
last_reviewed: 2026-06-24
```

Try these in order. The numbered codes are Windows error codes shown by SmartDesk Connect on Windows laptops; macOS and Ubuntu show the same causes in plain text.

1. **"Authentication failed" (Windows error 691).** Your password may have expired or been changed recently. Reset it at https://id.smartdesk.example/reset and then update the password saved in the client.
2. **The MFA prompt never arrives.** Check that your phone has internet access, then open SmartDesk Authenticator manually and type the 6-digit code instead.
3. **"Server unreachable" (Windows error 809).** First confirm that other websites load on your connection. Some public and hotel networks block VPN traffic; switch to a mobile hotspot.
4. **Connected but internal sites do not load.** Disconnect, quit SmartDesk Connect completely, and connect again. Then clear your browser cache.
5. **The VPN drops after about 10 minutes.** Update the client in the Software Centre. Versions below 5.2 have a known session timeout bug.

If you see a certificate error, restart the laptop once and try again.

If none of these work, raise a request at https://helpdesk.smartdesk.example. Attach a screenshot of the error and note the time it happened, so IT can find the matching entry in the VPN logs.

## When the VPN is required, and split tunnelling

```yaml
id: IT-VPN-03
domain: IT
topic: VPN
title: When the VPN is required, and split tunnelling
owner: IT Security
last_reviewed: 2026-07-08
```

The VPN is **required** whenever you are off the SmartDesk AI office network and need an internal system: the code repositories, the finance and HR portals, internal file shares or customer data. It is **not** needed for email, chat or video calls, which are reachable directly after MFA.

The VPN uses **split tunnelling**. Only traffic for company systems goes through the tunnel; streaming and general browsing use your own connection. This keeps calls smooth and does not reduce security for company data.

Interpretation points:

- If you are on a home or coworking network and are unsure whether a system needs the VPN, assume it does. Connecting when it is not needed does no harm.
- Using the VPN does not make a personal device acceptable. Company data may only be opened on company-managed laptops, whatever the connection.
- Turning the VPN off while customer data is open, for example to join a call, is not permitted. Close the data first.
- Inside the Bengaluru and Pune offices you are on the corporate network already, so the VPN is not needed. It is harmless but will slow your connection.
- On public Wi-Fi such as airports, connect to the VPN before opening anything internal.
- Idle VPN sessions end automatically after 8 hours, so reconnect at the start of each working day.

Questions about a specific system go to IT Security at security@smartdesk.example.

## Request new software

```yaml
id: IT-SW-01
domain: IT
topic: software
title: Request new software
owner: IT Service Desk
last_reviewed: 2026-03-30
```

First check the **Software Centre** on your laptop. Software on the standard list can be installed there at once, with no request.

For anything else:

1. Open https://helpdesk.smartdesk.example and choose **New request**, then **Software**.
2. Give the name and version, what you need it for, and whether a licence fee applies.
3. Your manager approves the request by email. Paid licences also need approval from your cost-centre owner.
4. IT Security reviews new tools within 5 working days, checking the licence terms, where the data is stored and whether the tool collects usage data.
5. Procurement buys the licence if one is needed.
6. IT installs the software remotely, usually within 2 working days of approval.

Urgent requests can be marked **High** in the form; also call extension 4100 so the team sees it. A request without a business reason is returned to you, which is the most common cause of delay.

If your request is rejected, the reply states why and, where possible, suggests an approved alternative. You can appeal to IT Security at security@smartdesk.example with a fuller business case.

## Approved and restricted software list

```yaml
id: IT-SW-02
domain: IT
topic: software
title: Approved and restricted software list
owner: IT Security
last_reviewed: 2026-08-17
```

**Available from the Software Centre with no request:** Microsoft 365, Slack, Zoom, Google Chrome, Firefox, Visual Studio Code, PyCharm Community, Git, Docker Desktop, Postman, 7-Zip, Adobe Acrobat Reader and the approved password manager.

**Available after manager approval:** PyCharm Professional, Tableau, Figma, Snagit and Adobe Creative Cloud. Request these as described in IT-SW-01 (Request new software); they are paid licences.

**Not allowed on company laptops:**

- peer-to-peer file-sharing software;
- cracked or unlicensed software of any kind;
- personal cloud-sync clients that copy company files to a private account;
- remote-control tools other than the one IT provides;
- cryptocurrency miners.

Local administrator rights are not given by default. Developers who need them, for example to run Docker or install drivers, can request time-limited admin access for 30 days through the help desk. Access is removed automatically on day 31; request again if you still need it.

The list is reviewed twice a year. If a tool you use is removed, IT Security emails affected users at least 30 days ahead.

## Using an open-source or personal tool for work

```yaml
id: IT-SW-03
domain: IT
topic: software
title: Using an open-source or personal tool for work
owner: IT Security
last_reviewed: 2026-09-21
```

**Can I use a free or open-source tool that is not on the list?** Often yes, but request it first through https://helpdesk.smartdesk.example. Free does not mean approved: IT Security checks the licence terms and where your data goes.

**Can I use a tool I paid for myself?** Not for company work. Personally licensed software cannot be installed on a company laptop, and the cost cannot be claimed as an expense unless Procurement agreed in advance.

**Open-source libraries in code.** Developers may use libraries under permissive licences (MIT, BSD, Apache 2.0) directly. Libraries under copyleft licences such as GPL or AGPL need a review by Legal before they are included in a customer-facing product. Ask in the #legal-review channel.

**Online services (AI chat tools, file converters, note-taking apps).** Do not paste customer data, source code or internal documents into any online service that is not approved in IT-SW-02 (Approved and restricted software list). Many such services keep what you type and may use it to train their models. If unsure, ask IT Security at security@smartdesk.example before using it.

**If you already did.** If you realise you shared company data with an unapproved service, report it to security@smartdesk.example straight away. Prompt reporting is treated as helpful, not as a breach.

## Connect a visitor to guest Wi-Fi

```yaml
id: IT-WIFI-01
domain: IT
topic: wifi
title: Connect a visitor to guest Wi-Fi
owner: IT Service Desk
last_reviewed: 2026-01-27
```

Guests use the **SmartDesk-Guest** network. It is separate from the corporate network and provides internet access only.

1. The visitor connects to **SmartDesk-Guest** on their phone or laptop and opens any website. The sign-in page appears.
2. The visitor enters their name and the **sponsor's work email**. The sponsor is the SmartDesk AI employee they are visiting.
3. The sponsor receives an approval email and clicks **Approve**.
4. Access is granted for 24 hours.

Employees can extend a guest's access up to 7 days by raising a request at https://helpdesk.smartdesk.example with the guest's name and the dates. For visits longer than 7 days, ask your manager to request a contractor account instead.

**Common problems.** If the sign-in page does not appear, open `http://neverssl.com` or any plain `http` page, because some browsers skip the sign-in page on secure sites. If the approval email does not arrive, check the sponsor's spam folder and confirm the sponsor's address was typed correctly.

Do not give visitors the corporate Wi-Fi details, and do not connect their devices to the corporate network or plug them into office network ports.

## Corporate Wi-Fi networks and which to use

```yaml
id: IT-WIFI-02
domain: IT
topic: wifi
title: Corporate Wi-Fi networks and which to use
owner: IT Service Desk
last_reviewed: 2026-04-27
```

In both the Bengaluru and Pune offices there are two Wi-Fi networks:

- **SmartDesk-Secure** is for company laptops. It signs you in automatically with your work account and a device certificate, and gives access to internal systems without the VPN.
- **SmartDesk-Guest** is for visitors and personal phones.

Personal phones can use **SmartDesk-Guest** after self-registering with the employee's work email at https://helpdesk.smartdesk.example/wifi. Registration lasts 90 days and can be renewed on the same page. Personal devices cannot join SmartDesk-Secure.

**Common problems.**

- If SmartDesk-Secure keeps asking for a password after you changed your account password, forget the network on your laptop and rejoin it.
- If the laptop connects but has no internet, disconnect and reconnect once, then restart the laptop.
- If you see a certificate warning, do not accept it; raise a request, because it may mean an impostor network with a similar name.

Wi-Fi speed problems in a particular office area should be reported through the portal with the floor and the nearest meeting-point landmark, so IT can check the access point.

## Set up work email on your phone

```yaml
id: IT-MAIL-01
domain: IT
topic: email
title: Set up work email on your phone
owner: IT Service Desk
last_reviewed: 2026-05-26
```

Use the **Outlook** mobile app on iOS or Android. It is the only supported mail app for work email.

1. Install Outlook and open it.
2. Choose **Add account** and type your work email address, for example `first.last@smartdesk.example`.
3. Sign in on the SmartDesk AI sign-in page that appears and approve the MFA prompt.
4. Accept the **device security policy**. It requires a lock (PIN, fingerprint or face), and lets IT erase only the work data inside the app if the phone is lost or you leave the company.

Native mail apps, Gmail and other third-party apps are not supported, because they cannot enforce the security policy.

**Common problems.** If sign-in fails with "device not compliant", enable a phone lock and try again. If the account stops syncing after a password change, sign out of the account in Outlook and add it again. If you get a new phone, install Outlook and repeat these steps; the old phone's work data is removed when you remove the account.

Calendar and contacts sync through the same app. Work files should be opened in the app, not saved to the phone's gallery.

## Mailbox size limits and shared mailboxes

```yaml
id: IT-MAIL-02
domain: IT
topic: email
title: Mailbox size limits and shared mailboxes
owner: IT Service Desk
last_reviewed: 2026-07-29
```

Every mailbox is **50 GB**, with a warning at 45 GB. At 50 GB you cannot send new email until space is freed. Attachments are limited to **25 MB** per message; larger files should be shared as OneDrive links, which also saves space.

Deleted items are kept for 30 days before they are removed permanently. Company email is retained for 3 years for audit purposes, even if you delete it from your mailbox.

To free space, empty **Deleted Items**, sort by size to find large attachments, and move old mail to the online archive, which does not count towards your limit.

**Shared mailboxes** (for example support@ or payroll@ addresses) are created by IT. The mailbox owner, or your manager, raises a request at https://helpdesk.smartdesk.example with the mailbox name and the people who need access, saying whether each person needs read, send-as or full access. Shared mailboxes appear in Outlook automatically within 1 hour of approval. A shared mailbox has its own 50 GB limit.

If a shared mailbox owner leaves, the manager must nominate a new owner before the leaver's last day, otherwise the mailbox is closed after 30 days.

## Report a phishing email

```yaml
id: IT-SEC-01
domain: IT
topic: security
title: Report a phishing email
owner: IT Security
last_reviewed: 2026-10-05
```

If an email looks suspicious, treat it as phishing. Typical signs are an unexpected attachment, an urgent request for money or a password, a link whose address does not match the sender, or a sender name that does not match the address.

1. **Do not click links or open attachments.**
2. In Outlook, choose **Report** and then **Phishing**. This sends it to IT Security and removes it from your inbox.
3. If you cannot use the button, forward the email as an attachment (not a normal forward) to phishing@smartdesk.example.
4. If you already clicked a link or typed your password, call Security on extension 4999 immediately, then reset your password at https://id.smartdesk.example/reset.

Report suspicious text messages and messages on Slack the same way, with a screenshot, to phishing@smartdesk.example.

IT Security replies within 1 working day and sends a warning to all staff if the message is part of a wider campaign. If the email asked for a payment, also tell your manager and Finance. Reporting a phishing email, even a harmless one, is never penalised. Staying quiet about a click is what causes harm.
