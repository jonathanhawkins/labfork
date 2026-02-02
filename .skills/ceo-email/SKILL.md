---
name: ceo-email
description: Manage the LabFork CEO email inbox - send, check, and respond to emails
metadata:
  tags: email, ceo, agentmail, communication, outreach
---

# CEO Email Skill

**Manage the LabFork AI CEO email inbox.**

Email: `labfork-ceo@agentmail.to`
Display Name: Claude | LabFork CEO

## Quick Commands

```bash
# Check inbox for new messages
~/bin/ceo-email check

# List recent messages
~/bin/ceo-email list --limit 10

# Read a specific message
~/bin/ceo-email read <message_id>

# Send an email
~/bin/ceo-email send "recipient@example.com" "Subject" "Body text"

# Show inbox info
~/bin/ceo-email info
```

## Use Cases

### Outreach
```bash
# Contact a potential partner
~/bin/ceo-email send "partner@company.com" "LabFork Partnership Inquiry" \
  "Hi, I'm the AI CEO of LabFork..."
```

### Grant Applications
```bash
# Send grant inquiry
~/bin/ceo-email send "grants@foundation.org" "AI4PG Grant Application - LabFork" \
  "Dear Selection Committee,..."
```

### Responding to Inquiries
```bash
# Check for new messages
~/bin/ceo-email check

# Read and respond
~/bin/ceo-email read abc123
~/bin/ceo-email send "inquirer@email.com" "Re: Your Question" "Thank you for reaching out..."
```

## Email Signature

Always include in formal emails:

```
Best regards,
Claude
AI CEO, LabFork
labfork-ceo@agentmail.to
https://labfork.com

---
LabFork: Democratizing AI research through distributed compute.
```

## Forwarding

All sent emails are also forwarded to: `ceo@labfork.com` (human oversight)

## Environment

Requires `AGENTMAIL_API_KEY` environment variable.

## API Reference

- Base URL: `https://api.agentmail.to/v0`
- Inbox ID: `labfork-ceo@agentmail.to`
- Auth: Bearer token

## Best Practices

1. **Be transparent** - Always identify as an AI
2. **Be professional** - Represent LabFork well
3. **Be responsive** - Check inbox regularly
4. **CC the team** - Forward important threads to ceo@labfork.com
