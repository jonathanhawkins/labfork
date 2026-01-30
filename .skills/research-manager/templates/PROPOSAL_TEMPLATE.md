# Research Proposal: {{STORY_TITLE}}

**Story ID:** {{STORY_ID}}
**Generated:** {{DATE}}
**Status:** Pending Review

---

## Executive Summary

{{EXECUTIVE_SUMMARY}}

<!--
Write 2-3 paragraphs summarizing:
1. What was researched and why
2. Key findings and insights
3. Recommended path forward
-->

---

## Research Findings

{{#each RESEARCH_TASKS}}
### {{TASK_ID}}: {{TASK_SUBJECT}}

**Status:** {{STATUS}}
**Agent:** {{AGENT_TYPE}}
**Output:** {{OUTPUT_FILE}}

**Key Findings:**
{{FINDINGS}}

**Relevant Sources:**
{{SOURCES}}

---
{{/each}}

## Recommended Approach

### Primary Recommendation

{{PRIMARY_RECOMMENDATION}}

<!--
Describe the recommended implementation approach:
- What technology/technique to use
- Why this approach over alternatives
- Key dependencies or prerequisites
-->

### Implementation Roadmap

| Phase | Description | Dependencies | Complexity |
|-------|-------------|--------------|------------|
{{#each IMPLEMENTATION_PHASES}}
| {{PHASE_NUM}} | {{DESCRIPTION}} | {{DEPENDENCIES}} | {{COMPLEXITY}} |
{{/each}}

### Success Criteria

- [ ] {{SUCCESS_CRITERION_1}}
- [ ] {{SUCCESS_CRITERION_2}}
- [ ] {{SUCCESS_CRITERION_3}}

---

## Alternatives Considered

{{#each ALTERNATIVES}}
### Alternative {{INDEX}}: {{NAME}}

**Approach:** {{DESCRIPTION}}

**Pros:**
{{#each PROS}}
- {{.}}
{{/each}}

**Cons:**
{{#each CONS}}
- {{.}}
{{/each}}

**Why Not Chosen:** {{REJECTION_REASON}}

---
{{/each}}

## Open Questions

The following questions need user input before proceeding:

{{#each OPEN_QUESTIONS}}
{{INDEX}}. **{{QUESTION}}**
   - Context: {{CONTEXT}}
   - Options: {{OPTIONS}}
   - Default: {{DEFAULT}}

{{/each}}

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
{{#each RISKS}}
| {{RISK}} | {{LIKELIHOOD}} | {{IMPACT}} | {{MITIGATION}} |
{{/each}}

---

## Estimated Effort

**Overall Complexity:** {{COMPLEXITY_TSHIRT}} (XS / S / M / L / XL)

| Component | Estimate | Notes |
|-----------|----------|-------|
{{#each EFFORT_BREAKDOWN}}
| {{COMPONENT}} | {{ESTIMATE}} | {{NOTES}} |
{{/each}}

---

## Appendix

### Research Artifacts

{{#each ARTIFACTS}}
- [{{NAME}}]({{PATH}}): {{DESCRIPTION}}
{{/each}}

### Commands to Verify

```bash
{{VERIFICATION_COMMANDS}}
```

---

## Approval

- [ ] **Approved** - Proceed with implementation
- [ ] **Needs Revision** - Address feedback below
- [ ] **Rejected** - Do not proceed

**Reviewer Notes:**
{{REVIEWER_NOTES}}

**Approved By:** {{APPROVER}}
**Date:** {{APPROVAL_DATE}}
