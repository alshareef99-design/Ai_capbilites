---
id: cloud_memo_en
title: Board memo — cloud migration
department: strategy
doc_type: memo
language: en
---

# Decision memo: production hosting for 2026–2028

**Question:** Should we migrate production to Azure, move to AWS, or remain on-premises?

## Options

### Azure
Estimated three-year run cost **1.2 million SAR**. Uses the existing Microsoft estate (identity, email, endpoint). Faster compliance mapping to NCA and ISO 27001 controls already documented. Residual risk: Microsoft lock-in.

### AWS
Estimated three-year run cost **1.4 million SAR**. Stronger marketplace for data services. Higher integration cost with current Microsoft identity. Residual risk: dual-cloud operating model.

### On-premises
Estimated three-year run cost **1.8 million SAR** including hardware refresh in 2027. Full data residency. Capacity lead time of **five months**. Residual risk: slower recovery and scarce on-call skills.

## Recommendation

The steering committee **recommends Azure**. Rationale: lowest total cost, reuse of existing Microsoft controls, and an acceptable lock-in risk given a documented 4-month exit pattern in the Northwind contract lessons learned.

Open questions: confirm 2026 budget reservation; complete SOC 2 review of the landing-zone partner.
