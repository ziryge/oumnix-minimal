# Security Policy

Thank you for helping keep this project and its users safe.

## Supported Versions
We generally provide security fixes and responses for:
- The default branch (main) and the most recent tagged release.
- Older versions may receive fixes on a best‑effort basis only.

Note: The project is licensed under BSL 1.1 until the Change Date. Security fixes may be distributed as patches or new releases.

## Reporting a Vulnerability
Please use responsible disclosure and avoid filing public issues for security reports.

Preferred channels (choose one):
1) GitHub Security Advisory (private) — if enabled for this repository.
2) Email the maintainers — add your preferred security contact here (e.g., security@yourdomain.example). If no address is available, open a minimal issue requesting a private contact channel.

When reporting, include:
- A clear description of the issue and potential impact.
- Steps to reproduce or a minimal proof‑of‑concept.
- Affected version/commit and environment details (OS, Python, CUDA, GPU model).
- Suggested remediation, if available.

Please do not:
- Publicly disclose the issue or PoC prior to coordination.
- Access, modify, or exfiltrate data you do not own.
- Run tests against third‑party systems without authorization.

## Coordinated Disclosure and Timelines
- Acknowledgment: within 3 business days.
- Triage and initial assessment: within 7 business days.
- Fix or mitigation target: within 90 days of confirmation (may be adjusted based on severity/complexity).
- Public advisory: coordinated with the reporter once a fix/mitigation is available.

We’ll keep you informed of progress and timelines. In cases of active exploitation or critical severity, we may accelerate timelines.

## Scope
In scope:
- Code in this repository and official release artifacts.
- Issues leading to RCE, privilege escalation, authentication/authorization bypass, injection, cryptographic weaknesses, sensitive data exposure, memory corruption, or sandbox escapes.

Out of scope (non‑exhaustive):
- Denial‑of‑service caused by extremely large inputs or resource exhaustion without a clear bypass of built‑in limits.
- Social engineering, phishing, and issues that require physical access.
- Vulnerabilities in third‑party dependencies (please report upstream); we will track and update as appropriate.
- Best‑practice suggestions without a demonstrable security impact.

## Safe Harbor
We support good‑faith security research:
- If you comply with this policy, we will not pursue legal action against you.
- Make a reasonable effort to avoid privacy violations, data destruction, and service degradation.
- Only test against your own environments/data.
- Share PoC and details privately through the channels above.

## Credits and Recognition
With your permission, we can acknowledge reporters in release notes or advisories. We currently do not offer a formal bug bounty program.

## Contact
- Security contact: qorvuscompany@gmail.com
- Project owner: qrv0
