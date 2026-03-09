# Security Policy

## Scope

**cloud-incident-predictor** is a research and educational machine-learning project.
It generates synthetic data locally and runs entirely offline. It does **not**:

- Handle real production data or cloud credentials
- Expose any network services or APIs
- Process authentication, user data, or personally identifiable information

The primary security risk is **dependency vulnerabilities** in third-party
packages (scikit-learn, pandas, numpy, matplotlib).

---

## Supported versions

Only the latest release on the `main` branch receives security updates.

| Version | Supported |
|---------|-----------|
| Latest on `main` | ✅ Yes |
| Older releases   | ❌ No  |

---

## Reporting a vulnerability

**Please do not open a public GitHub issue for security vulnerabilities.**

Report security concerns privately by emailing:

**ignacio.garbayo@rai.usc.es**

Include in your report:

1. A description of the vulnerability and its potential impact
2. Steps to reproduce the issue (code snippet or description)
3. Affected versions or components
4. Any suggested mitigation, if you have one

You can expect a best-effort response within **14 days**. I will acknowledge
receipt as soon as possible and provide an update on the timeline for a fix.

---

## Disclosure policy

This project follows **coordinated disclosure**:

1. Reporter contacts maintainer privately.
2. Maintainer confirms the vulnerability and works on a fix.
3. Fix is released on `main` with a patch version bump and CHANGELOG entry.
4. Reporter is credited in the CHANGELOG (unless they prefer to remain anonymous).
5. A public disclosure is made after the fix is released.

---

## CRA notice

This software is provided as-is for educational and research purposes.
Per **Article 10 of the EU Cyber Resilience Act (CRA)**, open-source software
developed and supplied outside the course of commercial activity is exempt
from the CRA's conformity requirements. This project falls within that exemption.

Nonetheless, dependency security is taken seriously. Users are encouraged to
run `pip install --upgrade -r requirements.txt` regularly and check
[PyPI advisories](https://pypi.org/security/) for the packages used.
