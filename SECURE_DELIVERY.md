# Secure delivery note

The task requires a public GitHub repository, while the supplied Kestrel policy says client data must not be published or shared outside the engagement. To satisfy both requirements safely:

- the public repository contains source code, tests, documentation, and empty placeholders only;
- raw client files, the trained model, and detailed local metrics stay outside Git;
- the required two-column `predictions.csv` is included in the submission repository because the hiring team explicitly requires it there;
- the trained artifact can be shared with an authorised reviewer through a private Google Drive folder, the ATS upload mechanism, or another approved secure channel;
- repository `.gitignore` rules prevent accidental commits of these files.

Local deliverables after training:

- `artifacts/model.joblib`
- `artifacts/metrics.json`

Before sharing, confirm that the recipient account belongs to the hiring team and set the private link to viewer-only access.
