# Upload to a repository

Extract the ZIP and upload the **contents of `faba-framework/`**, rather than the
ZIP itself. No remote repository is configured and no upload is performed by
these scripts. Review the source and complete your intended anonymization first.
Do not remove attribution/license notices for third-party code or assets.

```bash
cd faba-framework
git init
git add .
git status --short
git commit -m "Initial source release"
git branch -M main
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

No dataset files, model weights, poisoned samples, feature PKLs, result logs,
virtual environments, SSH credentials or Git history are included. `.gitignore`
excludes common runtime products. The manifest records the exact source release;
changes made during subsequent anonymization will require regenerating hashes
if you want that manifest to describe the revised release.
