# Clone the research project with existing results

Publish the separate research directory containing this document. The original
unrelated repository is not needed. The root `.gitignore` keeps the source,
documentation, configurations, small result JSON files, and manifests, while
excluding downloaded datasets, environments, checkpoints, caches, bulk arrays,
and uncompressed prediction dumps.

The portable baseline bundle is `reference_results/tvr_demul/`. Commit this
directory: it contains existing per-query predictions in gzip format, matching
metrics for both saved NMS protocols, and provenance. No baseline model needs
to run after cloning. Historical result JSON under `runs/` and reports under
`docs/` are also retained; no blanket `runs/` or `*.json` ignore rule is used.

From this research directory, initialize Git if it is not already initialized:

```bash
git init -b main
git add .gitignore README.md docs scripts src tests configs examples manifests reference_results
git add runs
git status --short
git diff --cached --stat
```

Review the staged files before committing. GitHub blocks individual files above
100 MiB in ordinary Git, and warns above 50 MiB. The compact baseline files are
below those thresholds. See [GitHub's size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

Create an empty GitHub repository, replace the placeholders below with its SSH
URL, then publish:

```bash
git commit -m "Add VCMR research code, documentation, and baseline reference results"
git remote add origin git@github.com:YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

If Git or `origin` already exists, inspect `git status` and `git remote -v`
instead of reinitializing or blindly replacing a remote.

On the new device:

```bash
git clone git@github.com:YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
python3 - <<'PY'
import hashlib
import json
from pathlib import Path

r = Path('reference_results/tvr_demul')
manifest = json.loads((r / 'manifest.json').read_text())
for entry in manifest['files']:
    p = r / entry['path']
    h = hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    assert p.stat().st_size == entry['bytes'], p
    assert h.hexdigest() == entry['sha256'], p
print('Baseline reference bundle verified.')
PY
```

The clone includes enough evidence to reuse baseline scores and saved final
predictions. It does not include TVR data, environments, weights, or third-party
checkouts. Those are unnecessary for reading the results; the new method will
need its own data and environment setup using the provided instructions.

Give the next agent the research prompt with the updated section 10 that says
to reuse existing baseline results. Point it to `reference_results/tvr_demul/`.
The prompt on disk may still contain the earlier baseline-reproduction wording
unless that replacement has been applied. Explicitly tell the agent that the
baseline reuse policy takes precedence. Historical READMEs, design notes, and
launch scripts describe the completed reranking study; do not launch that study
again or treat its fixed-candidate design as the new research scope. A fresh
independent study uses its own directory and attempt ledger.

`.gitignore` affects untracked files, not files already in Git history. For an
unwanted file staged or tracked before these rules were added, inspect it and
use `git rm --cached -- path/to/file` to stop tracking it while retaining the
local copy. A large file already in an earlier commit also needs removal from
history before pushing. See [Git's ignore documentation](https://git-scm.com/docs/gitignore).
