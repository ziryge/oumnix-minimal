# Contributing to Oumnix AI

Thank you for your interest in contributing! We welcome pull requests and issues for improvements, bug fixes, and documentation. Please read this document to get started.

## Quick Start
1. Fork the repository and create a new branch:
   - `git checkout -b feat/your-feature-name`
2. Install dependencies:
   - `pip install -r requirements.txt`
3. Run quick tests to ensure the environment works:
   - `python test_train.py`
4. Make your changes and add tests when applicable.
5. Run training quickly to smoke‑test:
   - Streaming (recommended): `python train_streaming.py --epochs 1 --batch_size 2 --use_amp --out_dir tmp_checkpoints`
   - Simple: `python train_simple.py --epochs 1 --batch_size 4 --use_amp --out_dir tmp_checkpoints_simple`
6. Commit and push:
   - `git commit -m "feat: add X"`
   - `git push origin feat/your-feature-name`
7. Open a Pull Request and describe what changed and why.

## Code Style
- Prefer descriptive names and type hints.
- Keep functions small and focused.
- Add docstrings for public functions and classes.
- Avoid hard‑coding values; use parameters or config.

## Testing
- Use the provided `test_train.py` for a basic environment check.
- For new modules, add small tests or smoke scripts when feasible.
- Avoid committing large dataset files.

## Commits
- Conventional commits appreciated (feat, fix, docs, refactor, test, chore).
- Keep PRs small and focused. Link issues when possible.

## Security
- Do not include secrets in code or commits.
- If you find a vulnerability, please open a private issue or contact the maintainers.

## License and DCO
- By contributing, you agree your contributions are licensed under the repository’s license (BSL 1.1 until the Change Date).
- Include a Signed‑off‑by line if your organization requires DCO.

## Communication
- Please follow the [Code of Conduct](CODE_OF_CONDUCT.md).
- Be respectful and constructive in issues and PRs.

## Areas of Interest
- Memory system (Infinity Window, Hot/Warm‑KV, PQ + Low‑Rank)
- Metacognition (strategy selection, causal/analogy engines)
- Neurochemistry (advanced modulation)
- FP8/AMP optimizations and training stability
- Streaming training pipeline and dataset adapters
- Tokenizer and data tooling
- UI/UX for advanced CLI and web interface

Thanks again for contributing!
