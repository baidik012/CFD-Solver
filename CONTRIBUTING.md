# Contributing

Small club, small repo. Here's how we work.

## Making Changes

Branch first, always.
```bash
git checkout -b feature/what-youre-fixing
```

Write your code, test it, commit it with a message that actually describes what you did, then open a PR.
```bash
pip install -e ".[dev]"
pytest tests/
git commit -m "Add sinusoidal lid profile to reduce corner singularities"
git push origin feature/what-youre-fixing
```

## Code Style

PEP 8. Meaningful names. No magic numbers. If it needs explaining, add a comment.

## PRs

Say what changed and why. If you touched the visualization, include a plot. If you're not sure if something's ready, open a draft PR and ask.

## Don't

- Commit `.png`, `.mp4`, or log files — those go in `output/`
- Leave commented-out code in
- Touch fifteen things in one PR

## Questions?

Open an issue or ask in the group chat.
