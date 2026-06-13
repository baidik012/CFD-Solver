# Contributing

## Making Changes

1. **Branch first.** Don't push straight to `main`.
   ```bash
   git checkout -b feature/what-youre-fixing
   ```

2. **Write the code.** Keep it readable. If something isn't obvious, add a comment.

3. **Run the tests before opening a PR:**
   ```bash
   pip install -e ".[dev]"
   pytest tests/
   ```

4. **Commit with a clear message** — what changed and why.
   ```bash
   git commit -m "Add channel flow boundary conditions"
   ```

5. **Push and open a PR.**
   ```bash
   git push origin feature/what-youre-fixing
   ```

## Code Style

- PEP 8. If your linter complains, fix it.
- Use meaningful names. `velocity` beats `v`. `pressure_poisson_rhs` beats `pr`.
- No magic numbers. If `0.5` shows up in three places, give it a name.

## What to Put in a PR

- What changed and why
- Link to related issues if any
- A plot or screenshot if you touched visualization
- How you tested it

## What We Don't Want

- `.png`, `.mp4`, or big `.log` files — those go in `output/`, not the repo
- Commented-out code that "might be useful later"
- PRs that touch 15 unrelated things at once

## Questions?

Open an issue. Small club, someone will get back to you.
