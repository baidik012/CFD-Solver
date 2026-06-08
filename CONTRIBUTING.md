# Contributing

Here's how we work together on this project.

## Making Changes

1. **Branch first.** Never push straight to `main`.
   ```bash
   git checkout -b feature/what-youre-fixing
   ```

2. **Write the code.** Keep it readable. If it's not obvious what something does, add a comment.

3. **Test it.** Install dev dependencies and run tests before you open a PR:
   ```bash
   pip install -e ".[dev]"
   pytest tests/
   ```

4. **Commit.** Write a short message explaining *what* changed and *why*.
   ```bash
   git commit -m "Add channel flow boundary conditions"
   ```

5. **Push and open a PR.**
   ```bash
   git push origin feature/what-youre-fixing
   ```

## Code Style

- Follow PEP 8. If your linter complains, fix it.
- Use meaningful names. `velocity` is better than `v`. `pressure_poisson_rhs` is better than `pr`.
- No magic numbers. If you're using `0.5` in three places, make it a constant with a name.

## What to Include in a PR

- A clear description of what changed
- Link to any related issues
- A screenshot or plot if you added visualization
- A note on how you tested it

## What We Don't Want

- `.png`, `.mp4`, or large `.log` files — those go in `output/`, not the repo
- Commented-out code that "might be useful later"
- Pull requests that touch 15 unrelated things at once

## Questions?

Open an issue. We're a small club — someone will get back to you.