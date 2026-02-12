# Development Setup

To set up the development environment for the CFD-Solver project, please follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/baidik012/CFD-Solver.git
   cd CFD-Solver
   ```

2. **Install dependencies**:
   Make sure you have Python and pip installed. Then run:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

# Running Tests

To run tests, you can use the following command:
```bash
pytest
```
Make sure to have all the required test dependencies installed as specified in `requirements.txt`.

# Code Quality

We follow code quality standards by using linters such as flake8. To check for code quality, run:
```bash
flake8 .
```

# Debugging

For debugging, consider using built-in tools such as `pdb` in Python. You can set breakpoints in your code using:
```python
import pdb; pdb.set_trace()
```

# Best Practices for Contributing Developers

- **Write clear commit messages**: Use imperative mood and keep them short.
- **Follow coding standards**: Stick to PEP 8 for Python code.
- **Document your code**: Write docstrings and comments where necessary.
- **Test your changes**: Ensure all tests pass before submitting a pull request.
- **Communicate with the team**: Use issues and PR comments to discuss changes or ask for help.

Happy coding!