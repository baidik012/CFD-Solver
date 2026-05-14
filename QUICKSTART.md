# Complete Beginner's Guide

From having nothing installed to running your first simulation.

---

## Part 1: Install the Tools

### 1.1 Install Python

**Windows:**
1. Go to [python.org/downloads](https://python.org/downloads)
2. Download the latest Python 3.x (green button)
3. Run the installer
4. **IMPORTANT:** Check "Add Python to PATH" before clicking Install
5. Click "Install Now"

**Mac:**
```bash
# If you have Homebrew (recommended)
brew install python3

# Or download from python.org/downloads
```

**Linux:**
```bash
sudo apt update
sudo apt install python3 python3-pip
```

**To verify Python installed:**
```bash
python --version
# Should show: Python 3.10.x or newer
```

---

### 1.2 Install Git

**Windows:**
1. Go to [git-scm.com/download/win](https://git-scm.com/download/win)
2. Download and run the installer
3. Use all default options (Next, Next, Next...)
4. Select "Git Bash Here" context menu option if asked

**Mac:**
```bash
# Usually already installed, verify:
git --version

# If not, install via:
xcode-select --install
```

**Linux:**
```bash
sudo apt install git
```

---

## Part 2: Get the Code

### 2.1 Open Your Terminal

**Windows:** Press `Win + R`, type `cmd`, press Enter. Then type `git bash` and press Enter.

**Mac:** Press `Cmd + Space`, type `Terminal`, press Enter.

**Linux:** Press `Ctrl + Alt + T`.

---

### 2.2 Clone the Repo

If the repo is **public** (recommended):
```bash
git clone https://github.com/baidik012/CFD-Solver.git
cd CFD-Solver
```

If the repo is **private** (you were added as collaborator):
```bash
git clone https://github.com/baidik012/CFD-Solver.git
cd CFD-Solver
# GitHub will ask for your username and password
# Use your GitHub username and a Personal Access Token as password
```

**To create a Personal Access Token:**
1. Go to [github.com/settings/tokens](https://github.com/settings/tokens)
2. Click "Generate new token (classic)"
3. Name it something like "CFD Solver"
4. Check **repo** scope
5. Click Generate
6. Copy the token — you won't see it again

When GitHub asks for password, use:
- Username: your GitHub username
- Password: the token (it looks like `ghp_xxxxxxxxxxxxxxxxxxxx`)

---

## Part 3: Set Up the Environment

### 3.1 Create a Virtual Environment

A virtual environment keeps this project's dependencies separate from other Python projects. Don't skip this.

```bash
# From the CFD-Solver directory
python -m venv venv
```

This creates a folder called `venv` that contains its own Python installation.

---

### 3.2 Activate the Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

You'll know it's active when your prompt shows `(venv)` at the start, like:
```
(venv) baidik@MSI:~/CFD-Solver$
```

---

### 3.3 Install Dependencies

```bash
pip install -r requirements.txt
```

This installs numpy, matplotlib, scipy, etc. — everything the solver needs.

---

## Part 4: Run the Solver

### 4.1 Run the Example

```bash
python examples/staggered_cavity.py
```

You'll see output like:
```
Lid-driven cavity: 128x128 grid, 1000 steps

Step    0: |∇·u|∞ = 5.12e-03, CFL = 0.012
Step  100: |∇·u|∞ = 4.21e-04, CFL = 0.089
...
Step 1000: |∇·u|∞ = 1.33e-06, CFL = 0.152
Solve time: 0.87s
Saved output/staggered_result.png
```

It takes a few seconds. Let it finish.

---

### 4.2 View the Results

Open the file `output/staggered_result.png`:
- **Windows:** Double-click it in File Explorer
- **Mac:** Double-click in Finder
- **Linux:** Double-click in Files app

You should see two plots:
- **Left:** Pressure field (red = high, blue = low)
- **Right:** Velocity magnitude with arrows showing flow direction

---

## Part 5: Experiment

### 5.1 Change the Parameters

Open `examples/staggered_cavity.py` in a text editor (VS Code, Notepad++, etc.) and change values:

```python
Nx, Ny = 128, 128  # Try 64 or 256
nu = 0.01          # Try 0.001 (thinner) or 0.1 (thicker)
dt = 0.001         # Try 0.0005 (more stable) or 0.002 (faster but risky)
steps = 1000       # Try 500 (quicker) or 2000 (more settled)
```

Save the file and run again.

---

### 5.2 What to Try

**Faster lid (more turbulent):**
Change `u_bc={"top": 1.0}` to `"top": 2.0`

**Lower viscosity (more chaotic):**
Change `nu=0.01` to `nu=0.001`

**Higher resolution (more detail, slower):**
Change `Nx, Ny = 128, 128` to `Nx, Ny = 256, 256`

---

## Troubleshooting

### "python not found" or "python is not recognized"

Python isn't in your PATH. Reinstall Python and check "Add Python to PATH".

---

### "pip not found"

Run this:
```bash
python -m pip install -r requirements.txt
```

---

### "No module named 'cfd_solver'"

Make sure you're in the `CFD-Solver` directory and the virtual environment is activated. Your prompt should show `(venv)`.

Run `pwd` to check your current directory. It should end with `CFD-Solver`.

---

### Import error or missing package

Deactivate and re-create the environment:
```bash
deactivate
rm -rf venv          # Windows: rmdir /s /q venv
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

### "Permission denied" or "cannot push"

You're trying to push changes but don't have write access. That's fine — you can still run the solver and experiment locally. If you want to contribute, ask the repo owner to add you as a collaborator.

---

### Everything else

1. Make sure the virtual environment is active (prompt shows `(venv)`)
2. Make sure you're in the `CFD-Solver` folder
3. Try the full re-setup above

Still stuck? Open an issue on GitHub with the error message.

---

## What's Next?

Once you've run a few experiments:

1. Read [DEVELOPMENT.md](DEVELOPMENT.md) to understand how the solver works
2. Look at the code in `src/cfd_solver/solver/` — it's meant to be readable
3. Try adding a new parameter or boundary condition
4. Run the divergence check — it should be near zero if things are working

---

**Quick reference for next time:**

```bash
# Activate environment
source venv/bin/activate        # Mac/Linux
# venv\Scripts\activate         # Windows

# Run solver
python examples/staggered_cavity.py

# View results
# Open output/staggered_result.png
```