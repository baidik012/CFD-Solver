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

```bash
git clone https://github.com/baidik012/CFD-Solver.git
cd CFD-Solver
```

If the repo is **private**, GitHub will ask for credentials:
- Username: your GitHub username
- Password: a [Personal Access Token](https://github.com/settings/tokens) (not your account password)

---

## Part 3: Set Up the Environment

**Windows** — double-click `setup.bat` in the repo folder.

**Mac/Linux:**
```bash
chmod +x setup.sh
./setup.sh
```

This creates a virtual environment and installs everything automatically. Takes about a minute.

---

## Part 4: Run the Solver

**Windows** — double-click `run.bat`.

**Mac/Linux:**
```bash
chmod +x run.sh
./run.sh
```

You'll see prompts like:
```
Grid cells in x [32]:
Grid cells in y [32]:
Viscosity (nu) [0.01]:
Time step (dt) [0.001]:
Number of steps [200]:
Lid speed [1.0]:
```

Press **Enter** to accept each default, or type a value to change it. The solver runs and opens the result image automatically.

---

## Part 5: Experiment

Just run `run.bat` (or `./run.sh`) again and choose different parameters.

### What to try

| Change | Effect |
|--------|--------|
| Grid: `64` or `128` | More detail, slower |
| Viscosity: `0.001` | Thinner fluid, more chaotic |
| Viscosity: `0.1` | Thicker fluid, smoother |
| Steps: `500` | Longer simulation |
| Lid speed: `2.0` | Faster driving, more turbulent |

**Stability rule:** If the solver blows up (values go to infinity), reduce the time step or increase viscosity. The diffusion scheme is stable for any dt, but the advection can become unstable at fine grids with large time steps.

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

### Solver blows up (values go to infinity)

The diffusion scheme is stable for any dt, but the advection can become unstable at fine grids. Try:
- Reduce `dt` (e.g., 0.0005 instead of 0.001)
- Reduce grid size (e.g., 64x64 instead of 128x128)
- Increase `nu` (more viscous fluid is easier to simulate)

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
# Windows
run.bat

# Mac/Linux
./run.sh
```
