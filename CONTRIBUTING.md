# Contributing to TabulixML

Thank you for your interest in contributing to **TabulixML**! We welcome contributions that improve reliability, performance, edge-case coverage, and documentation.

---

## Development Workflow

### 1. Clone the Project
Fork and clone the repository to your local machine:
```bash
git clone https://github.com/your-username/tabulixml.git
cd tabulixml
```

### 2. Install Dependencies
Create a virtual environment (optional but recommended) and install TabulixML in editable mode along with development dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

pip install -e ".[dev]"
```

### 3. Run the Test Suite
Ensure the existing test suite passes before making any changes:
```bash
pytest
```

### 4. Make Your Changes
- Maintain code simplicity and documentation integrity.
- Follow the core design principles:
  - **Immutability**: `inspect()`, `preview()`, and `clean()` must never mutate the caller's input DataFrame.
  - **Transparency**: Every change must be recorded in `history()`, and reported in `report()` and `preview()`.
  - **Safety**: Outliers, constant columns, and inconsistent category values must be detected and flagged as warnings, never deleted automatically.
- Keep external runtime dependencies strictly limited to `pandas`, `numpy`, and `scipy`.
- Add comprehensive pytest tests in `tests/` covering your changes and edge cases.

### 5. Verify and Test
Re-run pytest and ensure all tests pass with zero regressions:
```bash
pytest -v --tb=short
```

### 6. Submit a Pull Request
- Create a feature branch: `git checkout -b feature/your-feature-name`
- Commit your changes with clear, descriptive commit messages.
- Push your branch to GitHub and open a Pull Request against `main`.
- Describe the motivation, changes made, and test results in your PR description.
