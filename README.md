# PySSE

Python project for stochastic Schrödinger equation (SSE) simulations driven by
ultrafast electric-field pulses.

The repository is organized to separate reusable source code, notebook-based
experiments, minimum input data, and generated artifacts.

## Repository layout

```text
PySSE/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
├── src/
│   └── pysse/
│       ├── __init__.py
│       ├── class_sse.py
│       └── class_field.py
├── notebooks/
│   ├── SSE_first_test.ipynb
│   ├── SSE_second_test.ipynb
│   ├── SSE_third_test.ipynb
│   └── Vacuum_LiCN_qjump_EuMar.ipynb
├── data_minimal/
├── artifacts/
└── docs/
    ├── conf.py
    ├── index.rst
    ├── quickstart.rst
    ├── installation.rst
    ├── usage.rst
    ├── sse_model.rst
    ├── field_model.rst
    └── api.rst
```

## Quick start

From the repository root:

```bash
python3 -m venv pysse_env
source pysse_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

The editable install registers the `pysse` package in your environment so
notebooks and scripts can use `from pysse import SSE, ElectricFieldPulse`.

## Data policy

- `data_minimal/` contains the smallest input files required to run examples.
- `notebooks/` are exploratory tests demonstrating how classes are used.
- `artifacts/` stores generated figures or other notebook outputs.

## Documentation

Build docs locally with:

```bash
cd docs
make clean html
open _build/html/index.html
```

If you add or rename ``.rst`` pages, run ``make clean html`` so Sphinx does
not reuse a stale cached environment (incremental builds may skip new files).

Online docs: https://spoksonat.github.io/pysse/

Documentation includes:

- notebook examples in `notebooks/` (`examples.rst`),
- mathematical model used by `SSE` in `class_sse.py`,
- pulse-modeling section for `ElectricFieldPulse` in `class_field.py`,
- API reference generated from docstrings.
