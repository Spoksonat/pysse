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

```bash
python3 -m venv pysse_env
source pysse_env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Optional editable install:

```bash
pip install -e .
```

## Data policy

- `data_minimal/` contains the smallest input files required to run examples.
- `notebooks/` are exploratory tests demonstrating how classes are used.
- `artifacts/` stores generated figures or other notebook outputs.

## Documentation

Build docs locally with:

```bash
cd docs
sphinx-build -b html . _build/html
```

Documentation includes:

- mathematical model used by `SSE` in `class_sse.py`,
- pulse-modeling section for `Electric_Field_Pulse` in `class_field.py`,
- API reference generated from docstrings.
