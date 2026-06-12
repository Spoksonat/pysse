Quickstart
==========

From the repository root:

1. Create and activate a virtual environment:

   .. code-block:: bash

      python3 -m venv pysse_env
      source pysse_env/bin/activate

2. Install dependencies and register the package:

   .. code-block:: bash

      pip install --upgrade pip
      pip install -r requirements.txt
      pip install -e .

   The editable install is required so notebooks and scripts can import the
   package with:

   .. code-block:: python

      from pysse import SSE, ElectricFieldPulse

3. Run one of the notebooks in ``notebooks/`` to validate the workflow. See
   :doc:`examples` for a description of each test case.
