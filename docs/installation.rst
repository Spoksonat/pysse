Installation
============

Requirements
------------

- Python 3.10 or newer.
- ``pip`` available in your environment.

Install
-------

From the repository root:

.. code-block:: bash

   python3 -m venv pysse_env
   source pysse_env/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   pip install -e .

The editable install registers ``pysse`` in your environment. Without it,
imports such as ``from pysse import SSE`` will fail in notebooks and scripts.

Documentation dependencies
--------------------------

To build the HTML documentation locally:

.. code-block:: bash

   pip install -e ".[docs]"
   cd docs
   make clean html

See the :doc:`quickstart` page for the runtime workflow after installation.
