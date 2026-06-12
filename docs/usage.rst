Usage
=====

Package import
--------------

.. code-block:: python

   from pysse import SSE, ElectricFieldPulse

Notebook workflow
-----------------

- Build pulse inputs with ``ElectricFieldPulse``.
- Prepare the ``gamma`` channel table and initial populations.
- Instantiate ``SSE`` and read:
  - ``population_average``,
  - ``population_std``.

Data directories
----------------

- ``data_minimal/``: smallest files needed by the models.
- ``artifacts/``: generated images and derived outputs.
