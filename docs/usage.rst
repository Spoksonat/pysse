Usage
=====

Package import
--------------

.. code-block:: python

   from pysse import SSE, Electric_Field_Pulse

Notebook workflow
-----------------

- Build pulse inputs with ``Electric_Field_Pulse``.
- Prepare ``Gamma`` and initial populations.
- Instantiate ``SSE`` and read:
  - ``population_average``,
  - ``population_std``.

Data directories
----------------

- ``data_minimal/``: smallest files needed by the models.
- ``artifacts/``: generated images and derived outputs.
