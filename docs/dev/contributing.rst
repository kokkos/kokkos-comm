Contributing
============

Code Formatting
---------------

All code should be formatted by clang-format 8:

.. code-block:: bash

  shopt -s globstar
  clang-format-8 -i {src,unit_tests,perf_tests}/**/*.[ch]pp
