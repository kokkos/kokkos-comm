Contributing
============

Workflow
--------

Contributions are accepted as pull requests against the ``develop`` branch.
Unless there are extenuating circumstances, pull requests must pass all automatic tests before being merged, and will require approvals from core team members.
Please freely use Github Issues/Discussions, the Kokkos Team slack, or emails to discuss any proposed contributions.

In very limited circumstances, modifications may need to be tested as branches in the repository rather than pull requests.
Such changes should always be made in consultation with the core development team.

Code Formatting
---------------

All code shall be formatted by clang-format 8:

.. code-block:: bash

  shopt -s globstar
  clang-format-8 -i {src,unit_tests,perf_tests}/**/*.[ch]pp


Behavioral Expectations
-----------------------

Those who are unwilling or unable to collaborate in a respectful manner, regardless of time, place, or medium, are expected to redirect their effort and attention to other projects.
