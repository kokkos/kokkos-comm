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

All code shall be formatted by clang-format 14:

.. code-block:: bash

  shopt -s globstar
  clang-format-14 -i {src,unit_tests,perf_tests}/**/*.[ch]pp


Alternatively, you can use docker/podman: (expects $PWD to be the kokkos-comm tree)

.. code-block:: bash

  shopt -s globstar
  podman run -v $PWD:/src xianpengshen/clang-tools:14 clang-format -i {src,unit_tests,perf_tests}/**/*.[ch]pp

Site-Specific Documentation
---------------------------

These sites may be non-public and usually require credentials or affiliation at the institution to access.

* `Sandia National Laboratories <https://gitlab-ex.sandia.gov/cwpears/kokkos-comm-internal/-/wikis/home>`_

Behavioral Expectations
-----------------------

Those who are unwilling or unable to collaborate in a respectful manner, regardless of time, place, or medium, are expected to redirect their effort and attention to other projects.
