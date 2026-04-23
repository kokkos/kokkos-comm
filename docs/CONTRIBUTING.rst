************
Contributing
************

Workflow
========

Contributions are accepted as pull requests (PR) against the ``develop`` branch on `GitHub <https://github.com/kokkos/kokkos-comm/pulls>`_.
Unless there are extenuating circumstances, PRs must pass all automatic tests before being merged, and will require approvals from core team members.
Please freely use GitHub Issues/Discussions, the Kokkos Team Slack, or emails to discuss any proposed contributions.

In very limited circumstances, modifications may need to be tested as branches in the repository rather than pull requests.
Such changes should always be made in consultation with the core development team.

For a more detailed overview of what is expected of a "good" PR, please refer to `the related section <https://kokkos.org/kokkos-core-wiki/developer-guides/prs-and-reviews.html>`_ in the Kokkos documentation.


Pre-commit hooks
================

Kokkos Comm provides a pre-commit configuration to allow contributors to easily check that their changes will pass CI with regards to code formatting, spell-checking, etc.

To enable it on your machine, run the following commands (or check `the pre-commit docs <https://pre-commit.com/>`):

.. code-block:: console

    $ pip install pre-commit
    $ pre-commit install

Then, you can either manually run the pre-commit hooks with:

.. code-block:: console

    $ pre-commit run [--all-files]

Or let Git run them automatically when committing.

Kokkos Comm pre-commits perform the following checks:

* Correct code formatting in C++ files (using ``clang-format``)

* Correct formatting in CMake files (using ``gersemi``)

* Correctness of YAML files (mainly CI config)

* Trim trailing white space

* Fix missing new line in EOF

* Correct spelling and fixing typos

* Large files haven't been inadvertently added


Code formatting
===============

All code shall be formatted with ``clang-format`` v14.x.
If not using the provided pre-commit hooks, one can format the files using the following command:

.. code-block:: console

    $ shopt -s globstar
    $ clang-format-14 -i {src,unit_tests,perf_tests}/**/*.[ch]pp


Alternatively, you can use Docker/Podman:

.. code-block:: console

    $ shopt -s globstar
    $ podman run --rm -v ${PWD}:/src ghcr.io/cwpearson/clang-format-14 clang-format -i {src,unit_tests,perf_tests}/**/*.[ch]pp

.. important:: The above command expects ``$PWD`` to be the Kokkos Comm tree.


Site-Specific Documentation
===========================

These sites may be non-public and usually require credentials or affiliation at the institution to access:

* `Sandia National Laboratories <https://gitlab-ex.sandia.gov/cwpears/kokkos-comm-internal/-/wikis/home>`_


Behavioral Expectations
=======================

Those who are unwilling or unable to collaborate in a respectful manner, regardless of time, place, or medium, are expected to redirect their effort and attention to other projects.
