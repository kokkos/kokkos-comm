***************************
Extending the documentation
***************************

Using reStructuredText
======================

Useful resources:

* `Basics of rST <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
* `Documenting C++ with rST <https://www.sphinx-doc.org/en/master/usage/domains/cpp.html>`_


Building a local copy of the docs
=================================

1. Create a Python virtual environment at ``.venv``:

    .. code-block:: console

        $ python3 -m venv .venv

2. Activate the virtual environment:

    .. code-block:: console

        $ source .venv/bin/activate

3. Install the documentation pre-requisites:

    .. code-block:: console

        $ pip install -r docs/requirements.txt

4. Build the documentation:

    .. code-block:: console

        $ make -C docs html

5. Open the documentation in your favorite browser:

    .. code-block:: console

        $ <BROWSER> docs/_build/html/index.html


Contributing to the docs
========================

Documentation contributions follow the same workflow as code contributions.
Open a pull request (PR) against the ``develop`` branch on
`GitHub <https://github.com/kokkos/kokkos-comm/pulls>`_, and make sure the
build and any pre-commit hooks pass before requesting review (see
:doc:`CONTRIBUTING </CONTRIBUTING>` for the full contributor workflow).

A few things to keep in mind when working on the docs:

* **Format:** write documentation in reStructuredText (rST). See the sections
  above for useful resources. If you add a new page, remember to wire it into a
  ``toctree`` so that Sphinx picks it up.

* **Pre-commit hooks:** the project's pre-commit configuration also checks
  documentation sources for spelling mistakes, trailing whitespace, and missing
  newlines at EOF. Install and run the hooks as described in
  :doc:`CONTRIBUTING </CONTRIBUTING>`.

* **Verify locally:** before opening a PR, build a local copy of the docs (see
  the previous section) and skim the rendered output for warnings or layout
  issues. Sphinx emits warnings for broken cross-references and malformed
  directives, so treat a clean build as the baseline.
