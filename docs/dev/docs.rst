Extending the Documentation
===========================

Using reStructedText
--------------------

* `Basics of rST <https://www.sphinx-doc.org/en/master/usage/restructuredtext/basics.html>`_
* `Documenting C++ with rST <https://www.sphinx-doc.org/en/master/usage/domains/cpp.html>`_

Building a local copy of the docs
---------------------------------

.. code-block:: bash

    # create a venv at .venv
    python3 -m venv .venv

    # activate the venv
    . .venv/bin/activate

    # install the docs prereqs
    pip install -r docs/requirements.txt

    # builds the docs
    make -C docs html

    # open docs/_build/html/index.html in your favorite browser
    