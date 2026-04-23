# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

from datetime import datetime
import re
import subprocess


# -- Utilities ----------------------------------------------------------------
def get_version(src_dir: str) -> str:
    """
    Extracts the version string from a CMakeLists.txt file.

    The function looks for a line formatted like:
        project(NAME VERSION 1.2.0 LANGUAGES CXX)
    and returns the version (e.g., "1.2.0").

    Parameters
    ----------
    src_dir : str
        The directory including CMakeLists.txt

    Returns
    -------
    str
        The version string if found.

    Raises
    ------
    ValueError
        If the version cannot be found in the file.
    """
    cmake_file = src_dir + "CMakeLists.txt"

    # Define a regex pattern to capture the version string after 'VERSION'.
    # Explanation:
    #   - project\(: matches the literal "project(".
    #   - [^)]*?: lazily matches any characters except the closing parenthesis.
    #   - \bVERSION\s+: matches the word "VERSION" followed by one or more spaces.
    #   - ([\d\.]+): capture group for the version number (digits and dots).
    pattern = re.compile(r"project\([^)]*\bVERSION\s+([\d\.]+)", re.IGNORECASE)
    with open(cmake_file, "r", encoding="utf-8") as f:
        content = f.read()

    match = pattern.search(content)
    if match:
        return match.group(1)
    else:
        raise ValueError("Version information not found in the CMakeLists.txt file.")


def get_HEAD_short_hash() -> str:
    """
    Retrieve the short hash of the last commit (HEAD) in the current git repository.

    Returns:
        str: The short hash of the HEAD commit.

    Raises:
        subprocess.CalledProcessError: If the git command fails (e.g., not in a git repository).
        FileNotFoundError: If git is not installed or not in PATH.
    """
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


# -- Project information -----------------------------------------------------
project = "Kokkos Comm"
author = "Kokkos Project Contributors"
copyright = f"2024-{datetime.now().year}, {author}"

version = f"{get_version('../')}-dev{get_HEAD_short_hash()}"
release = "latest"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_show_sphinx = False
