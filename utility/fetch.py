"""
File Finder Utility

This utility function helps find the file path of a given filename within a specified directory.
It recursively searches through the directory and its subdirectories to locate the target file.

Usage:
1. Call the `find_file_path` function, providing the directory path and the target filename.
2. The function returns the full file path if the file is found; otherwise, it returns None.
"""

import os


def find_file_path(filename):
    """__doc__
    Find the file path of a given filename within the current working directory and its subdirectories.

    Args:
        filename (str): The name of the file to find.

    Returns:
        str: The full file path if the file is found, None otherwise.
    """
    for root, dirs, files in os.walk(os.getcwd()):
        if filename in files:
            return os.path.join(root, filename)
    return None
