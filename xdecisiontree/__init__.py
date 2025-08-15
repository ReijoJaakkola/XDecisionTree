# xdecisiontree/__init__.py

"""
XDecisionTreeClassifier: Decision tree classifier with human-readable rule extraction.
"""

# Version
__version__ = "1.0.0"

# Import the main class
from .xdecisiontree import XDecisionTreeClassifier

# Optional: expose submodules if you plan to add more later
# e.g., from .utils import some_function

# Define what is exported when using `from xdecisiontree import *`
__all__ = ["XDecisionTreeClassifier"]