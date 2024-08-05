"""
Stats collection from the PDB databank.
"""

# Add imports here
from .polars import load_coords, COORD_LINE_SCHEMA
from ._pdbscan import nglview_show_openmm, scan_pdb

# By default, imported items are not rendered in the docs unless they are
# included in __all__.
__all__ = [
    "load_coords",
    "nglview_show_openmm",
    "COORD_LINE_SCHEMA",
    "scan_pdb",
]

# Handle versioneer
from ._version import get_versions

versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
