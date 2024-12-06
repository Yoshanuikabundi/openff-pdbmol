"""
Exceptions for the PDB loader.
"""

__all__ = [
    "NoMatchingResidueDefinitionError",
    "MultipleMatchingResidueDefinitionsError",
]


class NoMatchingResidueDefinitionError(ValueError):
    """Exception raised when a residue is missing from the database"""

    def __init__(self):
        super().__init__("No residue definitions covered all atoms in a residue")


class MultipleMatchingResidueDefinitionsError(ValueError):
    def __init__(self):
        super().__init__("Multiple residue definitions matched a residue")
