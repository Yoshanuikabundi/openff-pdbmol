"""
Exceptions for the PDB loader.
"""

__all__ = [
    "NoMatchingResidueDefinitionError",
]


class NoMatchingResidueDefinitionError(ValueError):
    """Exception raised when a residue is missing from the database"""

    def __init__(
        self,
        res_name: str,
        res_seq: int,
        chain_id: str,
        i_code: str,
    ):
        super().__init__(
            f"No residue definitions covered all atoms in {res_name}#{res_seq}{i_code.strip()}:{chain_id}"
        )
