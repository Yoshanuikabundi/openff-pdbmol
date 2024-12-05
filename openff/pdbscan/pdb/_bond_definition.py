from dataclasses import dataclass
from typing import Literal

__all__ = [
    "BondDefinition",
    "PEPTIDE_BOND",
    "LINKING_TYPES",
]


@dataclass
class BondDefinition:
    """
    Description of a bond in a residue from the Chemical Component Dictionary (CCD).
    """

    atom1: str
    atom2: str
    order: int
    aromatic: bool
    stereo: Literal["E", "Z"] | None


# TODO: Fill in this data
PEPTIDE_BOND = BondDefinition(
    atom1="C", atom2="N", order=1, aromatic=False, stereo=None
)
LINKING_TYPES: dict[str, BondDefinition | None] = {
    # "D-beta-peptide, C-gamma linking".upper(): [],
    # "D-gamma-peptide, C-delta linking".upper(): [],
    # "D-peptide COOH carboxy terminus".upper(): [],
    # "D-peptide NH3 amino terminus".upper(): [],
    # "D-peptide linking".upper(): [],
    # "D-saccharide".upper(): [],
    # "D-saccharide, alpha linking".upper(): [],
    # "D-saccharide, beta linking".upper(): [],
    # "DNA OH 3 prime terminus".upper(): [],
    # "DNA OH 5 prime terminus".upper(): [],
    # "DNA linking".upper(): [],
    # "L-DNA linking".upper(): [],
    # "L-RNA linking".upper(): [],
    # "L-beta-peptide, C-gamma linking".upper(): [],
    # "L-gamma-peptide, C-delta linking".upper(): [],
    # "L-peptide COOH carboxy terminus".upper(): [],
    # "L-peptide NH3 amino terminus".upper(): [],
    "L-peptide linking".upper(): PEPTIDE_BOND,
    # "L-saccharide".upper(): [],
    # "L-saccharide, alpha linking".upper(): [],
    # "L-saccharide, beta linking".upper(): [],
    # "RNA OH 3 prime terminus".upper(): [],
    # "RNA OH 5 prime terminus".upper(): [],
    # "RNA linking".upper(): [],
    "non-polymer".upper(): None,
    # "other".upper(): [],
    "peptide linking".upper(): PEPTIDE_BOND,
    "peptide-like".upper(): PEPTIDE_BOND,
    # "saccharide".upper(): [],
}
