"""
Classes for defining custom residues.
"""

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
from typing import (
    Collection,
    Literal,
    Mapping,
    Self,
)

from openff.toolkit import Molecule
from openff.units import elements

from ._pdb_molecule import PDBAtom, PDBBond, PDBMolecule

__all__ = [
    "AtomDefinition",
    "BondDefinition",
    "ResidueDefinition",
]


@dataclass(frozen=True)
class AtomDefinition:
    """
    Description of an atom in a residue from the Chemical Component Dictionary (CCD).
    """

    name: str
    synonyms: list[str]
    symbol: str
    leaving: bool
    charge: int
    aromatic: bool
    stereo: Literal["S", "R"] | None


@dataclass(frozen=True)
class BondDefinition:
    """
    Description of a bond in a residue from the Chemical Component Dictionary (CCD).
    """

    atom1: str
    atom2: str
    order: int
    aromatic: bool
    stereo: Literal["E", "Z"] | None


@dataclass(frozen=True)
class ResidueDefinition:
    """
    Description of a residue from the Chemical Component Dictionary (CCD).
    """

    residue_name: str
    description: str
    linking_bond: BondDefinition | None
    atoms: list[AtomDefinition]
    bonds: list[BondDefinition]

    def __post_init__(self):
        if self.linking_bond is None and True in {atom.leaving for atom in self.atoms}:
            raise ValueError(
                "Leaving atoms were specififed, but there is no linking bond"
            )

    @classmethod
    def from_molecule(
        cls,
        name: str,
        molecule: Molecule,
        linking_bond: BondDefinition | None = None,
        description: str = "",
    ) -> Self:
        atoms: list[AtomDefinition] = []
        for i, atom in enumerate(molecule.atoms):
            atoms.append(
                AtomDefinition(
                    name=atom.name,
                    synonyms=[],
                    symbol=atom.symbol,
                    leaving=bool(atom.metadata.get("leaving_atom")),
                    charge=atom.formal_charge,
                    stereo=atom.stereochemistry,
                    aromatic=atom.is_aromatic,
                )
            )
        bonds: list[BondDefinition] = []
        for bond in molecule.bonds:
            bonds.append(
                BondDefinition(
                    atom1=bond.atom1.name,
                    atom2=bond.atom2.name,
                    order=bond.bond_order,
                    aromatic=bond.is_aromatic,
                    stereo=bond.stereochemistry,
                )
            )

        return cls(
            residue_name=name,
            description=description,
            linking_bond=linking_bond,
            atoms=atoms,
            bonds=bonds,
        )

    @classmethod
    def from_capped_molecule(
        cls,
        name: str,
        molecule: Molecule,
        leaving_atom_indices: Collection[int],
        linking_bond: BondDefinition,
        description: str = "",
    ) -> Self:
        molecule = deepcopy(molecule)
        for i in leaving_atom_indices:
            molecule.atom(i).metadata["leaving_atom"] = True
        return cls.from_molecule(
            name=name,
            molecule=molecule,
            linking_bond=linking_bond,
            description=description,
        )

    @classmethod
    def from_smiles(
        cls,
        name: str,
        mapped_smiles: str,
        atom_names: Mapping[int, str],
        leaving_atoms: Collection[int] = (),
        linking_bond: BondDefinition | None = None,
        description: str = "",
    ) -> Self:
        molecule = Molecule.from_mapped_smiles(mapped_smiles)
        leaving_atom_indices = set(leaving_atoms)
        for i, atom in enumerate(molecule.atoms, start=1):
            if i in leaving_atom_indices:
                atom.metadata["leaving_atom"] = True
            atom.name = atom_names[i]

        return cls.from_molecule(
            name=name,
            molecule=molecule,
            linking_bond=linking_bond,
            description=description,
        )

    def to_openff_molecule(self) -> Molecule:
        molecule = Molecule()
        atoms = {}
        for atom in self.atoms:
            atoms[atom.name] = molecule.add_atom(
                atomic_number=elements.NUMBERS[atom.symbol],
                formal_charge=atom.charge,
                is_aromatic=atom.aromatic,
                stereochemistry=atom.stereo,
                name=atom.name,
                metadata={
                    "residue_name": self.residue_name,
                    "leaving": atom.leaving,
                },
            )

        for bond in self.bonds:
            molecule.add_bond(
                atom1=atoms[bond.atom1],
                atom2=atoms[bond.atom2],
                bond_order=bond.order,
                is_aromatic=bond.aromatic,
                stereochemistry=bond.stereo,
            )

        molecule.properties.update(
            {
                "linking_bond": self.linking_bond,
            }
        )

        return molecule

    def to_pdb_molecule(self) -> PDBMolecule:
        molecule = PDBMolecule()
        atoms: dict[str, int] = {}
        for atom in self.atoms:
            new_atom = PDBAtom(
                atomic_number=elements.NUMBERS[atom.symbol],
                formal_charge=atom.charge,
                is_aromatic=atom.aromatic,
                stereochemistry=atom.stereo,
                name=atom.name,
                metadata={
                    "residue_name": self.residue_name,
                    "leaving": atom.leaving,
                },
            )
            atoms[atom.name] = len(molecule.atoms)
            molecule.add_atom(new_atom)

        for bond in self.bonds:
            new_bond = PDBBond(
                atom1=atoms[bond.atom1],
                atom2=atoms[bond.atom2],
                bond_order=bond.order,
                is_aromatic=bond.aromatic,
                stereochemistry=bond.stereo,
            )
            molecule.add_bond(new_bond)

        molecule.properties.update(
            {
                "linking_bond": self.linking_bond,
            }
        )

        return molecule

    @cached_property
    def name_to_canonical_name(self) -> dict[str, str]:
        """Map from each atoms' name and synonyms to its name."""
        canonical_names = {atom.name for atom in self.atoms}
        mapping = {name: name for name in canonical_names}
        for atom in self.atoms:
            for synonym in atom.synonyms:
                if synonym in mapping and mapping[synonym] != atom.name:
                    raise ValueError(
                        f"synonym {synonym} degenerately defined for canonical"
                        + f" names {mapping[synonym]} and {atom.name} in"
                        + f" residue {self.residue_name}"
                    )
                if synonym in canonical_names:
                    raise ValueError(
                        f"synonym {synonym} of atom {atom.name} clashes with"
                        + f" another canonical name in residue {self.residue_name}"
                    )
                mapping[synonym] = atom.name
        return mapping
