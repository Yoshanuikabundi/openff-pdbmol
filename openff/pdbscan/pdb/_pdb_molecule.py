# TODO: Replace this with openff molecules?

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from typing import (
    Any,
    Generator,
    Literal,
)

import numpy as np
from networkx import Graph

from openff.toolkit import Molecule
from openff.units import unit

__all__ = [
    "PDBAtom",
    "PDBBond",
    "PDBMolecule",
]


@dataclass
class PDBAtom:
    atomic_number: int
    formal_charge: int
    is_aromatic: bool | None
    stereochemistry: Literal["S", "R", None]
    name: str
    x: float | None = None
    y: float | None = None
    z: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def coords(self) -> tuple[float, float, float] | None:
        if self.x is None or self.y is None or self.z is None:
            return None
        else:
            return (self.x, self.y, self.z)


@dataclass
class PDBBond:
    atom1: int
    atom2: int
    bond_order: int | None = None
    is_aromatic: bool | None = None
    stereochemistry: Literal["E", "Z", None] = None


@dataclass
class PDBMolecule:
    atoms: list[PDBAtom] = field(default_factory=list)
    bonds: list[PDBBond] = field(default_factory=list)
    properties: dict[str, Any] = field(default_factory=dict)

    def add_atom(self, atom: PDBAtom):
        self.atoms.append(atom)

    def add_bond(self, bond: PDBBond):
        self.bonds.append(bond)

    def atoms_bonded_to(self, index: int) -> Generator[int, None, None]:
        for bond in self.bonds:
            if bond.atom1 == index:
                yield bond.atom2
            if bond.atom2 == index:
                yield bond.atom1

    def are_bonded(self, atom1: int, atom2: int) -> bool:
        for bond in self.bonds:
            if (bond.atom1 == atom1 and bond.atom2 == atom2) or (
                bond.atom1 == atom2 and bond.atom2 == atom1
            ):
                return True
        return False

    def is_empty(self) -> bool:
        return self.n_atoms == 0 and self.n_bonds == 0 and len(self.properties) == 0

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    def get_bond_network(self) -> dict[int, set[int]]:
        network: dict[int, set[int]] = {}
        for bond in self.bonds:
            network.setdefault(bond.atom1, set()).add(bond.atom2)
            network.setdefault(bond.atom2, set()).add(bond.atom1)
        return network

    def identify_linkers(self, linked_atomname: str) -> tuple[int, PDBAtom, set[int]]:
        possible_partners = [
            (i, atom)
            for i, atom in enumerate(self.atoms)
            if atom.name == linked_atomname
        ]
        bond_network = self.get_bond_network()
        for partner, partner_atom in possible_partners:
            leavers = set()
            candidates = set(bond_network[partner])
            while candidates:
                candidate = candidates.pop()
                candidate_atom = self.atoms[candidate]
                if candidate_atom.metadata.get("leaving", False):
                    leavers.add(candidate)
                    candidates.update(bond_network[candidate] - leavers)
            if leavers:
                return (partner, partner_atom, leavers)

        meta = self.atoms[0].metadata
        raise ValueError(
            f"""No partners found in {meta["residue_name"]}#{meta["res_seq"]}: expected {
                linked_atomname
            }, found {[
                (atom.name, atom.metadata.get("atom_serial", ""))
                for (_, atom) in possible_partners
            ]}, but none of them have leaving atoms"""
        )

    def to_openff_molecule(self) -> Molecule:
        molecule = Molecule()
        molecule.properties.update(self.properties)
        conformer = []

        for atom in self.atoms:
            molecule._add_atom(
                atomic_number=atom.atomic_number,
                formal_charge=atom.formal_charge,
                is_aromatic=atom.is_aromatic,
                stereochemistry=atom.stereochemistry,
                name=atom.name,
                metadata=atom.metadata,
                invalidate_cache=False,
            )

            if atom.coords is None:
                # TODO: Come up with something clever here
                conformer.append((0.0, 0.0, 0.0))
            else:
                conformer.append(atom.coords)

        for bond in self.bonds:
            molecule._add_bond(
                atom1=bond.atom1,
                atom2=bond.atom2,
                bond_order=bond.bond_order,
                is_aromatic=bond.is_aromatic,
                stereochemistry=bond.stereochemistry,
                invalidate_cache=False,
            )

        molecule._invalidate_cached_properties()

        molecule.add_conformer(np.asarray(conformer) * unit.angstrom)
        return molecule

    def to_networkx(self) -> Graph:
        return self.to_openff_molecule().to_networkx()

    def to_smiles(self) -> str:
        return self.to_openff_molecule().to_smiles()

    def combine_with(self, other: "PDBMolecule"):
        """
        Combine molecules by unifying leaving atoms with the opposite molecule

        Preserves leaving annotations in ``other`` but not in ``self``.
        """
        # If this is empty, short circuit
        if self.is_empty():
            self.atoms = deepcopy(other.atoms)
            self.bonds = deepcopy(other.bonds)
            self.properties = deepcopy(other.properties)
            return

        # Identify the bond linking the two molecules
        self_linking_bond = self.properties["linking_bond"]
        other_linking_bond = other.properties["linking_bond"]
        if self_linking_bond is None:
            raise ValueError(
                f"Residue {self.atoms[-1].metadata['res_name']} does not form linkages"
            )
        if other_linking_bond is None:
            raise ValueError(
                f"Residue {other.atoms[0].metadata['res_name']} does not form linkages"
            )
        if self_linking_bond != other_linking_bond:
            raise ValueError(
                f"Residue {self.atoms[-1].metadata['res_name']} cannot be linked to"
                + f" Residue {other.atoms[0].metadata['res_name']}"
            )

        # Identify the atoms participating in the bond and those leaving
        this_partner, this_partner_atom, this_leavers = self.identify_linkers(
            self_linking_bond.atom1
        )
        other_partner, other_partner_atom, other_leavers = other.identify_linkers(
            self_linking_bond.atom2
        )

        # Add atoms
        self.properties.update(other.properties)
        self_to_combined: dict[int, int] = {}
        n_atoms_removed = 0
        for i, atom in list(enumerate(self.atoms)):
            if i in this_leavers:
                del self.atoms[i - n_atoms_removed]
                n_atoms_removed += 1
            else:
                atom.metadata.update({"leaving": False})
                self_to_combined[i] = i - n_atoms_removed

        other_to_combined: dict[int, int] = {}
        for i, atom in enumerate(other.atoms):
            if i in other_leavers:
                continue

            other_to_combined[i] = len(self.atoms)
            self.add_atom(deepcopy(atom))

        # Add bonds
        n_bonds_removed = 0
        for i, bond in list(enumerate(self.bonds)):
            if bond.atom1 in this_leavers or bond.atom2 in this_leavers:
                del self.bonds[i - n_bonds_removed]
                n_bonds_removed += 1
            else:
                bond.atom1 = self_to_combined[bond.atom1]
                bond.atom2 = self_to_combined[bond.atom2]

        self.add_bond(
            PDBBond(
                atom1=self_to_combined[this_partner],
                atom2=other_to_combined[other_partner],
                bond_order=self_linking_bond.order,
                is_aromatic=self_linking_bond.aromatic,
                stereochemistry=self_linking_bond.stereo,
            )
        )

        for bond in other.bonds:
            if bond.atom1 in other_leavers or bond.atom2 in other_leavers:
                continue

            self.add_bond(
                PDBBond(
                    atom1=other_to_combined[bond.atom1],
                    atom2=other_to_combined[bond.atom2],
                    bond_order=bond.bond_order,
                    is_aromatic=bond.is_aromatic,
                    stereochemistry=bond.stereochemistry,
                )
            )

    def sort_atoms_by_metadata(self, key: str):
        enumerated_atoms = list(enumerate(self.atoms))
        enumerated_atoms.sort(key=lambda t: t[1].metadata.get(key, 0xFFFFFF))
        old_to_new = {old: new for (new, (old, _)) in enumerate(enumerated_atoms)}
        for bond in self.bonds:
            bond.atom1 = old_to_new[bond.atom1]
            bond.atom2 = old_to_new[bond.atom2]
        self.atoms = [atom for _, atom in enumerated_atoms]
