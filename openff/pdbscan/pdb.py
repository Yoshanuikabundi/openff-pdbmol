import dataclasses
import gzip
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from io import StringIO
from os import PathLike
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generator,
    Iterable,
    Iterator,
    Literal,
    Self,
    TypeVar,
)
from urllib.request import urlopen

import numpy as np
from networkx import Graph
from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader

from openff.toolkit import Molecule, Topology
from openff.units import Quantity, elements, unit


class __UNSET__:
    pass


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
            f"No partners found in {meta["residue_name"]}#{meta["res_seq"]}: expected {
                linked_atomname
            }, found {[
                (atom.name, atom.metadata.get("atom_serial", ""))
                for (_, atom) in possible_partners
            ]}, but none of them have leaving atoms"
        )

    def titrate_added_atoms(self, protonation_variants: Mapping[str, list[str]]):
        atoms_to_remove = []
        for i, atom in enumerate(self.atoms):
            if (
                "pdb_index" not in atom.metadata
                and atom.atomic_number == 1
                and atom.name
                in protonation_variants.get(atom.metadata.get("residue_name", ""), [])
            ):
                candidates = [*self.atoms_bonded_to(i)]
                if len(candidates) == 1:
                    bonded_atom = self.atoms[candidates[0]]
                    bonded_atom.formal_charge -= 1
                    atoms_to_remove.append(i)

        n_atoms_removed = 0
        for i in atoms_to_remove:
            i = i - n_atoms_removed
            for bond_index, bond in list(enumerate(self.bonds)):
                if bond.atom1 == i or bond.atom2 == i:
                    del self.bonds[bond_index]
                bond.atom1 = bond.atom1 if bond.atom1 < i else bond.atom1 - 1
                bond.atom2 = bond.atom2 if bond.atom2 < i else bond.atom2 - 1
            del self.atoms[i]
            n_atoms_removed += 1

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
        self_linking_type = self.properties["linking_type"]
        other_linking_type = other.properties["linking_type"]

        if self_linking_type not in LINKING_TYPES:
            raise ValueError(f"Unknown linking type {self_linking_type}")
        if other_linking_type not in LINKING_TYPES:
            raise ValueError(f"Unknown linking type {other_linking_type}")

        self_linking_bond = LINKING_TYPES[self_linking_type]
        other_linking_bond = LINKING_TYPES[other_linking_type]
        if self_linking_bond is None:
            raise ValueError(f"Linking type {self_linking_type} does not form linkages")
        if other_linking_bond is None:
            raise ValueError(
                f"Linking type {other_linking_type} does not form linkages"
            )
        if self_linking_bond != other_linking_bond:
            raise ValueError(
                f"Molecule of linking type {self_linking_type} cannot be linked to"
                + f" molecule of linking type {other_linking_type}"
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
                bond_order={"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}[
                    self_linking_bond.order
                ],
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


T = TypeVar("T")


def unwrap(container: Iterable[T], msg: str = "") -> T:
    """
    Unwrap an iterable only if it has a single element; raise ValueError otherwise
    """
    if msg:
        msg += ": "

    iterator = iter(container)

    try:
        value = next(iterator)
    except StopIteration:
        raise ValueError(msg + "container has no elements")

    try:
        next(iterator)
    except StopIteration:
        return value

    raise ValueError(msg + "container has multiple elements")


def flatten(container: Iterable[Iterable[T]]) -> Iterable[T]:
    for inner in container:
        yield from inner


def float_or_unknown(s: str) -> float | None:
    if s == "?":
        return None
    return float(s)


@dataclass
class AtomDefinition:
    """
    Description of an atom in a residue from the Chemical Component Dictionary (CCD).
    """

    name: str
    synonyms: list[str]
    symbol: str
    leaving: bool
    x: float | None
    y: float | None
    z: float | None
    charge: int
    aromatic: bool
    stereo: Literal["S", "R"] | None


@dataclass
class BondDefinition:
    """
    Description of a bond in a residue from the Chemical Component Dictionary (CCD).
    """

    atom1: str
    atom2: str
    order: Literal["SING", "DOUB", "TRIP", "QUAD", "AROM", "DELO", "PI", "POLY"]
    aromatic: bool
    stereo: Literal["E", "Z"] | None


@dataclass
class ResidueDefinition:
    """
    Description of a residue from the Chemical Component Dictionary (CCD).
    """

    residueName: str
    description: str
    smiles: list[str]
    linking_type: str
    atoms: list[AtomDefinition]
    bonds: list[BondDefinition]

    @classmethod
    def from_ccd_str(cls, s) -> Self:
        # TODO: Handle residues like CL with a single atom properly (no tables)
        data = []
        with StringIO(s) as file:
            PdbxReader(file).read(data)
        block = data[0]

        residueName = (
            block.getObj("chem_comp").getValue("mon_nstd_parent_comp_id").upper()
        )
        if residueName == "?":
            residueName = block.getObj("chem_comp").getValue("id").upper()
        residue_description = block.getObj("chem_comp").getValue("name")
        linking_type = block.getObj("chem_comp").getValue("type").upper()

        descriptorsData = block.getObj("pdbx_chem_comp_descriptor")
        typeCol = descriptorsData.getAttributeIndex("type")
        smilesCol = descriptorsData.getAttributeIndex("descriptor")
        smiles = []
        for row in descriptorsData.getRowList():
            if row[typeCol] in ["SMILES", "SMILES_CANONICAL"]:
                smiles.append(row[smilesCol])
                break

        atomData = block.getObj("chem_comp_atom")
        atomNameCol = atomData.getAttributeIndex("atom_id")
        altAtomNameCol = atomData.getAttributeIndex("alt_atom_id")
        symbolCol = atomData.getAttributeIndex("type_symbol")
        leavingCol = atomData.getAttributeIndex("pdbx_leaving_atom_flag")
        xCol = atomData.getAttributeIndex("pdbx_model_Cartn_x_ideal")
        yCol = atomData.getAttributeIndex("pdbx_model_Cartn_y_ideal")
        zCol = atomData.getAttributeIndex("pdbx_model_Cartn_z_ideal")
        chargeCol = atomData.getAttributeIndex("charge")
        aromaticCol = atomData.getAttributeIndex("pdbx_aromatic_flag")
        stereoCol = atomData.getAttributeIndex("pdbx_stereo_config")

        atoms = [
            AtomDefinition(
                name=row[atomNameCol],
                synonyms=(
                    [row[altAtomNameCol]]
                    if row[altAtomNameCol] != row[atomNameCol]
                    else []
                ),
                symbol=row[symbolCol][0:1].upper() + row[symbolCol][1:].lower(),
                leaving=row[leavingCol] == "Y",
                x=float_or_unknown(row[xCol]),
                y=float_or_unknown(row[yCol]),
                z=float_or_unknown(row[zCol]),
                charge=int(row[chargeCol]),
                aromatic=row[aromaticCol] == "Y",
                stereo=None if row[stereoCol] == "N" else row[stereoCol],
            )
            for row in atomData.getRowList()
        ]

        bondData = block.getObj("chem_comp_bond")
        if bondData is not None:
            atom1Col = bondData.getAttributeIndex("atom_id_1")
            atom2Col = bondData.getAttributeIndex("atom_id_2")
            orderCol = bondData.getAttributeIndex("value_order")
            aromaticCol = bondData.getAttributeIndex("pdbx_aromatic_flag")
            stereoCol = bondData.getAttributeIndex("pdbx_stereo_config")
            bonds = [
                BondDefinition(
                    atom1=row[atom1Col],
                    atom2=row[atom2Col],
                    order=row[orderCol],
                    aromatic=row[aromaticCol] == "Y",
                    stereo=None if row[stereoCol] == "N" else row[stereoCol],
                )
                for row in bondData.getRowList()
            ]
        else:
            bonds = []

        ret = cls(
            residueName=residueName,
            smiles=smiles,
            description=residue_description,
            linking_type=linking_type,
            atoms=atoms,
            bonds=bonds,
        )
        return ret

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
                    "residue_name": self.residueName,
                    "leaving": atom.leaving,
                },
            )

        for bond in self.bonds:
            # TODO: Handle CCD entries that have not been kekulized
            molecule.add_bond(
                atom1=atoms[bond.atom1],
                atom2=atoms[bond.atom2],
                bond_order={"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}[bond.order],
                is_aromatic=bond.aromatic,
                stereochemistry=bond.stereo,
            )

        # for smiles in self.smiles:
        #     try:
        #         from_smiles = Molecule.from_smiles(smiles)
        #     except ValueError:
        #         continue

        #     if molecule.is_isomorphic_with(
        #         from_smiles, atom_stereochemistry_matching=False
        #     ):
        #         raise ValueError(
        #             f"Molecule {molecule} from CCD entry {self.residueName}"
        #             + f" with SMILES {molecule.to_smiles()} does not match"
        #             + f" SMILES {smiles} from the CCD entry, parsed as"
        #             + f" {from_smiles}"
        #         )

        molecule.properties.update(
            {
                "linking_type": self.linking_type,
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
                    "residue_name": self.residueName,
                    "leaving": atom.leaving,
                },
            )
            atoms[atom.name] = len(molecule.atoms)
            molecule.add_atom(new_atom)

        for bond in self.bonds:
            new_bond = PDBBond(
                atom1=atoms[bond.atom1],
                atom2=atoms[bond.atom2],
                bond_order={"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}[bond.order],
                is_aromatic=bond.aromatic,
                stereochemistry=bond.stereo,
            )
            molecule.add_bond(new_bond)

        molecule.properties.update(
            {
                "linking_type": self.linking_type,
            }
        )

        return molecule

    @cached_property
    def _name_to_canonical_name(self) -> dict[str, str]:
        mapping = {}
        canonical_names = {atom.name for atom in self.atoms}
        for atom in self.atoms:
            for synonym in atom.synonyms:
                if synonym in mapping and mapping[synonym] != atom.name:
                    raise ValueError(
                        f"synonym {
                            synonym
                        } degenerately defined for canonical names {
                            mapping[synonym]
                        } and {
                            atom.name
                        } in residue {
                            self.residueName
                        }"
                    )
                if synonym in canonical_names:
                    raise ValueError(
                        f"synonym {
                        synonym
                    } of atom {
                        atom.name
                    } clashes with another canonical name in residue {
                        self.residueName
                    }"
                    )
                mapping[synonym] = atom.name
        return mapping

    def get_canonical_name(self, name: str) -> str:
        return self._name_to_canonical_name.get(name, name)

    def format_names_with_synonyms(self, res_name: str, atom_names: set[str]) -> str:
        atoms = []
        residue_synonyms = {atom.name: atom.synonyms for atom in self.atoms}
        for atom_name in atom_names:
            atom_synonyms = residue_synonyms.get(atom_name, [])
            if len(atom_synonyms) > 1:
                atoms.append(
                    f"{atom_name} (or one of its synonyms {','.join(atom_synonyms)})"
                )
            if len(atom_synonyms) == 1:
                atoms.append(f"{atom_name} (or its synonym {atom_synonyms[0]})")
            else:
                atoms.append(atom_name)
        return ", ".join(atoms)


def dec_hex(s: str) -> int:
    """
    Interpret a string as a decimal or hexadecimal integer.

    For a string of length n, the string is interpreted as decimal if the value
    is < 10^n. This makes the dec_hex representation identical to a decimal
    integer, except for strings that cannot be parsed as a decimal. For these
    strings, the first hexadecimal number is interpreted as 10^n, and subsequent
    numbers continue from there. For example, in PDB files, a fixed width column
    format, residue numbers for large systems follow this representation:

        "   1" -> 1
        "   2" -> 2
        ...
        "9999" -> 9999
        "A000" -> 10000
        "A001" -> 10001
        ...
        "A009" -> 10009
        "A00A" -> 10010
        "A00B" -> 10011
        ...
        "A00F" -> 10015
        "A010" -> 10016
        ...
    """

    try:
        return int(s, 10)
    except ValueError:
        n = len(s)
        parsed_as_hex = int(s, 16)
        smallest_hex = 0xA * 16 ** (n - 1)
        largest_dec = 10**n - 1
        return parsed_as_hex - smallest_hex + largest_dec + 1


@dataclass
class PdbData:
    model: list[int | None] = field(default_factory=list)
    serial: list[int] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    alt_loc: list[str] = field(default_factory=list)
    res_name: list[str] = field(default_factory=list)
    chain_id: list[str] = field(default_factory=list)
    res_seq: list[int] = field(default_factory=list)
    i_code: list[str] = field(default_factory=list)
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    z: list[float] = field(default_factory=list)
    occupancy: list[float] = field(default_factory=list)
    temp_factor: list[float] = field(default_factory=list)
    element: list[str] = field(default_factory=list)
    charge: list[int] = field(default_factory=list)
    terminated: list[bool] = field(default_factory=list)
    conects: list[set[int]] = field(default_factory=list)
    cryst1_a: float | None = None
    cryst1_b: float | None = None
    cryst1_c: float | None = None
    cryst1_alpha: float | None = None
    cryst1_beta: float | None = None
    cryst1_gamma: float | None = None

    def _append_coord_line(self, line: str):
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if hasattr(value, "append"):
                value.append(__UNSET__)
                assert value[-1] is __UNSET__

        self.model[-1] = None
        self.serial[-1] = int(line[6:11])
        self.name[-1] = line[12:16].strip()
        self.alt_loc[-1] = line[16].strip() or ""
        self.res_name[-1] = line[17:20].strip()
        self.chain_id[-1] = line[21].strip()
        self.res_seq[-1] = dec_hex(line[22:26])
        self.i_code[-1] = line[26].strip() or " "
        self.x[-1] = float(line[30:38])
        self.y[-1] = float(line[38:46])
        self.z[-1] = float(line[46:54])
        self.occupancy[-1] = float(line[54:60])
        self.temp_factor[-1] = float(line[60:66])
        self.element[-1] = line[76:78].strip()
        self.charge[-1] = int(line[78:80].strip() or 0)
        self.terminated[-1] = False
        self.conects[-1] = set()

        # Ensure we've assigned a value to every field
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if hasattr(value, "append"):
                assert value[-1] is not __UNSET__

    @classmethod
    def parse_pdb(cls, lines: Iterable[str]) -> Self:
        conects = {}
        # Read all CONECT records
        for line in lines:
            if line.startswith("CONECT "):
                a = int(line[6:11])
                bs = []
                for start, stop in [(11, 16), (16, 21), (21, 26), (26, 31)]:
                    try:
                        b = int(line[start:stop])
                    except (ValueError, IndexError):
                        continue
                    bs.append(b)
                    conects.setdefault(b, set()).add(a)
                conects.setdefault(a, set()).update(bs)

        model_n = None
        data = cls()
        for line in lines:
            if line.startswith("MODEL "):
                model_n = int(line[10:14])
            if line.startswith("ENDMDL "):
                model_n = None
            if line.startswith("HETATM") or line.startswith("ATOM  "):
                data._append_coord_line(line)
                data.model[-1] = model_n
                data.conects[-1].update(conects.get(data.serial[-1], []))
            if line.startswith("TER   "):
                terminated_resname = line[17:20].strip() or data.res_name[-1]
                terminated_chainid = line[21].strip() or data.chain_id[-1]
                terminated_resseq = dec_hex(line[22:26]) or data.res_seq[-1]
                for i in range(-1, -999, -1):
                    if (
                        data.res_name[i] == terminated_resname
                        and data.chain_id[i] == terminated_chainid
                        and data.res_seq[i] == terminated_resseq
                    ):
                        data.terminated[i] = True
                    else:
                        break
                else:
                    assert False, "last residue too big"
            if line.startswith("CRYST1"):
                data.cryst1_a = float(line[6:15])
                data.cryst1_b = float(line[15:24])
                data.cryst1_c = float(line[24:33])
                data.cryst1_alpha = float(line[33:40])
                data.cryst1_beta = float(line[40:47])
                data.cryst1_gamma = float(line[47:54])

        return data

    def residues(self):
        indices = []
        prev = None
        for i, atom in enumerate(
            zip(
                self.model,
                self.res_name,
                self.chain_id,
                self.res_seq,
                self.i_code,
            )
        ):
            if prev == atom or prev is None:
                indices.append(i)
            else:
                yield indices
                indices = [i]
            prev = atom

        yield indices

    def __getitem__(self, index) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)[index]
            for field in dataclasses.fields(self)
        }


class CcdCache(Mapping[str, list[ResidueDefinition]]):
    def __init__(
        self,
        path: Path,
        preload: list[str] = [],
        patches: dict[str, Callable[[ResidueDefinition], list[ResidueDefinition]]] = {},
    ):
        self.path = path.resolve()
        self.path.mkdir(parents=True, exist_ok=True)

        self.definitions = {}
        self.patches = dict(patches)

        self._load_protonation_variants()

        for file in path.glob("*.cif"):
            self._add_definition_from_str(file.read_text())

        for resname in set(preload) - set(self.definitions):
            self.get(resname)

    def __repr__(self):
        return f"CcdCache(path={
            self.path
        }, preload={
            list(self.definitions)
        }, patches={
            self.patches!r
        })"

    def __getitem__(self, key: str) -> list[ResidueDefinition]:
        res_name = key.upper()
        if res_name in self.definitions:
            return self.definitions[res_name]

        try:
            s = (self.path / f"{res_name.upper()}.cif").read_text()
        except Exception:
            s = self._download_cif(res_name)

        return self._add_definition_from_str(s, res_name=res_name)

    def _apply_patches(self, definition: ResidueDefinition) -> list[ResidueDefinition]:
        patch_res = self.patches.get(definition.residueName.upper(), lambda x: [x])
        patch_all = self.patches.get("*", lambda x: [x])
        patched_definitions = list(patch_all(definition))
        patched_definitions = list(
            flatten(patch_res(res) for res in patched_definitions)
        )
        return patched_definitions

    def _add_definition_from_str(
        self, s: str, res_name: str | None = None
    ) -> list[ResidueDefinition]:
        definition = ResidueDefinition.from_ccd_str(s)
        if res_name is None:
            res_name = definition.residueName.upper()

        patched_definitions = self._apply_patches(definition)

        assert all(
            res_name == definition.residueName.upper()
            for definition in patched_definitions
        )

        self.definitions.setdefault(res_name, []).extend(patched_definitions)
        return patched_definitions

    def _load_protonation_variants(self):
        path = self.path / "aa-variants-v1.cif.gz"
        if not path.exists():
            with urlopen(
                "https://files.wwpdb.org/pub/pdb/data/monomers/aa-variants-v1.cif.gz",
            ) as stream:
                b = stream.read()
            path.write_bytes(b)

        with gzip.open(path) as f:
            s = f.read().decode("utf-8")

        assert s.startswith("data_")
        for block in s[5:].split("\ndata_"):
            block = "data_" + block
            self._add_definition_from_str("data_" + block)

    def _download_cif(self, resname: str) -> str:
        with urlopen(
            f"https://files.rcsb.org/ligands/download/{resname.upper()}.cif",
        ) as stream:
            s = stream.read().decode("utf-8")
        path = self.path / f"{resname.upper()}.cif"
        path.write_text(s)
        return s

    def __contains__(self, value) -> bool:
        if value in ["UNK", "UNL"]:
            # These residue names are reserved for unknown ligands/peptide residues
            return False

        if len(value) == 3:
            # All residue names of three letters are assigned
            return True

        try:
            self.get(value)
            return True
        except Exception:
            return False

    def __iter__(self) -> Iterator[str]:
        return self.definitions.__iter__()

    def __len__(self) -> int:
        return self.definitions.__len__()


def fix_caps(res: ResidueDefinition) -> list[ResidueDefinition]:
    res.linking_type = "PEPTIDE LINKING"

    if res.residueName == "ACE":
        for atom in res.atoms:
            if atom.name == "H":
                atom.leaving = True
                break

    return [res]


# PROTONATION_VARIANTS = {
#     "HIS": [
#         "[N:1]([C@:2]([C:3](=[O:4])[O:11][H:21])([C:5]([C:6]1=[C:8]([H:18])[N:10]([H:20])[C:9]([H:19])=[N:7]1)([H:15])[H:16])[H:14])([H:12])[H:13]",
#         "[N:1]([C@:2]([C:3](=[O:4])[O:11][H:21])([C:5]([C:6]1=[C:8]([H:18])[N:10]=[C:9]([H:19])[N:7]1[H:17])([H:15])[H:16])[H:14])([H:12])[H:13]",
#         "[N:1]([C@:2]([C:3](=[O:4])[O:11][H:21])([C:5]([C:6]1=[C:8]([H:18])[N:10]=[C:9]([H:19])[N-:7]1)([H:15])[H:16])[H:14])([H:12])[H:13]",
#     ],
#     "GLU": [
#         "[N:1]([C@:2]([C:3](=[O:4])[O:10][H:19])([C:5]([C:6]([C:7](=[O:8])[O-:9])([H:16])[H:17])([H:14])[H:15])[H:13])([H:11])[H:12]",
#     ],
#     "ASP": [
#         "[N:1]([C@:2]([C:3](=[O:4])[O:9][H:16])([C:5]([C:6](=[O:7])[O-:8])([H:13])[H:14])[H:12])([H:10])[H:11]"
#     ],
#     "LYS": [
#         "[N:1]([C@:2]([C:3](=[O:4])[O:10][H:25])([C:5]([C:6]([C:7]([C:8]([N:9]([H:22])([H:23]))([H:20])[H:21])([H:18])[H:19])([H:16])[H:17])([H:14])[H:15])[H:13])([H:11])[H:12]"
#     ],
#     "ARG": [
#         "[N:1]([C@:2]([C:3](=[O:4])[O:12][H:27])([C:5]([C:6]([C:7]([N:8]([C:9]([N:10]([H:23])[H:24])=[N:11][H:25])[H:22])([H:20])[H:21])([H:18])[H:19])([H:16])[H:17])[H:15])([H:13])[H:14]"
#     ],
#     "CYS": [
#         "[N:1]([C@:2]([C:3](=[O:4])[O:7][H:14])([C:5]([S-:6])([H:11])[H:12])[H:10])([H:8])[H:9]"
#     ],
#     "TYR": [
#         "[N:1]([C@:2]([C:3](=[O:4])[O:13][H:24])([C:5]([c:6]1[c:7]([H:19])[c:9]([H:21])[c:11]([O-:12])[c:10]([H:22])[c:8]1[H:20])([H:17])[H:18])[H:16])([H:14])[H:15]"
#     ],
# }


# def add_protonation_variants(res: ResidueDefinition) -> list[ResidueDefinition]:
#     residue_definitions = [res]

#     for variant_smiles in PROTONATION_VARIANTS[res.residueName]:
#         variant = Molecule.from_smiles(variant_smiles, allow_undefined_stereo=True)
#         mappings = variant.properties["atom_map"]
#         atoms = []
#         for i, variant_atom in enumerate(variant.atoms):
#             # TODO: Handle case where extra atom is added in variant
#             db_atom = res.atoms[mappings[i] - 1]
#             new_atom = AtomDefinition(
#                 name=db_atom.name,
#                 synonyms=db_atom.synonyms,
#                 symbol=variant_atom.symbol,
#                 leaving=db_atom.leaving,
#                 x=db_atom.x,
#                 y=db_atom.y,
#                 z=db_atom.z,
#                 charge=variant_atom.formal_charge,
#                 aromatic=variant_atom.is_aromatic,
#                 stereo=variant_atom.stereochemistry,
#             )
#             atoms.append(new_atom)
#         bonds = []
#         for variant_bond in variant.bonds:
#             new_bond = BondDefinition(
#                 atom1=atoms[variant_bond.atom1_index].name,
#                 atom2=atoms[variant_bond.atom2_index].name,
#                 order={1: "SING", 2: "DOUB", 3: "TRIP"}[variant_bond.bond_order],
#                 aromatic=variant_bond.is_aromatic,
#                 stereo=variant_bond.stereochemistry,
#             )
#             bonds.append(new_bond)
#         residue_definitions.append(
#             ResidueDefinition(
#                 residueName=res.residueName,
#                 smiles=[variant_smiles],
#                 linking_type=res.linking_type,
#                 atoms=atoms,
#                 bonds=bonds,
#             )
#         )

#     return residue_definitions


ATOM_NAME_SYNONYMS = {
    "NME": {"HN2": ["H"]},
    "NA": {"NA": ["Na"]},
    "CL": {"CL": ["Cl"]},
}


def add_synonyms(res: ResidueDefinition) -> list[ResidueDefinition]:
    for atom in res.atoms:
        atom.synonyms.extend(ATOM_NAME_SYNONYMS[res.residueName].get(atom.name, []))
    return [res]


def disambiguate_alt_ids(res: ResidueDefinition) -> list[ResidueDefinition]:
    """
    CCD patch: put alt atom ids in their own residue definitions if needed

    This patch should be run before other patches that add synonyms, as it
    assumes that there is at most one synonym that came from the CCD alt id
    flag.

    Some CCD residues (like GLY) have alternative atom IDs that clash with
    canonical IDs for a different atom. This breaks synonyms because the
    clashing alternate ID is never assigned; the PDB file is interpreted as
    having two copies of the canonical ID atom. To fix this, we just split
    residue definitions with this clashing problem into two definitions, one
    with the canonical IDs and the other with the alternates.
    """
    clashes = []
    canonical_names = {atom.name for atom in res.atoms}
    for i, atom in enumerate(res.atoms):
        for synonym in atom.synonyms:
            if synonym in canonical_names:
                clashes.append(i)

    if clashes:
        res2 = deepcopy(res)
        old_to_new = {}
        for atom in res2.atoms:
            old_to_new[atom.name] = atom.name
            if atom.synonyms:
                synonym = unwrap(atom.synonyms)
                old_to_new[atom.name] = synonym
                atom.name = synonym
                atom.synonyms = []
        for bond in res2.bonds:
            bond.atom1 = old_to_new[bond.atom1]
            bond.atom2 = old_to_new[bond.atom2]
        for clash in clashes:
            res.atoms[clash].synonyms = []
        return [res, res2]
    else:
        return [res]


def combine_patches(
    *patches: dict[str, Callable[[ResidueDefinition], list[ResidueDefinition]]],
) -> dict[str, Callable[[ResidueDefinition], list[ResidueDefinition]]]:
    combined = {}
    for patch in patches:
        for key, fn in patch.items():
            if key in combined:
                existing_fn = combined[key]
                combined[key] = lambda x: list(flatten(fn(y) for y in existing_fn(x)))
            else:
                combined[key] = fn
    return combined


# TODO: Replace these patches with CONECT records?
CCD_RESIDUE_DEFINITION_CACHE = CcdCache(
    Path(__file__).parent / "../../.ccd_cache",
    patches=combine_patches(
        {"*": disambiguate_alt_ids},
        {
            "ACE": fix_caps,
            "NME": fix_caps,
        },
        # {key: add_protonation_variants for key in PROTONATION_VARIANTS},
        {key: add_synonyms for key in ATOM_NAME_SYNONYMS},
    ),
)


# TODO: Fill in this data
PEPTIDE_BOND = BondDefinition(
    atom1="C", atom2="N", order="SING", aromatic=False, stereo=None
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


def _load_unknown_residue(
    data: PdbData, indices: list[int], unknown_molecules: list[Molecule]
) -> PDBMolecule:
    atoms = []
    conects = set()
    serial_to_index = {}
    for i, pdb_index in enumerate(indices):
        new_atom = PDBAtom(
            atomic_number=elements.NUMBERS[data.element[pdb_index]],
            formal_charge=data.charge[pdb_index],
            is_aromatic=None,
            stereochemistry=None,
            name=data.name[pdb_index],
            x=data.x[pdb_index],
            y=data.y[pdb_index],
            z=data.z[pdb_index],
            metadata={
                "residue_name": data.res_name[pdb_index],
                "leaving": False,
                "pdb_index": pdb_index,
                "residue_number": str(data.res_seq[pdb_index]),
                "res_seq": data.res_seq[pdb_index],
                "insertion_code": data.i_code[pdb_index],
                "chain_id": data.chain_id[pdb_index],
                "atom_serial": data.serial[pdb_index],
            },
        )
        atoms.append(new_atom)
        serial_to_index[data.serial[pdb_index]] = i
        for conect_serial in data.conects[pdb_index]:
            conects.add(tuple(sorted([data.serial[pdb_index], conect_serial])))

    bonds = []
    for serial1, serial2 in conects:
        bonds.append(
            PDBBond(
                atom1=serial_to_index[serial1],
                atom2=serial_to_index[serial2],
            )
        )

    pdbmol = PDBMolecule(
        atoms=atoms,
        bonds=bonds,
        properties={"linking_type": "NON-POLYMER"},
    )

    for molecule in unknown_molecules:
        (match_found, mapping) = Molecule.are_isomorphic(
            pdbmol.to_networkx(),
            molecule.to_networkx(),
            return_atom_map=True,
            aromatic_matching=False,
            formal_charge_matching=False,
            bond_order_matching=False,
            atom_stereochemistry_matching=False,
            bond_stereochemistry_matching=False,
            strip_pyrimidal_n_atom_stereo=True,
        )
        assert mapping is not None
        if match_found:
            reverse_map = {}
            for i, atom in enumerate(pdbmol.atoms):
                reverse_map[mapping[i]] = i
                reference_atom = molecule.atom(mapping[i])
                atom.formal_charge = reference_atom.formal_charge
                atom.is_aromatic = reference_atom.is_aromatic
                atom.stereochemistry = reference_atom.stereochemistry
            pdbmol.bonds.clear()
            for bond in molecule.bonds:
                pdbmol.add_bond(
                    PDBBond(
                        atom1=reverse_map[bond.atom1_index],
                        atom2=reverse_map[bond.atom2_index],
                        bond_order=bond.bond_order,
                        is_aromatic=bond.is_aromatic,
                        stereochemistry=bond.stereochemistry,
                    )
                )
            break
    else:
        res_name = pdbmol.atoms[0].metadata["residue_name"]
        res_seq = pdbmol.atoms[0].metadata["res_seq"]
        chain_id = pdbmol.atoms[0].metadata["chain_id"]
        raise ValueError(
            f"Unknown residue {chain_id}:{res_name}#{res_seq} could not be assigned chemistry from unknown_molecules"
        )

    return pdbmol


def _load_residue_from_database(
    data: PdbData,
    res_atom_idcs: list[int],
    residue_database: Mapping[str, list[ResidueDefinition]],
) -> PDBMolecule:
    prototype_index = res_atom_idcs[0]

    res_name = data.res_name[prototype_index]
    res_seq = data.res_seq[prototype_index]
    chain_id = data.chain_id[prototype_index]
    i_code = data.i_code[prototype_index]
    residue_wide_metadata = {
        "residue_number": str(res_seq),
        "res_seq": res_seq,
        "insertion_code": i_code,
        "chain_id": chain_id,
    }

    residues: list[PDBMolecule] = []
    unmatched_atoms = []
    for residue_definition in residue_database[res_name]:
        residue = residue_definition.to_pdb_molecule()
        print(f"Assessing {residue_definition.description} {residue.to_smiles()}")
        canonical_name_to_index = {
            residue_definition.get_canonical_name(data.name[i]): i
            for i in res_atom_idcs
        }

        res_def_atom_names = {atom.name for atom in residue.atoms}

        unmatched_atoms.append([])
        for name, index in canonical_name_to_index.items():
            if name in res_def_atom_names:
                continue
            else:
                unmatched_atoms[-1].append((name, index))
        if len(unmatched_atoms[-1]) > 0:
            # This residue definition doesn't cover all the atoms in the PDB file
            # So skip it and try the next one
            print(f"{unmatched_atoms[-1]=}")
            continue

        for atom in residue.atoms:
            if atom.name in canonical_name_to_index:
                pdb_index = canonical_name_to_index[atom.name]
                atom.x = data.x[pdb_index]
                atom.y = data.y[pdb_index]
                atom.z = data.z[pdb_index]
                atom_specific_metadata = {
                    "pdb_index": pdb_index,
                    "used_synonym": data.name[pdb_index],
                    "canonical_name": atom.name,
                    "atom_serial": data.serial[pdb_index],
                    "matched_residue_description": residue_definition.description,
                }
            else:
                atom_specific_metadata = {}

            atom.metadata.update(residue_wide_metadata | atom_specific_metadata)
        residues.append(residue)

    if len(residues) == 0:
        raise NoMatchingResidueDefinitionError(
            res_name,
            res_seq,
            chain_id,
            i_code,
            unmatched_atoms,
            residue_database[res_name],
        )

    # Choose the residue with the most atoms found in the PDB
    def key(residue: PDBMolecule) -> tuple[int, int, int]:
        coverage = 0
        leavers = 0
        for atom in residue.atoms:
            coverage += "pdb_index" in atom.metadata
            leavers += int(atom.metadata.get("leaving", False))
        excess_atoms = residue.n_atoms - coverage - leavers
        return (-coverage, excess_atoms, -leavers)

    sorted_residues = sorted(
        residues,
        key=key,
    )

    for residue in sorted_residues:
        print(
            [
                atom.metadata["res_seq"]
                for atom in residue.atoms
                if "res_seq" in atom.metadata
            ][0],
            residue.to_openff_molecule().to_smiles(),
            [
                atom.metadata["matched_residue_description"]
                for atom in residue.atoms
                if "matched_residue_description" in atom.metadata
            ][0],
            key(residue),
        )

    residue = sorted_residues[0]

    print(
        "chose",
        [
            atom.metadata["matched_residue_description"]
            for atom in residue.atoms
            if "matched_residue_description" in atom.metadata
        ][0],
    )

    residue.sort_atoms_by_metadata("pdb_index")

    return residue


def cryst_to_box_vectors(
    a: float, b: float, c: float, alpha: float, beta: float, gamma: float
) -> Quantity:
    from openmm.app.internal.unitcell import computePeriodicBoxVectors
    from openmm.unit import angstrom, degrees, nanometer

    box_vectors = computePeriodicBoxVectors(
        a * angstrom,
        b * angstrom,
        c * angstrom,
        alpha * degrees,
        beta * degrees,
        gamma * degrees,
    )
    return box_vectors.value_in_unit(nanometer) * unit.nanometer


def topology_from_pdb(
    path: PathLike,
    replace_missing_atoms: bool = False,
    use_canonical_names: bool = False,
    unknown_molecules: list[Molecule] = [],
    residue_database: Mapping[
        str, list[ResidueDefinition]
    ] = CCD_RESIDUE_DEFINITION_CACHE,
) -> Topology:
    path = Path(path)
    data = PdbData.parse_pdb(path.read_text().splitlines())

    molecules: list[PDBMolecule] = []
    current_molecule = PDBMolecule()
    prev_chain_id = __UNSET__
    prev_model = __UNSET__
    for res_atom_idcs in data.residues():
        prototype_index = res_atom_idcs[0]
        res_name = data.res_name[prototype_index]

        if prev_chain_id is __UNSET__:
            prev_chain_id = data.chain_id[prototype_index]
        if prev_model is __UNSET__:
            prev_model = data.model[prototype_index]

        # TODO: UNK is for unknown peptide residues, but we just treat it as a ligand
        # Note that for the CCD_RESIDUE_DEFINITION_CACHE, "not in" includes a check
        # for the residue names "UNL" and "UNK"
        if res_name not in residue_database:
            residue = _load_unknown_residue(data, res_atom_idcs, unknown_molecules)
        else:
            residue = _load_residue_from_database(data, res_atom_idcs, residue_database)

        if (
            residue.properties["linking_type"] == "NON-POLYMER"
            and not current_molecule.is_empty()
        ):
            molecules.append(current_molecule)
            current_molecule = PDBMolecule()
        # TODO: Ensure atoms are never re-ordered
        current_molecule.combine_with(residue)

        if (
            data.terminated[prototype_index]
            or data.chain_id[prototype_index] != prev_chain_id
            or data.model[prototype_index] != prev_model
            or current_molecule.properties["linking_type"] == "NON-POLYMER"
        ):
            molecules.append(current_molecule)
            current_molecule = PDBMolecule()

        # TODO: Load other data from PDB file
        # TODO: Incorporate CONECT records
        # TODO: Deal with multi-model files

        prev_chain_id = data.chain_id[prototype_index]
        prev_model = data.model[prototype_index]

    if not replace_missing_atoms:
        missing_atoms = []
        for molecule in molecules:
            for i, atom in enumerate(molecule.atoms):
                if "pdb_index" not in atom.metadata:
                    missing_atoms.append(atom)
        if missing_atoms:
            raise MissingAtomsFromPDBError(missing_atoms)

    if not use_canonical_names:
        for molecule in molecules:
            for atom in molecule.atoms:
                atom.name = atom.metadata.get("used_synonym", atom.name)

    topology = Topology.from_molecules(
        [pdbmol.to_openff_molecule() for pdbmol in molecules]
    )
    if (
        data.cryst1_a is not None
        and data.cryst1_b is not None
        and data.cryst1_c is not None
        and data.cryst1_alpha is not None
        and data.cryst1_beta is not None
        and data.cryst1_gamma is not None
    ):
        topology.box_vectors = cryst_to_box_vectors(
            data.cryst1_a,
            data.cryst1_b,
            data.cryst1_c,
            data.cryst1_alpha,
            data.cryst1_beta,
            data.cryst1_gamma,
        )
    for offmol in topology.molecules:
        offmol.add_default_hierarchy_schemes()

    return topology


class MissingAtomsFromPDBError(ValueError):
    def __init__(self, missing_atoms: list[PDBAtom]) -> None:
        message = [
            "The following atoms from the residue database were missing from the PDB file:"
        ]
        for atom in missing_atoms:
            chain = atom.metadata["chain_id"]
            res_name = atom.metadata["residue_name"]
            icode = atom.metadata["insertion_code"]
            res_seq = atom.metadata["res_seq"]

            residue = f"{res_name}#{res_seq:0>4}{icode}"
            if chain:
                residue = f"{chain}:" + residue
            message.append(f"    Atom {atom.name} in residue {residue}")

        self.missing_atoms = missing_atoms
        super().__init__("\n".join(message))


class NoMatchingResidueDefinitionError(ValueError):
    def __init__(
        self,
        res_name: str,
        res_seq: int,
        chain_id: str,
        i_code: str,
        unmatched_atoms: list[list[str]],
        residue_definitions: list[ResidueDefinition],
    ):
        message = [
            f"No residue definitions covered all atoms in {res_name}#{res_seq}{i_code.strip()}:{chain_id}",
            "The following definitions were considered:",
        ]
        assert len(unmatched_atoms) == len(residue_definitions)
        for i, (unmatched_names, residue) in enumerate(
            zip(unmatched_atoms, residue_definitions)
        ):
            defined_canonical_names = {atom.name for atom in residue.atoms}
            expected_names = residue.format_names_with_synonyms(
                res_name, defined_canonical_names
            )
            message.append(
                f"    {i}: {unmatched_names} were not among {expected_names}"
            )

        self.unmatched_atoms = unmatched_atoms
        super().__init__("\n".join(message))
