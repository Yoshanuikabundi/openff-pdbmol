from collections.abc import Mapping
from copy import deepcopy
from io import StringIO
from itertools import chain
from os import PathLike
from time import sleep
from typing import Any, Generator, Iterable, Self, Literal, Iterator, TypeVar
from pathlib import Path
from dataclasses import dataclass, field
import dataclasses
from urllib.request import urlopen

from openff.toolkit import Molecule, Topology
from openff.toolkit.topology import Atom, Bond
from openff.units import elements, unit
from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader
import numpy as np


class __UNSET__:
    pass


@dataclass
class PDBAtom:
    atomic_number: int
    formal_charge: int
    is_aromatic: bool
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

    def is_empty(self) -> bool:
        return len(self.atoms) == 0 and len(self.bonds) == 0 and len(self.properties) == 0

    @property
    def n_atoms(self) -> int:
        return len(self.atoms)

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    def identify_linkers(
        self, linked_atomname: str
    ) -> tuple[int, PDBAtom, set[int]]:
        possible_partners = [
            (i, atom)
            for i, atom in enumerate(self.atoms)
            if atom.name == linked_atomname
        ]
        for partner, partner_atom in possible_partners:
            leavers = set()
            candidates = set(self.atoms_bonded_to(partner))
            while candidates:
                candidate = candidates.pop()
                candidate_atom = self.atoms[candidate]
                if candidate_atom.metadata.get("leaving", False):
                    leavers.add(candidate)
                    candidates.update(set(self.atoms_bonded_to(candidate)) - leavers)
            if leavers:
                return (partner, partner_atom, leavers)
        raise ValueError("No partners found")

    def to_openff_molecule(self) -> Molecule:
        molecule = Molecule()
        molecule.properties.update(self.properties)
        conformer = []

        for atom in self.atoms:
            molecule._add_atom(
                atomic_number = atom.atomic_number,
                formal_charge= atom.formal_charge,
                is_aromatic= atom.is_aromatic,
                stereochemistry=atom.stereochemistry,
                name = atom.name,
                metadata = atom.metadata,
                invalidate_cache=False
            )

            if atom.coords is None:
                # TODO: Come up with something clever here
                conformer.append((0., 0., 0.))
            else:
                conformer.append(atom.coords)

        for bond in self.bonds:
            molecule._add_bond(
                atom1=bond.atom1,
                atom2=bond.atom2,
                bond_order = bond.bond_order,
                is_aromatic=bond.is_aromatic,
                stereochemistry=bond.stereochemistry,
                invalidate_cache=False
            )

        molecule._invalidate_cached_properties()

        molecule.add_conformer(np.asarray(conformer) * unit.angstrom)
        return molecule

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
            raise ValueError(f"Linking type {self_linking_type} does not form linkages")
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


        self.add_bond(PDBBond(
            atom1=self_to_combined[this_partner],
            atom2=other_to_combined[other_partner],
            bond_order={"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}[self_linking_bond.order],
            is_aromatic=self_linking_bond.aromatic,
            stereochemistry=self_linking_bond.stereo,
        ))

        for bond in other.bonds:
            if bond.atom1 in other_leavers or bond.atom2 in other_leavers:
                continue

            self.add_bond(PDBBond(
                atom1=other_to_combined[bond.atom1],
                atom2=other_to_combined[bond.atom2],
                bond_order=bond.bond_order,
                is_aromatic=bond.is_aromatic,
                stereochemistry=bond.stereochemistry,
            ))


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


@dataclass
class CcdAtomDefinition:
    """
    Description of an atom in a residue from the Chemical Component Dictionary (CCD).
    """

    name: str
    symbol: str
    leaving: bool
    x: float
    y: float
    z: float
    charge: int
    aromatic: bool
    stereo: Literal["S", "R"] | None


@dataclass
class CcdBondDefinition:
    """
    Description of a bond in a residue from the Chemical Component Dictionary (CCD).
    """

    atom1: str
    atom2: str
    order: Literal["SING", "DOUB", "TRIP", "QUAD", "AROM", "DELO", "PI", "POLY"]
    aromatic: bool
    stereo: Literal["E", "Z"] | None


@dataclass
class CcdResidueDefinition:
    """
    Description of a residue from the Chemical Component Dictionary (CCD).
    """

    residueName: str
    smiles: list[str]
    linking_type: str
    atoms: list[CcdAtomDefinition]
    bonds: list[CcdBondDefinition]

    @classmethod
    def from_str(cls, s) -> Self:
        data = []
        with StringIO(s) as file:
            PdbxReader(file).read(data)
        block = data[0]

        residueName = block.getObj("chem_comp").getValue("id").upper()
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
        symbolCol = atomData.getAttributeIndex("type_symbol")
        leavingCol = atomData.getAttributeIndex("pdbx_leaving_atom_flag")
        xCol = atomData.getAttributeIndex("pdbx_model_Cartn_x_ideal")
        yCol = atomData.getAttributeIndex("pdbx_model_Cartn_y_ideal")
        zCol = atomData.getAttributeIndex("pdbx_model_Cartn_z_ideal")
        chargeCol = atomData.getAttributeIndex("charge")
        aromaticCol = atomData.getAttributeIndex("pdbx_aromatic_flag")
        stereoCol = atomData.getAttributeIndex("pdbx_stereo_config")

        atoms = [
            CcdAtomDefinition(
                name=row[atomNameCol],
                symbol=row[symbolCol][0:1].upper() + row[symbolCol][1:].lower(),
                leaving=row[leavingCol] == "Y",
                x=float(row[xCol]),
                y=float(row[yCol]),
                z=float(row[zCol]),
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
                CcdBondDefinition(
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
            linking_type=linking_type,
            atoms=atoms,
            bonds=bonds,
        )
        return ret

    def to_molecule(self) -> Molecule:
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

    def _append_coord_line(self, line: str):
        for field in dataclasses.fields(self):
            getattr(self, field.name).append(__UNSET__)
            assert getattr(self, field.name)[-1] is __UNSET__

        self.model[-1] = None
        self.serial[-1] = int(line[6:11])
        self.name[-1] = line[12:16].strip()
        self.alt_loc[-1] = line[16].strip() or ""
        self.res_name[-1] = line[17:20].strip()
        self.chain_id[-1] = line[21].strip()
        self.res_seq[-1] = int(line[22:26])
        self.i_code[-1] = line[26].strip() or ""
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
            assert getattr(self, field.name)[-1] is not __UNSET__

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
                terminated_resseq = int(line[22:26]) or data.res_seq[-1]
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

        return data

    def residues(self):
        indices = []
        prev = None
        for i, atom in enumerate(
            zip(self.model, self.res_name, self.chain_id, self.res_seq, self.i_code)
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


class CcdCache(dict[str, CcdResidueDefinition]):
    def __init__(self, path: Path, preload: list[str] = []):
        self.path = path.resolve()
        self.path.mkdir(parents=True, exist_ok=True)

        for file in path.glob("*.cif"):
            definition = CcdResidueDefinition.from_str(file.read_text())
            self[definition.residueName.upper()] = definition

        for resname in set(preload) - set(self):
            self.get_resname(resname)

    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return "%s(%s)" % (type(self).__name__, dictrepr)

    def __getitem__(self, key):
        return super().__getitem__(key.upper())

    def __setitem__(self, key, val):
        return super().__setitem__(key.upper(), val)

    def update(self, *args, **kwargs):
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def _download_cif(self, resname: str) -> str:
        with urlopen(
            f"https://files.rcsb.org/ligands/download/{resname.upper()}.cif",
        ) as stream:
            s = stream.read().decode("utf-8")
        path = self.path / f"{resname.upper()}.cif"
        path.write_text(s)
        return s

    def get_resname(self, resname: str) -> CcdResidueDefinition:
        """
        Create a CCDResidueDefinition by parsing a CCD CIF file.
        """
        resname = resname.upper()
        if resname in self:
            return self[resname]

        s = self._download_cif(resname)
        self[resname] = CcdResidueDefinition.from_str(s)
        return self[resname]


CCD_RESIDUE_DEFINITION_CACHE = CcdCache(Path(__file__).parent / "../../.ccd_cache")

# TODO: Fill in this data
LINKING_TYPES: dict[str, CcdBondDefinition | None] = {
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
    "L-peptide linking".upper(): CcdBondDefinition(
        atom1="C", atom2="N", order="SING", aromatic=False, stereo=None
    ),
    # "L-saccharide".upper(): [],
    # "L-saccharide, alpha linking".upper(): [],
    # "L-saccharide, beta linking".upper(): [],
    # "RNA OH 3 prime terminus".upper(): [],
    # "RNA OH 5 prime terminus".upper(): [],
    # "RNA linking".upper(): [],
    "non-polymer".upper(): None,
    # "other".upper(): [],
    "peptide linking".upper(): CcdBondDefinition(
        atom1="C", atom2="N", order="SING", aromatic=False, stereo=None
    ),
    # "peptide-like".upper(): [],
    # "saccharide".upper(): [],
}

def topology_from_pdb(path: PathLike) -> Topology:
    path = Path(path)
    data = PdbData.parse_pdb(path.read_text().splitlines())

    molecules: list[PDBMolecule] = []
    current_molecule = PDBMolecule()
    prev_chain_id = __UNSET__
    prev_model = __UNSET__
    for res_atom_idcs in data.residues():
        prototype_index = res_atom_idcs[0]

        if prev_chain_id is __UNSET__:
            prev_chain_id = data.chain_id[prototype_index]
        if prev_model is __UNSET__:
            prev_model = data.model[prototype_index]

        res_name = data.res_name[prototype_index]
        residue = CCD_RESIDUE_DEFINITION_CACHE.get_resname(res_name).to_pdb_molecule()

        atom_names_to_indices = {data.name[i]: i for i in res_atom_idcs}
        ccd_residue_atom_names = {atom.name for atom in residue.atoms}

        residue_wide_metadata = {
            "residue_number": data.res_seq[prototype_index],
            "insertion_code": data.i_code[prototype_index],
            "chain_id": data.chain_id[prototype_index],
        }
        for atom in residue.atoms:
            if atom.name not in ccd_residue_atom_names:
                raise ValueError(
                    "Atom {atom.name} in PDB file not present in residue {res_name}"
                )

            if atom.name in atom_names_to_indices:
                pdb_index = atom_names_to_indices[atom.name]
                atom.x = data.x[pdb_index]
                atom.y = data.y[pdb_index]
                atom.z = data.z[pdb_index]
                atom_specific_metadata = {
                    "pdb_index": pdb_index,
                }
            else:
                atom_specific_metadata = {}

            atom.metadata.update(residue_wide_metadata | atom_specific_metadata)

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

    topology = Topology.from_molecules([pdbmol.to_openff_molecule() for pdbmol in molecules])

    return topology
