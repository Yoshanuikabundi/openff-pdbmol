from collections.abc import Mapping
from io import StringIO
from itertools import chain
from os import PathLike
from time import sleep
from typing import Any, Iterable, Self, Literal, Iterator, TypeVar
from pathlib import Path
from dataclasses import dataclass, field
import dataclasses
from urllib.request import urlopen

from openff.toolkit import Molecule, Topology
from openff.toolkit.topology import Atom, Bond
from openff.units import elements, unit
from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader


class __UNSET__:
    pass


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


def identify_linkers(
    molecule: Molecule, linked_atomname: str
) -> tuple[int, Atom, set[Atom]]:
    possible_partners = [
        (i, atom)
        for i, atom in enumerate(molecule.atoms)
        if atom.name == linked_atomname
    ]
    for partner, partner_atom in possible_partners:
        leavers = set()
        candidates = set(partner_atom.bonded_atoms)
        while candidates:
            candidate = candidates.pop()
            if candidate.metadata.get("leaving", False):
                leavers.add(candidate)
                candidates.update(set(candidate.bonded_atoms) - leavers)
        if leavers:
            return (partner, partner_atom, leavers)
    raise ValueError("No partners found")


def combine_molecules(this: Molecule, other: Molecule):
    """
    Combine molecules by unifying leaving atoms with the opposite molecule

    Preserves leaving annotations in ``other`` but not in ``this``.
    """
    # If this is empty, short circuit
    if this.n_atoms == 0:
        return Molecule(other)

    # Identify the bond linking the two molecules
    this_linking_type = this.properties["linking_type"]
    other_linking_type = this.properties["linking_type"]

    if this_linking_type != other_linking_type:
        raise ValueError(
            f"Molecule of linking type {this_linking_type} cannot be linked to"
            + f" molecule of linking type {other_linking_type}"
        )
    if this_linking_type not in LINKING_TYPES:
        raise ValueError(f"Unknown linking type {this_linking_type}")
    if other_linking_type not in LINKING_TYPES:
        raise ValueError(f"Unknown linking type {other_linking_type}")
    if LINKING_TYPES[this_linking_type] is None:
        raise ValueError(f"Linking type {this_linking_type} does not form linkages")

    linking_bond = LINKING_TYPES[other_linking_type]
    if linking_bond is None:
        raise ValueError(f"Linking type {other_linking_type} does not form linkages")

    # Identify the atoms participating in the bond and those leaving
    this_partner, this_partner_atom, this_leavers = identify_linkers(
        this, linking_bond.atom1
    )
    other_partner, other_partner_atom, other_leavers = identify_linkers(
        other, linking_bond.atom2
    )

    # Add atoms
    combined = Molecule()
    combined.properties.update(this.properties | other.properties)
    # TODO: Preserve conformers and other data
    this_to_combined = {}
    for i, atom in enumerate(this.atoms):
        if atom in this_leavers:
            continue

        this_to_combined[i] = combined._add_atom(
            atomic_number=atom.atomic_number,
            formal_charge=atom.formal_charge,
            is_aromatic=atom.is_aromatic,
            stereochemistry=atom.stereochemistry,
            name=atom.name,
            metadata=atom.metadata | {"leaving": False},
            invalidate_cache=False,
        )

    other_to_combined = {}
    for i, atom in enumerate(other.atoms):
        if atom in other_leavers:
            continue

        other_to_combined[i] = combined._add_atom(
            atomic_number=atom.atomic_number,
            formal_charge=atom.formal_charge,
            is_aromatic=atom.is_aromatic,
            stereochemistry=atom.stereochemistry,
            name=atom.name,
            metadata=atom.metadata,
            invalidate_cache=False,
        )

    # Add bonds
    for bond in this.bonds:
        if bond.atom1 in this_leavers or bond.atom2 in this_leavers:
            continue

        combined._add_bond(
            atom1=this_to_combined[bond.atom1_index],
            atom2=this_to_combined[bond.atom2_index],
            bond_order=bond.bond_order,
            is_aromatic=bond.is_aromatic,
            stereochemistry=bond.stereochemistry,
            invalidate_cache=False,
        )

    for bond in other.bonds:
        if bond.atom1 in other_leavers or bond.atom2 in other_leavers:
            continue

        combined._add_bond(
            atom1=other_to_combined[bond.atom1_index],
            atom2=other_to_combined[bond.atom2_index],
            bond_order=bond.bond_order,
            is_aromatic=bond.is_aromatic,
            stereochemistry=bond.stereochemistry,
            invalidate_cache=False,
        )

    combined.add_bond(
        atom1=this_to_combined[this_partner],
        atom2=other_to_combined[other_partner],
        bond_order={"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}[linking_bond.order],
        is_aromatic=linking_bond.aromatic,
        stereochemistry=linking_bond.stereo,
    )

    return combined


def topology_from_pdb(path: PathLike) -> Topology:
    path = Path(path)
    data = PdbData.parse_pdb(path.read_text().splitlines())

    molecules = []
    current_molecule = Molecule()
    prev_chain_id = __UNSET__
    prev_model = __UNSET__
    for res_atom_idcs in data.residues():
        prototype_index = res_atom_idcs[0]

        if prev_chain_id is __UNSET__:
            prev_chain_id = data.chain_id[prototype_index]
        if prev_model is __UNSET__:
            prev_model = data.model[prototype_index]

        res_name = data.res_name[prototype_index]
        residue = CCD_RESIDUE_DEFINITION_CACHE.get_resname(res_name).to_molecule()

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
                x = data.x[pdb_index]
                y = data.y[pdb_index]
                z = data.z[pdb_index]
                atom_specific_metadata = {
                    "pdb_index": pdb_index,
                    "pdb_coords": f"{x} {y} {z}",
                }
            else:
                atom_specific_metadata = {}

            atom.metadata.update(residue_wide_metadata | atom_specific_metadata)

        current_molecule = combine_molecules(current_molecule, residue)

        if (
            data.terminated[prototype_index]
            or data.chain_id[prototype_index] != prev_chain_id
            or data.model[prototype_index] != prev_model
            or current_molecule.properties["linking_type"] == "NON-POLYMER"
        ):
            molecules.append(current_molecule)
            current_molecule = Molecule()

        # TODO: Load other data from PDB file
        # TODO: Incorporate CONECT records
        # TODO: Deal with multi-model files

        prev_chain_id = data.chain_id[prototype_index]
        prev_model = data.model[prototype_index]

    topology = Topology.from_molecules(molecules)

    positions = []
    for atom in topology.atoms:
        if "pdb_coords" in atom.metadata:
            coords = atom.metadata["pdb_coords"]
        else:
            # TODO: Generate coordinates for atoms missing from PDB file
            coords = next(
                (
                    atom.metadata["pdb_coords"]
                    for atom in atom.bonded_atoms
                    if "pdb_coords" in atom.metadata
                ),
                "0 0 0",
            )
        positions.append([float(s) for s in coords.split()])
    topology.set_positions(positions * unit.angstrom)

    return topology
