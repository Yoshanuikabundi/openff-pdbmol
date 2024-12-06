"""
Tools for reading and patching the PDB Chemical Component Dictionary (CCD).
"""

import dataclasses
import gzip
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Callable, Iterator, Mapping
from urllib.request import urlopen

from openmm.app.internal.pdbx.reader.PdbxReader import PdbxReader

from ._utils import flatten, unwrap
from .residue import AtomDefinition, BondDefinition, ResidueDefinition

__all__ = [
    "CcdCache",
    "fix_caps",
    "ATOM_NAME_SYNONYMS",
    "add_synonyms",
    "disambiguate_alt_ids",
    "combine_patches",
    "PEPTIDE_BOND",
    "LINKING_TYPES",
]


class CcdCache(Mapping[str, list[ResidueDefinition]]):
    """
    Caches, patches, and presents the CCD as a Python ``Mapping``.

    This requires internet access to work.

    Parameters
    ==========
    path
        The path to which to download CCD entries.
    preload
        A list of residue names to download when initializing the class.
    patches
        Functions to call on the given ``ResidueDefinitions`` before they are
        returned. A map from residue names to a single callable. The patch
        corresponding to key ``"*"`` will be applied to all residues before the
        more specific patches. Use :py:func:`combine_patches` to combine
        multiple patches into one.
    """

    # TODO: Methods for adding entries from mapped SMILES

    def __init__(
        self,
        path: Path,
        preload: list[str] = [],
        patches: Mapping[
            str, Callable[[ResidueDefinition], list[ResidueDefinition]]
        ] = {},
    ):
        self._path = path.resolve()
        self._path.mkdir(parents=True, exist_ok=True)

        self._definitions: dict[str, list[ResidueDefinition]] = {}
        self._patches: dict[
            str, Callable[[ResidueDefinition], list[ResidueDefinition]]
        ] = dict(patches)

        self._load_protonation_variants()

        for file in path.glob("*.cif"):
            self._add_definition_from_str(file.read_text())

        for resname in set(preload) - set(self._definitions):
            self[resname]

    def __repr__(self):
        return (
            f"CcdCache(path={self._path},"
            + f" preload={list(self._definitions)},"
            + f" patches={self._patches!r})"
        )

    def __getitem__(self, key: str) -> list[ResidueDefinition]:
        res_name = key.upper()
        if res_name in ["UNK", "UNL"]:
            # These residue names are reserved for unknown ligands/peptide residues
            raise KeyError(res_name)
        if res_name not in self._definitions:
            try:
                s = (self._path / f"{res_name.upper()}.cif").read_text()
            except Exception:
                s = self._download_cif(res_name)

            self._add_definition_from_str(s, res_name=res_name)
        return self._definitions[res_name]

    def _apply_patches(self, definition: ResidueDefinition) -> list[ResidueDefinition]:
        # Get the patches for this definition, or the identity function if absent
        patch_res = self._patches.get(definition.residue_name.upper(), lambda x: [x])
        patch_all = self._patches.get("*", lambda x: [x])
        # Apply the patches and flatten as appropriate
        return list(flatten(patch_res(res) for res in patch_all(definition)))

    def _add_definition_from_str(self, s: str, res_name: str | None = None) -> None:
        definition = self._res_def_from_ccd_str(s)
        if res_name is None:
            res_name = definition.residue_name.upper()

        patched_definitions = self._apply_patches(definition)

        assert all(
            res_name == definition.residue_name.upper()
            for definition in patched_definitions
        )

        self._definitions.setdefault(res_name, []).extend(patched_definitions)

    def _load_protonation_variants(self):
        path = self._path / "aa-variants-v1.cif.gz"
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
        path = self._path / f"{resname.upper()}.cif"
        path.write_text(s)
        return s

    @staticmethod
    def _res_def_from_ccd_str(s: str) -> ResidueDefinition:
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
        linking_bond = LINKING_TYPES[linking_type]

        atomData = block.getObj("chem_comp_atom")
        atomNameCol = atomData.getAttributeIndex("atom_id")
        altAtomNameCol = atomData.getAttributeIndex("alt_atom_id")
        symbolCol = atomData.getAttributeIndex("type_symbol")
        leavingCol = atomData.getAttributeIndex("pdbx_leaving_atom_flag")
        chargeCol = atomData.getAttributeIndex("charge")
        aromaticCol = atomData.getAttributeIndex("pdbx_aromatic_flag")
        stereoCol = atomData.getAttributeIndex("pdbx_stereo_config")

        atoms = [
            AtomDefinition(
                name=row[atomNameCol],
                synonyms=set(
                    [row[altAtomNameCol]]
                    if row[altAtomNameCol] != row[atomNameCol]
                    else []
                ),
                symbol=row[symbolCol][0:1].upper() + row[symbolCol][1:].lower(),
                leaving=row[leavingCol] == "Y",
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
                    order={"SING": 1, "DOUB": 2, "TRIP": 3, "QUAD": 4}[row[orderCol]],
                    aromatic=row[aromaticCol] == "Y",
                    stereo=None if row[stereoCol] == "N" else row[stereoCol],
                )
                for row in bondData.getRowList()
            ]
        else:
            bonds = []

        return ResidueDefinition(
            residue_name=residueName,
            description=residue_description,
            linking_bond=linking_bond,
            atoms=tuple(atoms),
            bonds=tuple(bonds),
        )

    def __contains__(self, value) -> bool:
        if value in self._definitions:
            return True

        try:
            self[value]
        except Exception:
            # This catches KeyError, but also failures to download the residue
            return False
        else:
            return True

    def __iter__(self) -> Iterator[str]:
        return self._definitions.__iter__()

    def __len__(self) -> int:
        return self._definitions.__len__()


def fix_caps(res: ResidueDefinition) -> list[ResidueDefinition]:
    """
    Fix ``"NON-POLYMER"`` residues so they can be used as caps for peptides.
    """

    return [
        dataclasses.replace(
            res,
            linking_bond=PEPTIDE_BOND,
            atoms=[
                dataclasses.replace(
                    atom,
                    leaving=True if atom.name == "H" else atom.leaving,
                )
                for atom in res.atoms
            ],
        )
    ]


ATOM_NAME_SYNONYMS = {
    "NME": {"HN2": ["H"]},
    "NA": {"NA": ["Na"]},
    "CL": {"CL": ["Cl"]},
}
"""Map from residue name and then canonical atom name to a list of synonyms"""


def add_synonyms(res: ResidueDefinition) -> list[ResidueDefinition]:
    """
    Patch a residue definition to include synonyms from :py:data:`ATOM_NAME_SYNONYMS`.
    """
    for atom in res.atoms:
        atom.synonyms.update(ATOM_NAME_SYNONYMS[res.residue_name].get(atom.name, []))
    return [res]


def disambiguate_alt_ids(res: ResidueDefinition) -> list[ResidueDefinition]:
    """
    CCD patch: put alt atom ids in their own residue definitions if needed

    This patch should be run before other patches that add synonyms, as it
    assumes that there is at most one synonym (from the CCD alt id
    flag).

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
        old_to_new = {}
        for atom in res.atoms:
            if atom.synonyms:
                old_to_new[atom.name] = unwrap(atom.synonyms)
            else:
                old_to_new[atom.name] = atom.name

        res1 = dataclasses.replace(
            res,
            atoms=[
                dataclasses.replace(
                    atom,
                    synonyms=[] if i in clashes else deepcopy(atom.synonyms),
                )
                for i, atom in enumerate(res.atoms)
            ],
        )
        res2 = dataclasses.replace(
            res,
            atoms=[
                dataclasses.replace(
                    atom,
                    name=old_to_new[atom.name],
                    synonyms=[] if atom.synonyms else deepcopy(atom.synonyms),
                )
                for atom in res.atoms
            ],
            bonds=[
                dataclasses.replace(
                    bond,
                    atom1=old_to_new[bond.atom1],
                    atom2=old_to_new[bond.atom2],
                )
                for bond in res.bonds
            ],
        )
        return [res1, res2]
    else:
        return [res]


def combine_patches(
    *patches: dict[str, Callable[[ResidueDefinition], list[ResidueDefinition]]],
) -> dict[str, Callable[[ResidueDefinition], list[ResidueDefinition]]]:
    """Combine multiple ``dict`` objects of patches into a single patchset"""
    combined = {}
    for patch in patches:
        for key, fn in patch.items():
            if key in combined:
                existing_fn = combined[key]
                combined[key] = lambda x: list(flatten(fn(y) for y in existing_fn(x)))
            else:
                combined[key] = fn
    return combined


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

# TODO: Replace these patches with CONECT records?
CCD_RESIDUE_DEFINITION_CACHE = CcdCache(
    Path(__file__).parent / "../../.ccd_cache",
    patches=combine_patches(
        {"*": disambiguate_alt_ids},
        {
            "ACE": fix_caps,
            "NME": fix_caps,
        },
        {key: add_synonyms for key in ATOM_NAME_SYNONYMS},
    ),
)
"""The CCD, with commonly-required patches"""
