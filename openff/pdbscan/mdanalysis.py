from contextlib import contextmanager
from copy import deepcopy
from typing import Any

import MDAnalysis as mda
import rdkit.Chem
from MDAnalysis.guesser.tables import vdwradii as mda_vdwradii_table

from openff.toolkit import Molecule, Topology
from openff.units import unit


@contextmanager
def _patch_mda():
    """Temporarily patch MDAnalysis to support more PDB files

    This is a context manager; code ran in the provided context using the
    ``with`` keyword will be patched, while code outside the provided context
    will not. It can also decorate a function to apply the patch to all code in
    that function.
    """
    # Perform the patch
    mda.converters.RDKit.RDBONDORDER[4] = rdkit.Chem.BondType.QUADRUPLE
    mda.converters.RDKit.RDBONDORDER[5] = rdkit.Chem.BondType.QUINTUPLE

    yield

    # Undo the patch
    del mda.converters.RDKit.RDBONDORDER[4]
    del mda.converters.RDKit.RDBONDORDER[5]


def _case_insensitise_vdwradii(data: dict[str, Any]) -> dict[str, Any]:
    new_data = {}
    for key, value in data.items():
        new_data[key.lower()] = value
        new_data[key[0] + key[1:].lower()] = value
    return new_data


@_patch_mda()
def topology_from_pdb(
    pdbfile: str,
    vdwradii: dict[str, float] = {},
    overlap_factor: float = 0.55,
    shortest_bond: float = 0.1,
    use_box: bool = True,
) -> Topology:
    """Use MDAnalysis' PDB loader to load a PDB file.

    Bonds not explicitly included in CONECT records are guessed from atom-atom
    distances. The `vdw_radii`, `required_overlap`, and `lower_bound` parameters
    configure this guessing behavior.

    Parameters
    ==========

    pdbfile
        Path to the PDB file to load.
    vdwradii
        Additional van der Waals radii used to guess bonds. A dictionary mapping
        elements to radii in Angstroms that augments the table provided at
        `MDAnalysis.topology.tables.vdwradii`.
    overlap_factor
        The maximum proportion of the sum of two atom's van der Waals radii at
        which distance the two atoms are considered bonded. Larger values
        produce more bonds.
    shortest_bond
        Any bonds found that are shorter than this distance in Angstroms are
        discarded.
    use_pbcs
        If `True`, the PDB file's CRYST1 records are used to compute atom-atom
        distances and the output `Topology` will have box vectors set accordingly.
        If `False`, the box is ignored.

    """
    u = mda.Universe(pdbfile)

    if not use_box:
        u.dimensions = None

    u.atoms.guess_bonds(
        vdwradii={**_case_insensitise_vdwradii(mda_vdwradii_table), **vdwradii},
        fudge_factor=overlap_factor,
        lower_bound=shortest_bond,
    )

    rdmol = u.atoms.convert_to("RDKIT")
    rdmols = rdkit.Chem.GetMolFrags(rdmol, asMols=True)

    offmols = [Molecule.from_rdkit(rdmol) for rdmol in rdmols]
    offtop = Topology.from_molecules(offmols)
    if u.dimensions is not None:
        offtop.box_vectors = (
            mda.lib.mdamath.triclinic_vectors(u.dimensions) * unit.angstrom
        )

    return offtop
