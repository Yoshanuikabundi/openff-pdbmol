import itertools
from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import Sequence

import numpy as np
from typing_extensions import assert_never

from openff.toolkit import Molecule, Topology
from openff.units import elements, unit

from ._pdb_data import PdbData, ResidueMatch
from ._utils import cryst_to_box_vectors
from .ccd import CCD_RESIDUE_DEFINITION_CACHE
from .exceptions import (
    MultipleMatchingResidueDefinitionsError,
    NoMatchingResidueDefinitionError,
)
from .residue import ResidueDefinition

__all__ = [
    "topology_from_pdb",
]


def _load_unknown_residue(
    data: PdbData, indices: tuple[int, ...], unknown_molecules: Sequence[Molecule]
) -> Molecule | None:
    print(f"Consulting {unknown_molecules=}")
    conects = set()
    serial_to_mol_index = {}
    pdbmol = Molecule()
    for i, pdb_index in enumerate(indices):
        serial_to_mol_index[data.serial[pdb_index]] = pdbmol.add_atom(
            atomic_number=elements.NUMBERS[data.element[pdb_index]],
            formal_charge=data.charge[pdb_index],
            is_aromatic=False,
            stereochemistry=None,
            name=data.name[pdb_index],
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
        for conect_serial in data.conects[pdb_index]:
            conects.add(tuple(sorted([data.serial[pdb_index], conect_serial])))

    for serial1, serial2 in conects:
        pdbmol.add_bond(
            atom1=serial_to_mol_index[serial1],
            atom2=serial_to_mol_index[serial2],
            bond_order=1,
            is_aromatic=False,
        )

    for molecule in unknown_molecules:
        (match_found, mapping) = Molecule.are_isomorphic(
            pdbmol,
            molecule,
            return_atom_map=True,
            aromatic_matching=False,
            formal_charge_matching=False,
            bond_order_matching=False,
            atom_stereochemistry_matching=False,
            bond_stereochemistry_matching=False,
            strip_pyrimidal_n_atom_stereo=True,
        )
        print(f"{match_found=} {mapping=} {molecule=} {pdbmol=}")
        if match_found:
            assert mapping is not None
            molecule = Molecule(molecule)
            for i, atom in enumerate(molecule.atoms):
                pdbatom = pdbmol.atom(mapping[i])
                atom.metadata.update(pdbatom.metadata)
                atom.name = pdbatom.name
            molecule._conformers = None

            return molecule
    else:
        return None


def topology_from_pdb(
    path: PathLike[str],
    use_canonical_names: bool = False,
    unknown_molecules: Sequence[Molecule] = [],
    residue_database: Mapping[
        str, list[ResidueDefinition]
    ] = CCD_RESIDUE_DEFINITION_CACHE,
) -> Topology:
    """
    Load a PDB file into an OpenFF ``Topology``.

    This function requires all hydrogens (and all other atoms) to be present in
    the PDB file, and that atom and residue names are consistent with the
    ``residue_database``. In return, it provides full chemical information on
    the entire PDB file.

    To load a PDB file with molecules including any residue not found in the
    CCD, or with residues that differ from that specified under a particular
    residue name, provide your own ``residue_database``. Any mapping from a
    residue name to a list of :py:data:`ResidueDefinition
    <openff.pdbscan.pdb.residue.ResidueDefinition>` objects may be used,
    but the :py:mod:`ccd <openff.pdbscan.pdb.ccd>` module  provides tools for
    augmenting the CCD.

    Alternatively, to load a single-residue molecule that is not present in the
    CCD, name that molecule ``"UNL"`` (or any name not present in the
    ``residue_database``), specify its CONECT records, and provide the
    appropriate molecule to the ``unknown_molecules`` argument.

    Parameters
    ----------
    path
        The path to the PDB file.
    use_canonical_names
        If ``True``, atom names in the PDB file will be replaced by the
        canonical name for the same atom from the residue database.
    unknown_molecules
        A list of molecules to match residues not found in the
        ``residue_database`` against. Unlike ``residue_database``, this requires
        that CONECT records be present and performs a match between the chemical
        graphs rather than using residue and atom names to detect chemistry.
    residue_database
        The database of residues to identify the atoms in the PDB file by. By
        default, a patched version of the CCD. Chemistry is identified by atom
        and residue names. If multiple residue definitions match a particular
        residue, the first one encountered is applied.

    Notes
    -----

    This function uses a residue database to load a PDB file from its atom and
    residue names without guessing bonds. Bonds will be added by comparing atom
    and residue names to the residues defined in the ``residue_database``
    argument, which by default uses a patched version of the RCSB Chemical
    Component Dictionary (CCD). This is the dictionary of residue and atom names
    that the RCSB PDB is referenced against. The CCD is very large and cannot be
    distributed with this software, so by default internet access is required to
    use it.

    The following metadata are specified for all atoms produced by this function
    and can be accessed via ``topology.atom(i).metadata[key]``:

    ``"residue_name"``
        The residue name
    ``"residue_number"``
        The residue number as a string
    ``"res_seq"``
        The residue number as an integer
    ``"insertion_code"``
        The icode for the atom's residue. Used to align residue numbers between
        proteins with indels.
    ``"chain_id"``
        The letter identifier for the atom's chain.
    ``"pdb_index"``
        The atom's index in the PDB file. Sometimes called rank. Not to be
        confused with ``"atom_serial"``, which is the number given to the atom
        in the second column of the PDB file. Guaranteed to be unique and to
        match the index of the atom within the topology.
    ``"used_synonym"``
        The name of the atom that was found in the PDB file. By default,
        `atom.name` is set to this.
    ``"canonical_name"``
        The canonical name of the atom in the residue database. `atom.name` can
        be set to this with the `use_canonical_names` argument.
    ``"atom_serial"``
        The serial number of the atom, found in the second column of the PDB
        file. Not guaranteed to be unique.
    ``"matched_residue_description"``
        The residue description found in the residue database.
    ``"b_factor"``
        The temperature b-factor for the atom.
    ``"occupancy"``
        The occupancy for the atom.

    """
    # TODO: support streams and gzipped files
    path = Path(path)
    data = PdbData.parse_pdb(path.read_text().splitlines())

    molecules: list[Molecule] = []
    this_molecule = Molecule()
    prev_chain_id = data.chain_id[0]
    prev_model = data.model[0]
    conformer: list[tuple[float, float, float]] = []
    for res_atom_idcs, matches in zip(
        data.residue_indices, data.get_residue_matches(residue_database)
    ):
        # Check that we have a unique match, and error out or consult
        # unique_molecules as appropriate
        chemical_data: Molecule | ResidueMatch
        if len(matches) == 0:
            unknown_molecule = _load_unknown_residue(
                data, res_atom_idcs, unknown_molecules
            )
            if unknown_molecule is None:
                raise NoMatchingResidueDefinitionError(res_atom_idcs, data)
            else:
                chemical_data = unknown_molecule
        # If all matches would assign the same chemistry, accept it
        # assert all([]) == True
        elif all(a.agrees_with(b) for a, b in itertools.pairwise(matches)):
            if len(matches) > 1:
                print("Multiple filtered matches, but they are all equivalent")
            chemical_data = matches[0]

            # this is a debug assert, if it triggers there's a bug
            assert set(res_atom_idcs) == chemical_data.res_atom_idcs
        else:
            raise MultipleMatchingResidueDefinitionsError(matches, res_atom_idcs, data)

        prototype_index = res_atom_idcs[0]

        # Terminate the previous molecule and start a new one if we can see that
        # this is the start of a new molecule
        if this_molecule.n_atoms > 0 and (
            data.chain_id[prototype_index] != prev_chain_id
            or data.model[prototype_index] != prev_model
            or (
                isinstance(chemical_data, ResidueMatch)
                and not chemical_data.expect_prior_bond
            )
            or (isinstance(chemical_data, Molecule))
        ):
            this_molecule._invalidate_cached_properties()
            this_molecule.add_conformer(np.asarray(conformer) * unit.nanometer)
            molecules.append(this_molecule)
            this_molecule = Molecule()
            conformer = []

        conformer.extend((data.x[i], data.y[i], data.z[i]) for i in res_atom_idcs)

        # Apply the chemical data we've collected
        if isinstance(chemical_data, Molecule):
            this_molecule = chemical_data
        elif isinstance(chemical_data, ResidueMatch):
            add_to_molecule(
                this_molecule,
                res_atom_idcs,
                chemical_data,
                data,
                use_canonical_names,
            )
        else:
            assert_never(chemical_data)

        # Terminate the current molecule if we can see that this is the last residue
        if (
            data.terminated[prototype_index]
            or isinstance(chemical_data, Molecule)
            or (
                isinstance(chemical_data, ResidueMatch)
                and not chemical_data.expect_posterior_bond
            )
        ):
            this_molecule._invalidate_cached_properties()
            this_molecule.add_conformer(np.asarray(conformer) * unit.nanometer)
            molecules.append(this_molecule)
            this_molecule = Molecule()
            conformer = []

        # TODO: Load other data from PDB file
        # TODO: Incorporate CONECT records
        # TODO: Deal with multi-model files

        prev_chain_id = data.chain_id[prototype_index]
        prev_model = data.model[prototype_index]

    for offmol in molecules:
        offmol._invalidate_cached_properties()
        offmol.add_default_hierarchy_schemes()
    molecules.append(this_molecule)

    topology = Topology.from_molecules(molecules)  # type: ignore[call-arg]
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

    return topology


def add_to_molecule(
    this_molecule: Molecule,
    res_atom_idcs: tuple[int, ...],
    residue_match: ResidueMatch,
    data: PdbData,
    use_canonical_names: bool,
) -> None:
    # Add the residue to the current molecule
    atom_name_to_mol_idx = {}
    for pdb_index in res_atom_idcs:
        atom_def = residue_match.atom(pdb_index)

        if data.alt_loc[pdb_index] != "":
            # TODO: Support altlocs (probably in PdbData, maybe PdbData.residues()?)
            raise ValueError("altloc not yet supported")

        atom_name_to_mol_idx[atom_def.name] = this_molecule._add_atom(
            atomic_number=elements.NUMBERS[atom_def.symbol],
            formal_charge=atom_def.charge,
            is_aromatic=atom_def.aromatic,
            stereochemistry=atom_def.stereo,
            name=atom_def.name if use_canonical_names else data.name[pdb_index],
            metadata={
                "residue_name": data.res_name[pdb_index],
                "res_seq": data.res_seq[pdb_index],
                "residue_number": str(data.res_seq[pdb_index]),
                "insertion_code": data.i_code[pdb_index],
                "chain_id": data.chain_id[pdb_index],
                "pdb_index": pdb_index,
                "used_synonym": data.name[pdb_index],
                "canonical_name": atom_def.name,
                "atom_serial": data.serial[pdb_index],
                "matched_residue_description": residue_match.residue_definition.description,
                "b_factor": str(data.temp_factor[pdb_index]),
                "occupancy": str(data.occupancy[pdb_index]),
            },
            invalidate_cache=False,
        )

    for bond in residue_match.residue_definition.bonds:
        if bond.atom1 in atom_name_to_mol_idx and bond.atom2 in atom_name_to_mol_idx:
            this_molecule._add_bond(
                # TODO: Fix
                atom1=atom_name_to_mol_idx[bond.atom1],
                atom2=atom_name_to_mol_idx[bond.atom2],
                bond_order=bond.order,
                is_aromatic=bond.aromatic,
                stereochemistry=None,  # TODO: Calculate stereo from coords
                invalidate_cache=False,
            )

    # TODO: Add inter-residue bonds
