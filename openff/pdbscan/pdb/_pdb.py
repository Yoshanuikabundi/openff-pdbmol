from collections.abc import Mapping
from os import PathLike
from pathlib import Path

from openff.toolkit import Molecule, Topology
from openff.units import elements

from ._pdb_data import PdbData, ResidueMatch
from ._utils import cryst_to_box_vectors, with_neighbours
from .ccd import CCD_RESIDUE_DEFINITION_CACHE
from .exceptions import (
    MultipleMatchingResidueDefinitionsError,
    NoMatchingResidueDefinitionError,
)
from .residue import ResidueDefinition

__all__ = [
    "topology_from_pdb",
]


# def _load_unknown_residue(
#     data: PdbData, indices: list[int], unknown_molecules: list[Molecule]
# ) -> PDBMolecule:
#     atoms = []
#     conects = set()
#     serial_to_index = {}
#     for i, pdb_index in enumerate(indices):
#         new_atom = PDBAtom(
#             atomic_number=elements.NUMBERS[data.element[pdb_index]],
#             formal_charge=data.charge[pdb_index],
#             is_aromatic=None,
#             stereochemistry=None,
#             name=data.name[pdb_index],
#             x=data.x[pdb_index],
#             y=data.y[pdb_index],
#             z=data.z[pdb_index],
#             metadata={
#                 "residue_name": data.res_name[pdb_index],
#                 "leaving": False,
#                 "pdb_index": pdb_index,
#                 "residue_number": str(data.res_seq[pdb_index]),
#                 "res_seq": data.res_seq[pdb_index],
#                 "insertion_code": data.i_code[pdb_index],
#                 "chain_id": data.chain_id[pdb_index],
#                 "atom_serial": data.serial[pdb_index],
#             },
#         )
#         atoms.append(new_atom)
#         serial_to_index[data.serial[pdb_index]] = i
#         for conect_serial in data.conects[pdb_index]:
#             conects.add(tuple(sorted([data.serial[pdb_index], conect_serial])))

#     bonds = []
#     for serial1, serial2 in conects:
#         bonds.append(
#             PDBBond(
#                 atom1=serial_to_index[serial1],
#                 atom2=serial_to_index[serial2],
#             )
#         )

#     pdbmol = PDBMolecule(
#         atoms=atoms,
#         bonds=bonds,
#         properties={"linking_bond": None},
#     )

#     for molecule in unknown_molecules:
#         (match_found, mapping) = Molecule.are_isomorphic(
#             pdbmol.to_networkx(),
#             molecule.to_networkx(),
#             return_atom_map=True,
#             aromatic_matching=False,
#             formal_charge_matching=False,
#             bond_order_matching=False,
#             atom_stereochemistry_matching=False,
#             bond_stereochemistry_matching=False,
#             strip_pyrimidal_n_atom_stereo=True,
#         )
#         assert mapping is not None
#         if match_found:
#             reverse_map = {}
#             for i, atom in enumerate(pdbmol.atoms):
#                 reverse_map[mapping[i]] = i
#                 reference_atom = molecule.atom(mapping[i])
#                 atom.formal_charge = reference_atom.formal_charge
#                 atom.is_aromatic = reference_atom.is_aromatic
#                 atom.stereochemistry = reference_atom.stereochemistry  # type: ignore[assignment]
#             pdbmol.bonds.clear()
#             for bond in molecule.bonds:
#                 pdbmol.add_bond(
#                     PDBBond(
#                         atom1=reverse_map[bond.atom1_index],
#                         atom2=reverse_map[bond.atom2_index],
#                         bond_order=bond.bond_order,
#                         is_aromatic=bond.is_aromatic,
#                         stereochemistry=bond.stereochemistry,
#                     )
#                 )
#             break
#     else:
#         res_name = pdbmol.atoms[0].metadata["residue_name"]
#         res_seq = pdbmol.atoms[0].metadata["res_seq"]
#         chain_id = pdbmol.atoms[0].metadata["chain_id"]
#         raise ValueError(
#             f"Unknown residue {chain_id}:{res_name}#{res_seq} could not be assigned chemistry from unknown_molecules"
#         )

#     return pdbmol


def topology_from_pdb(
    path: PathLike,
    use_canonical_names: bool = False,
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

    prev_filtered_matches: list[ResidueMatch] = []
    matched_residues: list[ResidueMatch] = []
    # TODO: Refactor into loop over with_neighbours(data.residues()) by combining matching logic
    for _, this_matches, next_matches in with_neighbours(
        data.get_residue_matches(residue_database),
        [],
    ):
        this_filtered_matches = []
        for match in this_matches:
            # TODO: Simplify this and factor out into function with logic in PdbData.subset_matches_residue
            if len(match.missing_atoms) != 0:
                prior_bond_mismatched = match.expect_prior_bond != any(
                    prev_match.expect_posterior_bond
                    for prev_match in prev_filtered_matches
                )
                if prior_bond_mismatched:
                    continue

                # assert any([]) == False
                posterior_bond_mismatched = match.expect_posterior_bond != any(
                    next_match.expect_prior_bond for next_match in next_matches
                )
                if posterior_bond_mismatched:
                    continue

            this_filtered_matches.append(match)

        if len(this_filtered_matches) == 0:
            # TODO: Implement unique_molecules and additional_substructures here
            raise NoMatchingResidueDefinitionError()
        elif len(this_filtered_matches) == 1:
            matched_residues.append(this_filtered_matches[0])
        else:
            raise MultipleMatchingResidueDefinitionsError()

        prev_filtered_matches = this_filtered_matches

    # Now, we convert to OpenFF molecules
    molecules: list[Molecule] = []
    this_molecule = Molecule()
    pdb_index_to_molecule_index = {}
    prev_chain_id = data.chain_id[0]
    prev_model = data.model[0]
    conformer: list[tuple[float, float, float] | None] = []
    for res_atom_idcs, residue_match in zip(data.residues(), matched_residues):
        # this is a debug assert, if it triggers there's a bug
        assert set(res_atom_idcs) == residue_match.res_atom_idcs

        prototype_index = res_atom_idcs[0]

        res_seq = data.res_seq[prototype_index]
        residue_wide_metadata = {
            "residue_name": data.res_name[prototype_index],
            "res_seq": res_seq,
            "residue_number": str(res_seq),
            "insertion_code": data.i_code[prototype_index],
            "chain_id": data.chain_id[prototype_index],
        }

        # Terminate the previous molecule and start a new one if we can see that
        # this is the start of a new molecule
        if this_molecule.n_atoms > 0 and (
            data.chain_id[prototype_index] != prev_chain_id
            or data.model[prototype_index] != prev_model
            or not residue_match.expect_prior_bond
        ):
            this_molecule._invalidate_cached_properties()
            molecules.append(this_molecule)
            this_molecule = Molecule()

        # Add the residue to the current molecule
        # TODO: Break this into a function
        for pdb_index in res_atom_idcs:
            atom_def = residue_match.atom(pdb_index)

            if data.alt_loc[pdb_index] != "":
                # TODO: Support altlocs (probably in PdbData, maybe PdbData.residues()?)
                raise ValueError("altloc not yet supported")

            xyz = (data.x[pdb_index], data.y[pdb_index], data.z[pdb_index])
            conformer.append(xyz)

            pdb_index_to_molecule_index[pdb_index] = this_molecule._add_atom(
                atomic_number=elements.NUMBERS[atom_def.symbol],
                formal_charge=atom_def.charge,
                is_aromatic=atom_def.aromatic,
                stereochemistry=atom_def.stereo,
                name=atom_def.name if use_canonical_names else data.name[pdb_index],
                metadata={
                    **residue_wide_metadata,
                    "pdb_index": pdb_index,
                    "used_synonym": data.name[pdb_index],
                    "canonical_name": atom_def.name,
                    "atom_serial": data.serial[pdb_index],
                    "matched_residue_description": residue_match.residue_definition.description,
                    "b_factor": data.temp_factor[pdb_index],
                    "occupancy": data.occupancy[pdb_index],
                },
                invalidate_cache=False,
            )

        for bond in residue_match.residue_definition.bonds:
            this_molecule._add_bond(
                atom1=residue_match.atom(bond.atom1),
                atom2=residue_match.atom(bond.atom2),
                bond_order=bond.order,
                is_aromatic=bond.aromatic,
                stereochemistry=None,  # TODO: Calculate stereo from coords
                invalidate_cache=False,
            )

        # Terminate the current molecule if we can see that this is the last residue
        if data.terminated[prototype_index] or not residue_match.expect_posterior_bond:
            molecules.append(this_molecule)
            this_molecule = Molecule()

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
