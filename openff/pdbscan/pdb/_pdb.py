from collections.abc import Mapping
from os import PathLike
from pathlib import Path

from openff.toolkit import Molecule, Topology
from openff.units import elements

from ._pdb_data import PdbData
from ._pdb_molecule import PDBAtom, PDBBond, PDBMolecule
from ._utils import __UNSET__, cryst_to_box_vectors
from .ccd import CCD_RESIDUE_DEFINITION_CACHE
from .exceptions import NoMatchingResidueDefinitionError
from .residue import ResidueDefinition

__all__ = [
    "topology_from_pdb",
]


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
        properties={"linking_bond": None},
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

    print(f"matching atom names {[data.name[i] for i in res_atom_idcs]}")
    matching_definitions: list[tuple[ResidueDefinition, dict[str, int]]] = []
    for residue_definition in residue_database[res_name]:
        print(
            "    ... to residue definition with atoms ",
            [
                (atom.name, *atom.synonyms, f"{atom.leaving=}")
                for atom in residue_definition.atoms
            ],
            "...",
            sep="",
            end="",
        )
        canonical_name_to_index = data.subset_matches_residue(
            res_atom_idcs,
            residue_definition,
        )
        if canonical_name_to_index is not None:
            # TODO: Consider checking all definitions and raising an error if multiple match
            matching_definitions.append((residue_definition, canonical_name_to_index))
    if len(matching_definitions) == 0:
        raise NoMatchingResidueDefinitionError(res_name, res_seq, chain_id, i_code)

    # TODO: Pick the right residue definition
    residue_definition, canonical_name_to_index = matching_definitions[0]
    residue = residue_definition.to_pdb_molecule()

    residue_wide_metadata = {
        "residue_number": str(res_seq),
        "res_seq": res_seq,
        "insertion_code": i_code,
        "chain_id": chain_id,
    }
    for atom in residue.atoms:
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
            "b_factor": data.temp_factor[pdb_index],
            "occupancy": data.occupancy[pdb_index],
        }

        atom.metadata.update(residue_wide_metadata | atom_specific_metadata)

    residue.sort_atoms_by_metadata("pdb_index")

    return residue


def topology_from_pdb(
    path: PathLike,
    use_canonical_names: bool = False,
    unknown_molecules: list[Molecule] = [],
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

    # TODO: Refactor this into multiple loops, something like:
    #       - get all possible matches for each residue
    #       - filter down possible matches accounting for adjacent residues
    #       - apply unique_molecules
    #       - apply additional_substructures
    #       - break matches into molecules and convert to `openff.toolkit.Molecule`s

    molecules: list[PDBMolecule] = []
    current_molecule = PDBMolecule()
    prev_chain_id = __UNSET__
    prev_model = __UNSET__
    # Chunk atom indices into residues and iterate over them (see PdbData.residues())
    for res_atom_idcs in data.residues():
        prototype_index = res_atom_idcs[0]
        res_name = data.res_name[prototype_index]

        # If this is the first residue, we'll need to initialize the variables
        # that refer to the previous residue as though the previous residue
        # was in the same chain and model (imagine there's an empty zeroth
        # residue)
        if prev_chain_id is __UNSET__:
            prev_chain_id = data.chain_id[prototype_index]
        if prev_model is __UNSET__:
            prev_model = data.model[prototype_index]

        # Identify this residue and get a PDBMolecule of it
        # TODO: UNK is for unknown peptide residues, but we just treat it as a ligand
        # Note that for the CCD_RESIDUE_DEFINITION_CACHE, "not in" includes a check
        # for the residue names "UNL" and "UNK"
        if res_name not in residue_database:
            residue = _load_unknown_residue(data, res_atom_idcs, unknown_molecules)
        else:
            residue = _load_residue_from_database(data, res_atom_idcs, residue_database)

        # Terminate the previous molecule and start a new one if we can see that
        # this is the start of a new molecule
        if (
            residue.properties["linking_bond"] is None
            or data.chain_id[prototype_index] != prev_chain_id
            or data.model[prototype_index] != prev_model
        ):
            molecules.append(current_molecule)
            current_molecule = PDBMolecule()

        # Add the PDBMolecule matching the current residue to the growing molecule
        # TODO: Test that atoms are never re-ordered
        current_molecule.combine_with(residue)

        # Terminate the current molecule if we can see that this is the last residue
        if (
            data.terminated[prototype_index]
            or current_molecule.properties["linking_bond"] is None
        ):
            molecules.append(current_molecule)
            current_molecule = PDBMolecule()

        # TODO: Load other data from PDB file
        # TODO: Incorporate CONECT records
        # TODO: Deal with multi-model files

        prev_chain_id = data.chain_id[prototype_index]
        prev_model = data.model[prototype_index]

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
