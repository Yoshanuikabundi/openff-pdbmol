import dataclasses
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Iterable, Iterator, Mapping, Self, Sequence

from ._utils import __UNSET__, dec_hex
from .residue import AtomDefinition, ResidueDefinition


@dataclass(frozen=True)
class ResidueMatch:
    index_to_atomdef: dict[int, AtomDefinition]
    residue_definition: ResidueDefinition
    missing_atoms: set[str]

    def atom(self, identifier: int | str) -> AtomDefinition:
        if isinstance(identifier, int):
            return self.index_to_atomdef[identifier]
        elif isinstance(identifier, str):
            return self.residue_definition.name_to_atom[identifier]
        else:
            raise TypeError(f"unknown identifier type {type(identifier)}")

    @cached_property
    def res_atom_idcs(self) -> set[int]:
        return set(self.index_to_atomdef)

    @cached_property
    def missing_leaving_atoms(self) -> set[str]:
        return {
            atom_name
            for atom_name in self.missing_atoms
            if self.atom(atom_name).leaving
        }

    @cached_property
    def expect_prior_bond(self) -> bool:
        if self.residue_definition.linking_bond is None:
            return False

        linking_atom = self.residue_definition.linking_bond.atom2
        print(f"{linking_atom=}")
        bonded_to_linking_atom = self.residue_definition.atoms_bonded_to(linking_atom)
        print(f"{linking_atom=}, {bonded_to_linking_atom=}")

        # TODO: Check all leaving atoms for the prior bond
        return any(
            atom in self.missing_leaving_atoms for atom in bonded_to_linking_atom
        )

    @cached_property
    def expect_posterior_bond(self) -> bool:
        if self.residue_definition.linking_bond is None:
            return False

        linking_atom = self.residue_definition.linking_bond.atom1

        # TODO: Check all leaving atoms for the posterior bond
        return any(
            atom in self.missing_leaving_atoms
            for atom in self.residue_definition.atoms_bonded_to(linking_atom)
        )


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
        for field_ in dataclasses.fields(self):
            value = getattr(self, field_.name)
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
        for field_ in dataclasses.fields(self):
            value = getattr(self, field_.name)
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
        for atom_idx, residue_info in enumerate(
            zip(
                self.model,
                self.res_name,
                self.chain_id,
                self.res_seq,
                self.i_code,
            )
        ):
            if prev == residue_info or prev is None:
                indices.append(atom_idx)
            else:
                yield indices
                indices = [atom_idx]
            prev = residue_info

        yield indices

    def __getitem__(self, index) -> dict[str, Any]:
        return {
            field.name: getattr(self, field.name)[index]
            for field in dataclasses.fields(self)
        }

    def subset_matches_residue(
        self,
        res_atom_idcs: Sequence[int],
        residue_definition: ResidueDefinition,
    ) -> ResidueMatch | None:
        # Raise an error if the returned dict would be empty - this way the
        # return value's truthiness always reflects whether there was a match
        if len(res_atom_idcs) == 0:
            raise ValueError("cannot match empty res_atom_idcs")

        # Skip definitions with too few atoms
        if len(residue_definition.atoms) < len(res_atom_idcs):
            print("res def has too few atoms")
            return None

        # Skip non-linking definitions with the wrong number of atoms
        if residue_definition.linking_bond is None and len(
            residue_definition.atoms
        ) != len(res_atom_idcs):
            print("nonlinking res def has wrong number of atoms")
            return None

        # Get the map from the canonical names to the indices
        try:
            index_to_atomdef = {
                i: residue_definition.name_to_atom[self.name[i]] for i in res_atom_idcs
            }
        except KeyError as e:
            print(
                "name in pdb file missing from res def 1:",
                e,
                {
                    name: atom.name
                    for name, atom in residue_definition.name_to_atom.items()
                },
            )
            return None

        matched_atoms = set(atom.name for atom in index_to_atomdef.values())

        # Fail to match if any atoms in PDB file got matched to more than one name
        if len(matched_atoms) != len(res_atom_idcs):
            print("name in pdb file with multiple matches in res def")
            return None

        # This assert should be guaranteed by the above
        assert set(index_to_atomdef.keys()) == set(res_atom_idcs)

        missing_atoms = [
            atom for atom in residue_definition.atoms if atom.name not in matched_atoms
        ]

        # Match only if all atoms missing from the PDB file are leaving atoms
        if all(atom.leaving for atom in missing_atoms):
            print("matched!")
            return ResidueMatch(
                index_to_atomdef=index_to_atomdef,
                residue_definition=residue_definition,
                missing_atoms={atom.name for atom in missing_atoms},
            )
        else:
            print(
                "missing atom is not leaving:",
                [(atom.name, atom.synonyms, atom.leaving) for atom in missing_atoms],
            )
            return None

    def get_residue_matches(
        self,
        residue_database: Mapping[str, list[ResidueDefinition]],
    ) -> Iterator[list[ResidueMatch]]:
        for res_atom_idcs in self.residues():
            prototype_index = res_atom_idcs[0]
            res_name = self.res_name[prototype_index]

            print("matching new residue", res_name)

            matches = []
            for residue_definition in residue_database.get(res_name, []):
                match = self.subset_matches_residue(
                    res_atom_idcs,
                    residue_definition,
                )

                if match is not None:
                    matches.append(match)
            yield matches

    def are_alt_locs(self, i: int, j: int):
        if i == j:
            raise ValueError(f"i and j are the same ({i})")
        if max(i, j) - min(i, j) == 1:
            return (
                self.model[i],
                self.name[i],
                self.res_name[i],
                self.chain_id[i],
                self.res_seq[i],
                self.i_code[i],
            ) == (
                self.model[j],
                self.name[j],
                self.res_name[j],
                self.chain_id[j],
                self.res_seq[j],
                self.i_code[j],
            )
        else:
            return self.are_alt_locs(i, i + 1) and self.are_alt_locs(i + 1, j)
