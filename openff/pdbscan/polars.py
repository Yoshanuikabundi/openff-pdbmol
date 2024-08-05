import polars as pl
from pathlib import Path
from typing import Iterable
import gzip

__all__ = [
    "load_coords",
    "COORD_LINE_SCHEMA",
]


COORD_LINE_SCHEMA = {
    "id": pl.String,
    "err": pl.String,
    "lineNo": pl.Int32,
    "serial": pl.Int32,
    "name": pl.String,
    "altLoc": pl.String,
    "resName": pl.String,
    "chainID": pl.String,
    "resSeq": pl.Int32,
    "iCode": pl.String,
    "x": pl.Float32,
    "y": pl.Float32,
    "z": pl.Float32,
    "occupancy": pl.Float32,
    "tempFactor": pl.Float32,
    "element": pl.String,
    "charge": pl.String,
    "terminated": pl.Boolean,
    "modelNo": pl.Int32,
    "conects": pl.List(pl.Int32),
}


def _prepare_row(data: dict[str, list], path: Path):
    """
    Prepare a new row in a dictionary for a PDB file

    The dictionary is modified in place. The new row can be assigned to
    or read from as ``data[colname][-1]``.
    """
    for column in data.values():
        column.append(None)
    data["id"][-1] = path.stem[3:-4]


def load_coords(paths: Iterable[Path]) -> dict[str, list]:
    """
    Load the coordinate and CONECT records from PDB files into a dictionary.

    Parameters
    ----------
    paths
        Paths to the PDB files to load.

    Returns
    -------
    data
        A dictionary mapping from the keys of ``COORD_LINE_SCHEMA`` to lists of
        Python objects that can be implicitly coerced to the corresponding
        values, each of which is a Polars datatype.
    """
    data = {k: [] for k in COORD_LINE_SCHEMA}
    for path in paths:
        try:
            with gzip.open(path, "rt") as f:
                lines = f.readlines()
        except Exception as e:
            _prepare_row(data, path)
            data["err"][-1] = repr(e)
            continue
        model_n = 0
        conects = {}
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
        for line_n, line in enumerate(lines):
            if line.startswith("MODEL "):
                model_n = int(line[10:14])
            if line.startswith("ENDMDL "):
                model_n += 1
            if line.startswith("HETATM") or line.startswith("ATOM  "):
                _prepare_row(data, path)
                data["lineNo"][-1] = line_n
                data["serial"][-1] = int(line[6:11])
                data["name"][-1] = line[12:16].strip()
                data["altLoc"][-1] = line[16].strip() or None
                data["resName"][-1] = line[17:20].strip()
                data["chainID"][-1] = line[21].strip()
                data["resSeq"][-1] = int(line[22:26])
                data["iCode"][-1] = line[26].strip() or None
                data["x"][-1] = line[30:38].strip()
                data["y"][-1] = line[38:46].strip()
                data["z"][-1] = line[46:54].strip()
                data["occupancy"][-1] = line[54:60].strip()
                data["tempFactor"][-1] = line[60:66].strip()
                data["element"][-1] = line[76:78].strip()
                data["charge"][-1] = line[78:80].strip() or None
                data["terminated"][-1] = False
                data["modelNo"][-1] = model_n
                data["conects"][-1] = list(conects.get(data["serial"][-1], []))
            if line.startswith("TER   "):
                terminated_resname = line[17:20].strip() or data["resName"][-1]
                terminated_chainid = line[21].strip() or data["chainID"][-1]
                terminated_resseq = int(line[22:26]) or data["resSeq"][-1]
                for i in range(-1, -999, -1):
                    if (
                        data["resName"][i] == terminated_resname
                        and data["chainID"][i] == terminated_chainid
                        and data["resSeq"][i] == terminated_resseq
                    ):
                        data["terminated"][i] = True
                    else:
                        break
                else:
                    assert False, "last residue too big"
    return data
