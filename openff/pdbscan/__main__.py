import sys
import traceback
from enum import Enum
from functools import partial
from multiprocessing import get_context
from pathlib import Path
from typing import Annotated, Callable

import typer
from tqdm import tqdm

from openff.pdbscan.mdanalysis import topology_from_pdb as mda_loader

app = typer.Typer()


class LoadMethod(str, Enum):
    mdanalysis = "mdanalysis"


def _loader_outfn(function, path, **kwargs):
    try:
        function(path, **kwargs)
    except Exception as e:
        print("FAILED:", path, "-", e)
        return False
    else:
        print("WORKED:", path)
        return True


def loader(function, **kwargs) -> Callable[[str], None]:
    return partial(_loader_outfn, function, **kwargs)


PDB_LOADERS = {
    LoadMethod.mdanalysis: loader(mda_loader, use_box=False),
}


@app.command()
def pdbscan(
    method: Annotated[
        LoadMethod,
        typer.Argument(case_sensitive=False),
    ],
    pdbdir: Path,
    n_proc: Annotated[
        int | None,
        typer.Option(
            help="Number of subprocesses to run in parallel",
        ),
    ] = None,
    n_pdbs: Annotated[
        int,
        typer.Option(
            help="Number of PDB files to scan",
        ),
    ] = 10_000,
):
    pdbids = (pdbdir / "all_pdb_ids_shuffled.txt").read_text().split()[:n_pdbs]

    try:
        with get_context("spawn").Pool(n_proc) as pool:
            n_successful = sum(
                list(
                    tqdm(
                        pool.imap_unordered(
                            PDB_LOADERS[method],
                            (f"{pdbdir}/{pdbid}.pdb.gz" for pdbid in pdbids),
                        ),
                        total=n_pdbs,
                    )
                )
            )

    except KeyboardInterrupt:
        print("exiting early", file=sys.stderr)

    print(n_successful, "/", n_pdbs, "completed successfully", file=sys.stderr)


if __name__ == "__main__":
    app()
