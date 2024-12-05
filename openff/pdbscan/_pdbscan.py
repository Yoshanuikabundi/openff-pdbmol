import multiprocessing
from os import PathLike, cpu_count
from pathlib import Path
from typing import Callable, Iterable

import mdtraj
import nglview
import numpy as np
import openmm
import openmm.app
import polars as pl
from polars.type_aliases import SizeUnit

from openff.units import ensure_quantity


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            return
        yield batch


def nglview_show_openmm(
    topology: openmm.app.Topology, positions, image_molecules=False
):
    top = mdtraj.Topology.from_openmm(topology)

    if isinstance(positions, str) or isinstance(positions, Path):
        traj = mdtraj.load(positions, top=top)
        if image_molecules:
            traj.image_molecules(inplace=True)
    else:
        positions = ensure_quantity(positions, "openmm").value_in_unit(
            openmm.unit.nanometer
        )
        xyz = np.asarray([positions])
        box_vectors = topology.getPeriodicBoxVectors()
        if box_vectors is not None:
            (
                l1,
                l2,
                l3,
                alpha,
                beta,
                gamma,
            ) = mdtraj.utils.box_vectors_to_lengths_and_angles(
                *np.asarray(box_vectors.value_in_unit(openmm.unit.nanometer))
            )
            unitcell_angles, unitcell_lengths = [alpha, beta, gamma], [l1, l2, l3]
        else:
            unitcell_angles, unitcell_lengths = None, None
        traj = mdtraj.Trajectory(
            xyz, top, unitcell_lengths=unitcell_lengths, unitcell_angles=unitcell_angles
        )
    return nglview.show_mdtraj(traj)


def scan_pdb(
    *,
    inpaths: PathLike | Iterable[PathLike],
    outpath: PathLike,
    proc: Callable[[Iterable[Path]], dict[str, list]],
    schema: dict[str, pl.DataType],
    chunksize: int = 100,
    outfilesize: tuple[float, SizeUnit] = (5, "gb"),
    processes: int | None = None,
    silent: bool = False,
):
    """
    Scan the PDB into a dataframe and store the results on disk.

    Parameters
    ----------
    inpaths
        A path to a directory that stores a collection of PDB files in its
        subdirectories with the suffixes ".ent.gz", or a list of paths to PDB
        files.
    outpath
        Path to store output file(s)
    proc
        Function called on each path to convert that PDB file into a dictionary
        that can later be collated into the saved dataframe. The returned
        dictionary should map from the keys of ``schema`` to lists of values
        coercable to the corresponding values of ``schema``.
    schema
        Schema to conform the saved dataframe to. The dictionary returned by
        ``proc`` should have the same keys as the schema, and each value should
        be a list of types coercable into the datatype given as the
        corresponding value of the schema.
    chunksize
        Number of PDB files to be processed in series at a time. Decrease this
        if you run into memory pressure issues within a batch, increase this if
        CPUs are under-utilized during initial batch processing.
    outfilesize
        Approximate size the dataframe must grow to before offloading it from
        memory to disk. Decrease this if you run into memory pressure issues
        between batches, increase this to reduce the number of files created.
    processes
        Number of processes to split processing over. If ``None``, uses the
        value of ``os.cpu_count()``
    silent
        If ``True``, silence normal terminal output.
    """
    try:
        inpaths = Path(inpaths).glob("*/*.ent.gz")
    except TypeError:
        inpaths = map(Path, inpaths)

    if processes is None:
        processes = cpu_count()
        if processes is None:
            return ValueError(
                "Could not determine number of CPUs, please specify processes argument"
            )

    outpath = Path(outpath)
    if outpath.suffix == ".parquet":
        outpath = outpath.with_suffix("")
    outstr = str(outpath) + ".{i}.parquet"

    df = pl.DataFrame(schema=schema)
    i = 0
    for i, batch in enumerate(batched(inpaths, n=chunksize * processes)):
        if not silent:
            print("batch", i)
        with multiprocessing.Pool(processes=processes) as pool:
            batch_results = list(
                pool.imap_unordered(
                    proc,
                    batched(batch, chunksize),
                )
            )

        if not silent:
            print("concatenating results of batch", i)
        df = pl.concat([df, *(pl.DataFrame(d, schema=schema) for d in batch_results)])

        if not silent:
            print("rechunking after batch", i)
        df.rechunk()

        if not silent:
            print("done with batch", i, f"({df.estimated_size('mb')} mb)")
        if df.estimated_size(outfilesize[1]) >= outfilesize[0]:
            if not silent:
                print("writing recent batches to file")
            df.write_parquet(outstr.format(i))
            df = pl.DataFrame(schema=schema)

    if not silent:
        print("writing final batches to file")

    df.write_parquet(outstr.format(i))
