import gzip
from pathlib import Path
from urllib.request import urlopen

from openmm.app import PDBFile
from pdbfixer import PDBFixer

__all__ = ["proc_pdbid"]


def proc_pdbid(id: str, outfn: Path):
    fixer = fixer_from_pdbid(id)

    # We don't want to do any whole residue reconstruction
    # fixer.findMissingResidues()
    # chainLengths = [len([*chain.residues()]) for chain in fixer.topology.chains()]
    # for chainidx, residx in list(fixer.missingResidues):
    #     if residx == 0:
    #         fixer.missingResidues[chainidx, residx] = ["ACE"]
    #     elif residx == chainLengths[chainidx]:
    #         fixer.missingResidues[chainidx, residx] = ["NME"]

    # We want to keep non-standard residues and heterogens, using PDBFixer's
    # new CCD reading powers to fix them
    # fixer.findNonstandardResidues()
    # fixer.replaceNonstandardResidues()
    # fixer.removeHeterogens(keepWater=True)

    # We do want to fix any atoms, heavy or hydrogen, missing from the file
    fixer.missingResidues = {}
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(pH=7.4)

    with gzip.open(outfn, mode="wt") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)


def fixer_from_pdbid(id: str) -> PDBFixer:
    """
    Load a PDBFixer object from a PDBID.

    Differs from ``PDBFixer(pdbid=id)`` because this function downloads the PDB
    file in gzipped format, which reduces network usage, whereas the
    ``PDBFixer`` constructor downloads in uncompressed .PDB format.
    """
    with urlopen(f"https://files.rcsb.org/download/{id}.pdb.gz") as gzfile:
        pdbfile = gzip.GzipFile(fileobj=gzfile)
        fixer = PDBFixer(pdbfile=pdbfile)
    return fixer
