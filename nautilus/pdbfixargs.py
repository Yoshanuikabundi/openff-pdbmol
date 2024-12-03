import sys
from traceback import print_exception

from pdbfix import proc_pdbid


def main():
    pdb_ids = sys.argv[1:]
    failed_pdb_ids = {}
    for id in pdb_ids:
        try:
            proc_pdbid(id, f"{id}.pdb.gz")
        except Exception as exc:
            print(f"{id} failed:", file=sys.stderr)
            print(print_exception(exc), file=sys.stderr)
            failed_pdb_ids[id] = exc


if __name__ == "__main__":
    main()
