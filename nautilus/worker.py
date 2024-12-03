from time import sleep

import rediswq

from pdbfix import proc_pdbid


def main():
    print("Connecting to queue")

    host = "pdbscan-jm-redis"

    q = rediswq.RedisWQ(name="job2", host=host)
    print("Worker with sessionID: " + q.sessionID())
    print("Initial queue state: empty=" + str(q.empty()))
    while not q.empty():
        print("Requesting lease")
        item = q.lease(lease_secs=100, block=True, timeout=2)
        if item is not None:
            itemstr = item.decode("utf-8")
            print("Working on " + itemstr)
            proc_pdbid(itemstr, f"/opt/fixed_pdbs/{itemstr}.pdb.gz")
            q.complete(item)
        else:
            print("Waiting for work")
            sleep(10)
    print("Queue empty, exiting")


if __name__ == "__main__":
    main()
