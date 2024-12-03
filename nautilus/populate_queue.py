#!/usr/bin/env python3

"""
Populate a redis list via kubectl exec

Check the state of the list:
    echo lrange $QUEUE_NAME 0 -1 | kubectl exec --stdin $POD_NAME -- redis-cli

Clear the list:
    echo del $QUEUE_NAME | kubectl exec --stdin $POD_NAME -- redis-cli
"""

import subprocess
import sys

FIRST_ID_INDEX=0
SAMPLE_N=11000
SHUFFLED_IDS_FILENAME="all_pdb_ids_shuffled.txt"
QUEUE_NAME="job2"
POD_NAME="deploy/pdbscan-jm-redis"

def main():
    with open(SHUFFLED_IDS_FILENAME, 'r') as f:
        shuffled_pdb_ids = [s[:-1] for s in f.readlines()]
    sample_pdb_ids = shuffled_pdb_ids[FIRST_ID_INDEX:FIRST_ID_INDEX+SAMPLE_N]
    redis_command = f"rpush {QUEUE_NAME} {' '.join(sample_pdb_ids)}"
    if "--dry-run" in sys.argv:
        print(redis_command)
    else:
        subprocess.run(
            args=[
                "kubectl",
                "exec",
                "--stdin",
                POD_NAME,
                "--",
                "redis-cli",
            ],
            input=redis_command,
            text=True,
            check=True,
        )



if __name__ == "__main__":
    main()
