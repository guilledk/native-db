import os
import time
import signal
import threading

import pdbp
import psutil
import pytest


@pytest.fixture(scope='session', autouse=True)
def cap_memory():
    '''
    Ensure the test process doesnt eat up all memory (useful when debugging
    malformed `polars` queries.

    Launched a bg thread that kills the entire process group if more than
    70% of system memory is consumed by the test.

    '''
    # compute absolute RSS limit (70% of total RAM)
    limit = int(psutil.virtual_memory().total * 0.7)
    # ensure we’re the leader of our own process‐group
    os.setsid()

    def watchdog():
        me = psutil.Process(os.getpid())
        while True:
            try:
                mem = me.memory_info().rss
                if mem > limit:
                    # kill the entire group we created above
                    os.killpg(os.getpgrp(), signal.SIGKILL)
                    print(
                        'process group killed by cap_memory fixture!\n'
                        f'had {mem:,} bytes in use and configured limit is '
                        f'{limit:,} bytes'
                    )
                time.sleep(0.1)
            except Exception:
                break

    # start background monitor thread (daemon so it dies with the process)
    t = threading.Thread(target=watchdog, daemon=True)
    t.start()
    yield
