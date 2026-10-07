"""Atomic publication with bounded retries for transient Windows reader locks."""
import json,time

def replace_with_retry(source,target,timeout=15):
    deadline=time.monotonic()+timeout
    delay=.025
    while True:
        try:
            source.replace(target)
            return
        except PermissionError:
            remaining=deadline-time.monotonic()
            if remaining<=0:raise
            time.sleep(min(delay,remaining));delay=min(.5,delay*2)

def atomic_json(path,data):
    # Temporary writes must not be visible to readers enumerating '*.json'.
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    replace_with_retry(temp,path)
