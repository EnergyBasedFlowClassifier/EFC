import numpy as np
import pandas as pd
import hashlib

with open("TrafficLabelling /Pre_processed.csv", "r") as f_in:
    with open("TrafficLabelling /Pre_processed_unique.csv", "w") as f_out:
        seen = set()
        for line in f_in:
            line_windex = line.split(",")
            line_windex = ",".join(line_windex)
            line_hash = hashlib.md5(line_windex.encode()).digest()
            if line_hash not in seen:
                seen.add(line_hash)
                f_out.write(line)
