import hashlib

PATH = "GeneratedLabelledFlows/TrafficLabelling /"

with open(PATH + "Discretized_{}.csv".format('CICIDS17'), "r") as f_in, \
        open(PATH + "Discretized_unique_{}.csv".format('CICIDS17'), "w") as f_out:
    seen = set()
    for line in f_in:
        line_hash = hashlib.md5(''.join(line.split(",")[1:-2]).encode()).digest()
        if line_hash not in seen:
            seen.add(line_hash)
            f_out.write(line)
