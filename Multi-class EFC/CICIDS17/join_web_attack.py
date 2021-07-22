with open("TrafficLabelling /Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", encoding="utf8", errors='ignore') as f_in:
    with open("TrafficLabelling /Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_fixed.csv", "w") as f_out:
        for line in f_in:
            line = line.split(",")
            if line[-1][:10] == 'Web Attack':
                line[-1] = 'Web Attack\n'
            line = ",".join(line)
            f_out.write(line)
