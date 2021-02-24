with open("TrafficLabelling /Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv", encoding="utf8", errors='ignore') as f_in:
    with open("TrafficLabelling /Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX_fixed.csv", "w") as f_out:
        for line in f_in:
            line = line.split(",")
            if line[-1][-12:] == 'Brute Force\n':
                line[-1] = 'Web Attack Brute Force\n'
            if line[-1][-14:] == 'Sql Injection\n':
                line[-1] = 'Web Attack Sql Injection\n'
            if line[-1][-4:] == 'XSS\n':
                line[-1] = 'Web Attack XSS\n'
            line = ",".join(line)
            f_out.write(line)
