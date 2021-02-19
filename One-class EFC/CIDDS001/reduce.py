import numpy as np
import pandas as pd
import sys
import shutil
import os

def reduce_week1():
    counter_port = 0
    counter_benign = 0
    counter_ping = 0
    counter_dos = 0
    counter_brute = 0
    with open('CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week1.csv', "r") as fl:
        for line in fl:
            raw = line.split(",")[:-1]
            a = tuple(np.array(raw))
            with open("CIDDS-001/Reduced/OpenStack/week1.csv", "a") as fl2:
                if len(raw) > 0 and raw[-3]=='normal' and counter_benign <= 331763:
                    counter_benign += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if len(raw) > 0 and raw[-2]=='portScan' and counter_port <= 180000:
                    counter_port += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if len(raw) > 0 and raw[-2]=='pingScan' and counter_ping <= 3359:
                    counter_ping += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if len(raw) > 0 and raw[-2]=='dos' and counter_dos <= 331763:
                    counter_port += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if len(raw) > 0 and raw[-2]=='bruteForce' and counter_brute <= 1626:
                    counter_ping += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if counter_benign >= 331763 and counter_port >= 180000 and counter_ping >= 3359 and counter_dos >= 331763 and counter_brute >= 1626:
                    break


def reduce_week2():
    counter_port = 0
    counter_benign = 0
    counter_ping = 0
    counter_dos = 0
    counter_brute = 0
    with open('CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week2.csv', "r") as fl:
        for line in fl:
            raw = line.split(",")[:-1]
            a = tuple(np.array(raw))
            with open("CIDDS-001/Reduced/OpenStack/week2.csv", "a") as fl2:
                if len(raw) > 0 and raw[-3]=='normal' and counter_benign <= 341658:
                    counter_benign += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if len(raw) > 0 and raw[-2]=='portScan' and counter_port <= 80000:
                    counter_port += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if len(raw) > 0 and raw[-2]=='pingScan' and counter_ping <= 2731:
                    counter_ping += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if len(raw) > 0 and raw[-2]=='dos' and counter_dos <= 341658:
                    counter_port += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if len(raw) > 0 and raw[-2]=='bruteForce' and counter_brute <= 3366:
                    counter_ping += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")
                if counter_benign >= 341658 and counter_port >= 80000 and counter_ping >= 2731 and counter_dos >= 341658 and counter_brute >= 3366:
                    break

def reduce_week3():
    counter_port = 0
    counter_benign = 0
    counter_ping = 0
    counter_dos = 0
    counter_brute = 0
    with open('CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week3.csv', "r") as fl:
        for line in fl:
            raw = line.split(",")[:-1]
            a = tuple(np.array(raw))
            with open("CIDDS-001/Reduced/OpenStack/week3.csv", "a") as fl2:
                if len(raw) > 0 and raw[-3]=='normal' and counter_benign <= 634978:
                    counter_benign += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")

                if counter_benign >= 634978:
                    break


def reduce_week4():
    counter_port = 0
    counter_benign = 0
    counter_ping = 0
    counter_dos = 0
    counter_brute = 0
    with open('CIDDS-001/traffic/OpenStack/CIDDS-001-internal-week4.csv', "r") as fl:
        for line in fl:
            raw = line.split(",")[:-1]
            a = tuple(np.array(raw))
            with open("CIDDS-001/Reduced/OpenStack/week4.csv", "a") as fl2:
                if len(raw) > 0 and raw[-3]=='normal' and counter_benign <= 617589:
                    counter_benign += 1
                    for word in a:
                        fl2.write(word + ",")
                    fl2.write("\n")

                if counter_benign >= 617589:
                    break




#
#
# def reduce_week1():
#     counter_unk = 0
#     counter_benign = 0
#     counter_susp = 0
#     with open('CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week1.csv', "r") as fl:
#         for line in fl:
#             raw = line.split(",")[:-1]
#             a = tuple(np.array(raw))
#             with open("CIDDS-001/Reduced/ExternalServer/week1.csv", "a") as fl2:
#                 if len(raw) > 0 and raw[-3]=='normal' and counter_benign <= 40000:
#                     counter_benign += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if len(raw) > 0 and raw[-3]=='unknown' and counter_unk <= 15000:
#                     counter_unk += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if len(raw) > 0 and raw[-3]=='suspicious' and counter_susp <= 40000:
#                     counter_susp += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if counter_benign >= 40000 and counter_unk >= 15000 and counter_susp >= 40000:
#                     break
#
#
# def reduce_week2():
#     counter_susp = 0
#     counter_benign = 0
#     counter_unk = 0
#     with open('CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week2.csv', "r") as fl:
#         for line in fl:
#             raw = line.split(",")[:-1]
#             a = tuple(np.array(raw))
#             with open("CIDDS-001/Reduced/ExternalServer/week2.csv", "a") as fl2:
#                 if len(raw) > 0 and raw[-3]=='normal' and counter_benign <= 25000:
#                     counter_benign += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if len(raw) > 0 and raw[-3]=='unknown' and counter_unk <= 9000:
#                     counter_unk += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if len(raw) > 0 and raw[-3]=='suspicious' and counter_susp <= 25000:
#                     counter_susp += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if counter_benign >= 25000 and counter_unk >= 9000 and counter_susp >= 25000:
#                     break
# def reduce_week3():
#     counter_susp = 0
#     counter_benign = 0
#     counter_unk = 0
#     with open('CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week3.csv', "r") as fl:
#         for line in fl:
#             raw = line.split(",")[:-1]
#             a = tuple(np.array(raw))
#             with open("CIDDS-001/Reduced/ExternalServer/week3.csv", "a") as fl2:
#                 if len(raw) > 0 and raw[-3]=='normal' and counter_benign <= 6000:
#                     counter_benign += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if len(raw) > 0 and raw[-3]=='unknown' and counter_unk <= 30000:
#                     counter_unk += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if len(raw) > 0 and raw[-3]=='suspicious' and counter_susp <= 40000:
#                     counter_susp += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if counter_benign >= 6000 and counter_unk >= 30000 and counter_susp >= 40000:
#                     break
# def reduce_week4():
#     counter_susp = 0
#     counter_benign = 0
#     counter_unk = 0
#     with open('CIDDS-001/traffic/ExternalServer/CIDDS-001-external-week4.csv', "r") as fl:
#         for line in fl:
#             raw = line.split(",")[:-1]
#             a = tuple(np.array(raw))
#             with open("CIDDS-001/Reduced/ExternalServer/week4.csv", "a") as fl2:
#                 if len(raw) > 0 and raw[-3]=='normal' and counter_benign <= 50000:
#                     counter_benign += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if len(raw) > 0 and raw[-3]=='unknown' and counter_unk <= 18000:
#                     counter_unk += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if len(raw) > 0 and raw[-3]=='suspicious' and counter_susp <= 50000:
#                     counter_susp += 1
#                     for word in a:
#                         fl2.write(word + ",")
#                     fl2.write("\n")
#                 if counter_benign >= 50000 and counter_unk >= 18000 and counter_susp >= 50000:
#                     break
reduce_week1()
reduce_week2()
# reduce_week3()
# reduce_week4()
