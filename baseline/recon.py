import subprocess
import random
import argparse
import sys
import os
import multiprocessing as mp
import shutil
import time
import data_generator

bases = ['A', 'C', 'G', 'T']


def majority(list):
    return max(set(list), key=list.count)


def refine_majority(clu, i):
    ems = []
    for ele in clu:
        if len(ele) > i:
            ems.append(ele[i])
    if len(ems) == 0:
        ems.append(random.choice(bases))
    return ems


def recover_strand(cluster, strand_len):
    ans = ''
    recovered = ''

    for i in range(0, strand_len - 1):
        ch = majority(refine_majority(cluster, i))

        for j in range(len(cluster)):

            if len(cluster[j]) == i:
                cluster[j] += ch

            if cluster[j][i] != ch:

                ch2 = majority(refine_majority(cluster, i + 1))

                ch3_flag = -1
                if i + 2 < strand_len:
                    ch3_flag = 1
                    ch3 = majority(refine_majority(cluster, i + 2))

                ch4_flag = -1
                if i + 3 < strand_len:
                    ch4_flag = 1
                    ch4 = majority(refine_majority(cluster, i + 3))

                ch5_flag = -1
                if i + 4 < strand_len:
                    ch5_flag = 1
                    ch5 = majority(refine_majority(cluster, i + 4))

                if len(cluster[j]) > i + 2:
                    if cluster[j][i] == ch2 and (ch3_flag == -1 or cluster[j][i + 1] == ch3):  # erasure error
                        cluster[j] = cluster[j][:i] + ch + cluster[j][i:]

                    elif cluster[j][i + 1] == ch and cluster[j][i + 2] == ch2:  # insertion error
                        cluster[j] = cluster[j][:i] + cluster[j][i + 1:]

                    elif cluster[j][i + 1] == ch2 and (ch3_flag == -1 or cluster[j][i + 2] == ch3):  # subs
                        cluster[j] = cluster[j][:i] + ch + cluster[j][i + 1:]

                    elif cluster[j][i + 1] != ch2:

                        if cluster[j][i] == ch3 and (ch4_flag == -1 or cluster[j][i + 1] == ch4):  # erasure
                            cluster[j] = cluster[j][:i] + ch + ch2 + cluster[j][i:]

                        elif len(cluster[j]) > i + 3:
                            if cluster[j][i + 2] == ch3 and (ch4_flag == -1 or cluster[j][i + 3] == ch4):  # subs
                                cluster[j] = cluster[j][:i] + ch + ch2 + cluster[j][i + 1:]

                            elif cluster[j][i + 2] == ch and cluster[j][i + 3] == ch2:  # insertion
                                cluster[j] = cluster[j][:i] + cluster[j][i + 2:]

                            elif cluster[j][i + 1] == ch3 and (ch4_flag == -1 or cluster[j][i + 2] == ch4):
                                cluster[j] = cluster[j][:i] + ch + ch2 + cluster[j][i + 1:]

                            elif cluster[j][i + 1] == ch3 and (ch4_flag == -1 or cluster[j][i + 3] == ch4):
                                cluster[j] = cluster[j][:i] + ch + ch2 + cluster[j][i + 1:]

                            elif cluster[j][i + 2] == ch2 and cluster[j][i + 3] == ch3:
                                cluster[j] = cluster[j][:i] + ch + cluster[j][i + 2:]

                            elif cluster[j][i] == ch3 and (ch4_flag == -1 or cluster[j][i + 3] == ch4):
                                cluster[j] = cluster[j][:i] + ch + ch2 + ch3 + cluster[j][i + 3:]

                            elif cluster[j][i + 2] != ch3:

                                if cluster[j][i] == ch4 and (ch5_flag == -1 or cluster[j][i + 1] == ch5):  # erasure
                                    cluster[j] = cluster[j][:i] + ch + ch2 + ch3 + cluster[j][i:]

                                elif len(cluster[j]) > i + 4:
                                    if cluster[j][i + 3] == ch4 and (
                                            ch5_flag == -1 or cluster[j][i + 4] == ch5):  # subs
                                        cluster[j] = cluster[j][:i] + ch + ch2 + ch3 + cluster[j][i + 1:]

                                    elif cluster[j][i + 3] == ch and cluster[j][i + 4] == ch2:  # insertion
                                        cluster[j] = cluster[j][:i] + cluster[j][i + 3:]

                elif len(cluster[j]) == i + 2:
                    if cluster[j][i] == ch2:  # erasure error
                        cluster[j] = cluster[j][:i] + ch + cluster[j][i:]

                    elif cluster[j][i + 1] == ch2:  # subs
                        cluster[j] = cluster[j][:i] + ch + cluster[j][i + 1:]

                    elif cluster[j][i + 1] == ch:  # insertion error
                        cluster[j] = cluster[j][:i] + cluster[j][i + 1:]

                    else:
                        cluster[j] = cluster[j][:i] + ch

        recovered += ch
        ans = ans[:0] + recovered

    last_ch = majority(refine_majority(cluster, strand_len - 1))
    ans += last_ch

    return ans

L = 120
def reconstruct(cluster):
    strand_num = len(cluster)

    rev_cluster = []

    for i in range(0, len(cluster)):
        rev_cluster.append(cluster[i][::-1])

    mj = recover_strand(cluster, L)
    rev_mj = recover_strand(rev_cluster, L)

    rev_rev_mj = rev_mj[::-1]
    mj = mj[0:int(L / 2) - 1] + rev_rev_mj[int(L / 2) - 1:L]

    return mj


def reconstruct_clusters_mp(clusters):
    print('cpu_count: ', mp.cpu_count())
    pool = mp.Pool(24)
    reconstructed_strands = pool.map_async(reconstruct, clusters).get()
    pool.close()
    return reconstructed_strands


def reconstruct_clusters_sq(clusters):
    reconstructed_strands = []
    for x in clusters:
        reconstructed_strands.append(reconstruct(x))
    return reconstructed_strands


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DNA consensus')
    parser.add_argument('--L', type=int, default=120)
    parser.add_argument('--f', type=str)
    parser.add_argument('--log', type=str, default='./log/')
    parser.add_argument('--path', type=str, default='./')
    args = parser.parse_args()

    L = args.L
    fpath = args.path
    fname = args.f
    logpath = args.log
    if not os.path.exists(logpath):
        os.mkdir(logpath)

    fdata_name = fpath + 'test_data_' + fname + '.txt'
    flabel_name = fpath + 'test_label_' + fname + '.txt'

    strands = []
    clusters = []

    fdata = open(fdata_name, 'r')
    flabel = open(flabel_name, 'r')

    blanklines = 0
    while True:
        line = flabel.readline().strip('\n')
        if line == '':
            blanklines += 1
            if blanklines > 5:
                break
            continue
        blanklines = 0
        strands.append(line)
    cl = []

    blanklines = 0
    while True:
        line = fdata.readline().strip('\n')
        if line == '':
            blanklines += 1
            if blanklines > 5:
                break
            if len(cl) > 0:
                clusters.append(cl)
                cl = []
            continue
        blanklines = 0
        # print(line)
        cl.append(line)

    print(len(strands), len(clusters))

    start = time.time()
    results = reconstruct_clusters_mp(clusters)
    end = time.time()
    print("Trace reconstruction took " + str(end-start) +" seconds")

    strand_acc = 0
    errs = 0
    for i in range(len(strands)):
        err = sum(data_generator.positional_error(strands[i], results[i]))
        errs += err
        if err == 0:
            strand_acc += 1

    print('Base-level: ', 1.0 - errs / len(strands) / 120)
    print('Strand-level: ', strand_acc / len(strands))

    flog = open(logpath + fname + '.log', 'w')
    flog.write('Base-level acc: ' + str(1.0 - errs / len(strands) / 120) + '\n')
    flog.write('Strand-level acc: ' + str(strand_acc / len(strands)) + '\n')
    flog.write('Throughput: ' + str(len(strands) / (end-start)) + '\n')
