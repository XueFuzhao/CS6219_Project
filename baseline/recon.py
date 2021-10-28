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
    pool = mp.Pool(20)
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
    parser.add_argument('--N', type=int, default=10**3)
    parser.add_argument('--L', type=int, default=120)
    parser.add_argument('--n', type=int, default=10)
    parser.add_argument('--p', type=float, default=0.01)
    parser.add_argument('--path', type=str, default='./log/')
    args = parser.parse_args()

    N = args.N
    L = args.L
    n = args.n
    p = args.p

    path = args.path
    if not os.path.exists(path):
        os.mkdir(path)
    filesuff = '_N{}_L{}_n{}_p{}.txt'.format(N, L, n, p)

    strands = []
    clusters = []

    for i in range(N):
        data = data_generator.gen_cluster(L, n, p, p, p, i)
        strands.append(data['truth'])
        clusters.append(data['cluster'])

    f = open(path + 'strands' + filesuff, 'w')
    for strand in strands:
        f.write(strand + '\n')
    f.close()

    start = time.time()
    results = reconstruct_clusters_mp(clusters)
    end = time.time()
    print("Trace reconstruction took " + str(end-start) +" seconds")

    f = open(path + 'results' + filesuff, 'w')
    for result in results:
        f.write(result + '\n')
    f.close()

    position_errors = [0]*L
    for i in range(N):
        error_vec = data_generator.positional_error(strands[i], results[i])
        position_errors = list(map(lambda x, y: x + y, position_errors, error_vec))

    f = open(path + 'stats' + filesuff, 'w')
    f.write('N: {}, L: {}, n: {}, p: {}\n'.format(N, L, n, p))
    f.write('time: ' + str(end - start) + ' seconds\n')
    f.write('positional error:\n')
    for i in range(L):
        f.write(str(position_errors[i]) + ' ')
    f.close()
