import random
import argparse
import C

ALPHABET = 'ACGT'


# generate a noise strand based on the true strand
def gen_noise_sample(truth, sub_p, del_p, ins_p):
    res = []
    for w in truth:
        r = random.random()
        if r < sub_p:
            res.append(random.choice(ALPHABET))
        elif r < sub_p + ins_p:
            res.append(random.choice(ALPHABET))
            res.append(w)
        elif r > sub_p+ins_p+del_p:
            res.append(w)
    return ''.join(res)


# generate a random strand with fixed length
def gen_strand(length):
	res = []
	for i in range(length):
		res.append(random.choice(ALPHABET))
	return ''.join(res)


# generate a cluster
def gen_cluster(length, n, sub_p, del_p, ins_p, seed=0):
	random.seed(seed)
	res = {}
	res['truth'] = gen_strand(length)
	cluster = []
	for i in range(n):
		cluster.append(gen_noise_sample(res['truth'], sub_p, del_p, ins_p))
	res['cluster'] = cluster
	return res


# positional error
def positional_error(truth, result):
	error = [0] * len(truth)
	for i in range(len(truth)):
		if i < len(result) and truth[i] != result[i]:
			error[i] = 1
	return error


# edit distance
def edit_distance(truth, result):
	return C.editdistance(truth, result)


if __name__ == '__main__':
	data = gen_cluster(100, 10, 0.01, 0.01, 0.01)
	# print(data)
	for s in data['cluster']:
		print(edit_distance(data['truth'], s))
