import random
import argparse
# import C

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
def edit_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


if __name__ == '__main__':
        number_of_samples = 12800
        output_data_file_path = '../dataset/dev_data.txt'
        output_label_file_path = '../dataset/dev_label.txt'
        sequence_len = 120
        
        #data_list = []
        #label_list = []
        f_data = open(output_data_file_path, "w")
        f_label = open(output_label_file_path, "w")

        for i in range(number_of_samples):
            combined_data = gen_cluster(sequence_len, 10, 0.03, 0.03, 0.03, i)
            data = combined_data['cluster']
            label = combined_data['truth']
            for j in range(len(data)):
                f_data.write(data[j])
                f_data.write('\n')
            f_data.write('\n')
            f_label.write(label)
            f_label.write('\n')
            f_label.write('\n')
        f_data.close()
        f_label.close()





        #data = gen_cluster(100, 10, 0.01, 0.01, 0.01)
	#print(data)
	#for s in data['cluster']:
        #print(edit_distance(data['truth'], s))
