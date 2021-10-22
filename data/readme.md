# Data generator

File: ```data_genereator.py```

Methods:

1. Cluster generator: `gen_cluster(length, n, sub_p, del_p, ins_p, seed=0)`

    Generates a strand cluster with the underlying strand.    
    `length`: length of the underlying strand.    
    `n`: number of noise strands in the cluster.    
    `sub_p, del_p, ins_p`: probablity of substitution, deletion, insersion.   
    `seed`: random seed.    
    **output:** a dictionary with two components: `'truth'` and `'cluster'`, correspoding to the underlying strand and a list of `n` noise strands.


2. Position-level error: `positional_error(truth, result)`

    Computes the position-level error between `truth` and `result`.   
    `truth`: the underlying strand.   
    `result`: the result of consensus finding algorithms.   
    **output:** a binary vector with length `len(truth)`, where `1` stands for mismatching.

3. Strand-level error: `edit_distance(truth, result)`

    Computes the strand-level error between `truth` and `result`.   
    `truth`: the underlying strand.   
    `result`: the result of consensus finding algorithms.   
    **output:** a interger number stands for the edit distance between `truth` and `result`.    
    **note:** pybind11 is used.
