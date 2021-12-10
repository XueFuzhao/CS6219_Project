# CS6219_Project

We first download environment image from [NGC](https://ngc.nvidia.com/catalog/containers/nvidia:pytorch/tags).
In this project, we use 21.09-py3.

After installed the environment by singularity, we install SRU for fast RNN training and inference.
```
pip install sru
```

reference: https://github.com/asappresearch/sru


After prepared the environment, please use the following code to run the training:

```
bash run_train.sh
```

Use the commands to run the pre-trained models:

```
bash run_model.sh
```

Please note that our code requires GPU to run. We use V100 GPU to achieve the results reported.
