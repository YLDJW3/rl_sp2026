# Data
state_dim = 5
output_dim = 2
chunk_size = 8
batch_size = 128
data_size = 180
num_epochs = 400

# MSE policy
1. Use a multi-layer perceptron (MLP) network with two hidden layers, use ReLU as activation function, and use Adam optimizer with learning rate 1e-3
2. Training and inference share the same process

# Flow matching policy
1. Use a multi-layer perceptron (MLP) network with two hidden layers, use ReLU as activation function, and use Adam optimizer with learning rate 1e-3
2. Training and inference have different processes
    During training, sample time steps **uniformly** from [0, 1]
    During inference, given number of steps N, calculate time steps as $t_i = \frac{i}{N}$ for $i=0, 1, ..., N-1$  


# Result
1. https://wandb.ai/yangzf23-independent-developer/hw1-imitation/runs/ht9h3adi?nw=nwuseryangzf23
2. `reward=0.65` at `step=70_000`