
# clear environment

rm(list = ls()); gc()

setwd('C:/R_study/reinforcement/rl_classes') # set your working directory


## Classes

source('NN_class.R')

source('DATA_class.R')

source('RB_class.R')

source('TRAIN_class.R')


## Data

Dat <- Data$new()

# Dat$synthetic_noise(
# 	noise_sd = 0.1
# 	, n = 20000
# )
# 
# Dat$stock_data(
# 	symbol_name = 'AAPL'
# 	, from = "2000-01-01"
# 	, to = "2020-03-01"
# 	, dat.env = new.env()
# )

Dat$synthetic_signal(
	stepsize = 0.1
	, noise_sd = 0.0
	, noise_sd2 = 0.0
	, n = 20000
	)

Dat$make_features(max_lag_power = 6)


## Double neural networks

Nn <- NN$new(lstm_seq_length = 8L)

Nn$compile_nn(
	loss = 'mse'
	, metrics = 'mse'
	, optimizer = 'adam'
)

Nn2 <- Nn$clone()


## Replay buffer

Rb <- RB$new(
	buffer_size = 512
	, priority_alpha = 0.1
	)

Rb$init_rb()


## Train

Tr <- Train$new()

Tr$run(
	test_mode = F
	, batch_size = 64
	, discount_factor = 0.99
	, learn_rate = 0.001
	, max_iter = 5000
	, min_trans_cost = 0
	, print_returns_every = 100
	, magic_const = 1
)
