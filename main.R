
# clear environment

rm(list = ls()); gc()

setwd('C:/R_study/reinforcement/rl_classes') # set your working directory


## Classes

source('NN_class.R')

source('DATA_class.R')

source('RB_class.R')

source('LOGS_class.R')

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
	, noise_sd = 0.1
	, noise_sd2 = 0.1
	, n = 50000
	)

Dat$make_features(max_lag_power = 5L)


## Double neural networks

Nn <- NN$new(lstm_seq_length = 3L)

Nn$compile_nn(
	loss = 'mse'
	, metrics = 'mse'
	, optimizer = 'adam'
)

Nn2 <- Nn$clone()


## Replay buffer

Rb <- RB$new(
	buffer_size = 1024
	, priority_alpha = 0.5
	)

Rb$init_rb()


## Logging

Log <- Logs$new()


## Train

Tr <- Train$new()

Tr$run(
	test_mode = FALSE
	, batch_size = 4
	, discount_factor = 0.99
	, learn_rate = 0.1
	, max_iter = 10000
	, min_trans_cost = 0
	, print_returns_every = 500
	, lr_anneal = TRUE
)


## Save NN

Nn$save()
