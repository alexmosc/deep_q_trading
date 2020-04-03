
library(R6)

Train = R6::R6Class(
	'Train'
	, private = list(
		old_pos_price = NULL
		, new_pos_price = NULL
		, old_pos = c(0, 0, 1)
		, q_val_tracker = NULL
		, critic_loss_tracker = numeric()
		, returns_data = NULL
		, deal_count = 0L
		, reward_buffer = numeric()
		, features = NULL
		, Rb = NULL
		, Nn1 = NULL
		, Nn2 = NULL
		, feature_names = NULL
		, lstm_seq_length = NULL
		, buffer_size = NULL
		, S = NULL
		, A = NULL
		, R = NULL
		, SAR = NULL
		, iter = NULL
	)
	, public = list(
		
		act = function()
		{
			
			require(keras)
			require(magrittr)
			require(data.table)
			
			nn_dat <- 
				private$Rb$rb[
					(.N - private$lstm_seq_length + 1):.N
					, paste0(c(
						private$feature_names
						, 'buy'
						, 'sell'
						, 'hold'
						, 'floating'
					), '_next')
					, with = F
					]
			
			## reshape inputs
			
			train_x <- 
				nn_dat %>%
				as.matrix %>%
				t %>%
				as.vector %>%
				reticulate::array_reshape(
					x = .
					, dim = c(1, private$lstm_seq_length, private$Nn1$shape_input)
					, order = 'C'
				)
			
			
			## predict with NN
			
			predict(
				private$Nn1$nn
				, train_x
				) %>%
				as.numeric
			
		}
		
		, reward = function(fl)
		{
			
			if(
				all(private$old_pos == c(0,0,1)) == F
				& all(private$old_pos == private$A) == F
			)
			{
				
				r <- fl
				
			} else {
				
				r <- 0
				
			}
			
			r
			
		}
		
		, update_nn = function(
			batch_size
			, learn_rate
			, discount_factor
		)
		{
			
			require(keras)
			require(data.table)
			require(magrittr)
			
			
			## Get state names
			
			state_names <- 
				c(
					private$feature_names
					, 'buy'
					, 'sell'
					, 'hold'
					, 'floating'
				)
			
			next_state_names <- 
				paste0(
					c(
						private$feature_names
						, 'buy'
						, 'sell'
						, 'hold'
						, 'floating'
					)
				, '_next'
				)
			
			state_names_length <- length(state_names)
			
			
			## randomly load one of two models to generate max q action
			
			which_model <- runif(1) > 0.5
			
			if(
				which_model
			)
			{
				
				primary_nn <- private$Nn1
				
				secondary_nn <- private$Nn2
				
			} else {
				
				secondary_nn <- private$Nn1
				
				primary_nn <- private$Nn2
				
			}
			
			
			## Make indexes to sample train data
			
			sample_indexes <- 
				sample(
					seq(private$lstm_seq_length, private$buffer_size, 1)
					, batch_size
					, replace = F
					, prob = tail(
						private$Rb$rb_priority()
						, private$buffer_size - private$lstm_seq_length + 1
						)
				)
			
			
			## sample experiences to update Q ----
			
			batch <- 
				lapply(
					sample_indexes
					, function(x)
						{
							
							private$Rb$rb[(x - private$lstm_seq_length + 1):x]
							
					}
				) %>%
				rbindlist
			
			
			## approximate q values
			
			critic_train_x <- 
				array_reshape(
					as.vector(t(as.matrix(batch[, state_names, with = F])))
					, dim = c(length(sample_indexes), private$lstm_seq_length, state_names_length)
					, order = "C"
						    )
			
			next_critic_train_x <- 
				array_reshape(
					as.vector(t(as.matrix(batch[, next_state_names, with = F])))
					, dim = c(length(sample_indexes), private$lstm_seq_length, state_names_length)
					, order = "C"
					)
			
			
			## predict with NN
			
			q_values <- 
				predict(
				primary_nn$nn
				, critic_train_x
			)
			
			next_q_values <- 
				predict(
				primary_nn$nn
				, next_critic_train_x
			)
			
			which_next_max_q <- 
				apply(
					next_q_values
					, 1
					, which.max
					)
			
			
			## predict with NN
			
			next_q_values <- predict(
				secondary_nn$nn
				, next_critic_train_x
			)
			
			
			## update critic
			
			made_actions <- batch[
				seq(private$lstm_seq_length, nrow(batch), by = private$lstm_seq_length)
				, .(buy, sell, hold)
				]
			
			got_rewards <- batch[
				seq(private$lstm_seq_length, nrow(batch), by = private$lstm_seq_length)
				, reward
				]
			
			critic_train_y <- array(
				q_values
				, dim = c(length(sample_indexes)
						, 3L)
				)
			
			# insert discounted target
			
			for(
				i in 1:length(sample_indexes)
			)
			{
				critic_train_y[
					i
					, c(1:3)[made_actions[i]==1]
					] <- 
						got_rewards[i] + discount_factor * next_q_values[i, which_next_max_q[i]]
			}
			
			critic_train <- keras::fit(
				primary_nn$nn
				, critic_train_x
				, critic_train_y
				, epochs = 1
				, batch_size = length(sample_indexes)
				, shuffle = F
				, verbose = 0
				, lr = learn_rate
			)
			
			private$critic_loss_tracker <- 
				c(
					private$critic_loss_tracker
					, critic_train$metrics$mean_squared_error[length(critic_train$metrics$mean_squared_error)]
					)
			
			
			## assign models to objects
			
			if(
				which_model
			)
			{
				
				private$Nn1 <- primary_nn
				
			} else {
				
				private$Nn2 <- primary_nn
				
			}
			
			
			## Update priority vector
			
			private$Rb$priority_vec[sample_indexes] <- 
				abs(
					apply(
						critic_train_y - q_values
						, 1
						, sum
						)
					)
			
			
			invisible(self)
			
		}
		
		, stats_train = function()
		{
			
			require(TTR)
			
			par(mfrow = c(3, 2), oma=c(0,0,2,0))
			
			plot(
				log(TTR::SMA(private$critic_loss_tracker, 100)[!is.na(TTR::SMA(private$critic_loss_tracker, 100))])
				, type = 'l'
				, main = '100-average critic loss logarithm'
				, xlab = 'timestep'
				, ylab = ''
			)
			
			plot(private$returns_data[deal == 1, TTR::SMA(return, 100)][!is.na(private$returns_data[deal == 1, TTR::SMA(return, 100)])],
				type = 'l',
				main = '100-deal-average return plot'
				, xlab = 'episode'
				, ylab = ''
			)
			
			plot(private$returns_data[deal == 1, TTR::runSum(return, cumulative = T)],
				type = 'l',
				main = 'cumulative return plot'
				, xlab = 'episode'
				, ylab = ''
			)
			
			plot(tail(private$reward_buffer, 100),
				type = 'l',
				main = 'last 100 rewards'
				, xlab = 'timestep'
				, ylab = ''
			)
			
			plot(private$q_val_tracker[(nrow(private$q_val_tracker) - 100):(nrow(private$q_val_tracker) - 1), V1]
				, type = 'l'
				, col = 'red'
				, main = 'last 100 q values'
				, xlab = 'timestep'
				, ylab = ''
				, ylim = c(
					min(unlist(private$q_val_tracker[(nrow(private$q_val_tracker) - 99):nrow(private$q_val_tracker), .(V1, V2, V3)]))
					, max(unlist(private$q_val_tracker[(nrow(private$q_val_tracker) - 99):nrow(private$q_val_tracker), .(V1, V2, V3)]))
				)
			)
			
			lines(private$q_val_tracker[(nrow(private$q_val_tracker) - 100):(nrow(private$q_val_tracker) - 1), V2]
				 , type = 'l'
				 , col = 'black'
			)
			
			lines(private$q_val_tracker[(nrow(private$q_val_tracker) - 100):(nrow(private$q_val_tracker) - 1), V3]
				 , type = 'l'
				 , col = 'green'
			)
			
			plot(private$features[(private$iter - 99):private$iter, dat]
				, type = 'l'
				, main = 'last 100 quotes'
				, xlab = 'timestep'
				, ylab = ''
			)
			
			mtext(paste0('timestep = ', private$iter), line=0, side=3, outer=TRUE, cex=2)
			
			cat('########## intermediate stats were displayed for iteration', private$iter, '##########', '\n')
			
			invisible(self)
			
		}
		
		, dynamics = function(
			fl
		)
		{
			
			# closing buy trade -----------------------
			
			if(
				all(private$old_pos == c(1,0,0)) == T
				& all(private$A == private$old_pos) == F
			)
			{
				
				private$returns_data <- rbind(
					private$returns_data,
					data.table(
						new_state = c('buy','sell','hold')[private$A == 1]
						, old_state = c('buy','sell','hold')[private$old_pos == 1]
						, return = fl
						, time_step = private$iter
						, price = private$new_pos_price
						, deal = 1
					)
				)
				
				private$deal_count <- private$deal_count + 1L
				
			}
			
			
			## closing sell trade -----------------------
			
			else if(
				all(private$old_pos == c(0,1,0)) == T
				& all(private$A == private$old_pos) == F
			)
			{
				
				private$returns_data <- rbind(
					private$returns_data,
					data.table(
						new_state = c('buy','sell','hold')[private$A == 1]
						, old_state = c('buy','sell','hold')[private$old_pos == 1]
						, return = fl
						, time_step = private$iter
						, price = private$new_pos_price
						, deal = 1
					)
				)
				
				private$deal_count <- private$deal_count + 1L
				
			}

			
			## opening trade from hold -----------------------
			
			else if(
				all(private$old_pos == c(0,0,0)) == T
				& all(private$A == private$old_pos) == F
			)
			{
				
				private$returns_data <- rbind(
					private$returns_data,
					data.table(
						new_state = c('buy','sell','hold')[private$A == 1]
						, old_state = c('buy','sell','hold')[private$old_pos == 1]
						, return = 0
						, time_step = private$iter
						, price = private$new_pos_price
						, deal = 0
					)
				)
				
			}
			
			
			## continuing hold or trade -----------------------
			
			else 
			{
				
				private$returns_data <- rbind(
					private$returns_data,
					data.table(
						new_state = c('buy','sell','hold')[private$A == 1]
						, old_state = c('buy','sell','hold')[private$old_pos == 1]
						, return = 0
						, time_step = private$iter
						, price = private$new_pos_price
						, deal = 0
					)
				)
				
			}
			
			invisible(self)
			
		}
		
		, analyze = function(
			round_factor = 100
			, each_period = 5
		)
		{
			
			require(ggplot2)
			
			train_tise_analyze <- copy(private$returns_data)
			
			train_tise_analyze[, round_index:= round(time_step / round_factor)]
			
			train_tise_analyze[, round_index_residual := round_index %% each_period]
			
			plo <- ggplot() +
				facet_wrap(
					~ round_index, ncol = 3, scales = 'free'
					) +
				geom_line(
					data = train_tise_analyze[
						round_index_residual == 0 & round_index > 1
						]
						, aes(
							x = time_step
							, y = price
							)
					, size = 0.5
					, color = 'blue'
					, alpha = 1
					) +
				geom_point(
					data = train_tise_analyze[
						round_index_residual == 0 & round_index > 1 & new_state == 'buy'
						]
						 , aes(
						 	x = time_step
						 	, y = price
						 	)
					, size = 2
					, shape = 17
					, color = 'green'
					, fill = 'green'
					, alpha = 0.75
					) +
				geom_point(
					data = train_tise_analyze[
						round_index_residual == 0 & round_index > 1 & new_state == 'sell'
						]
						 , aes(
						 	x = time_step
						 	, y = price
						 	)
						 , size = 2
					, shape = 25
					, color = 'red'
					, fill = 'red'
					, alpha = 0.75
						 ) +
				geom_point(
					data = train_tise_analyze[
						round_index_residual == 0 & round_index > 1 & new_state == 'hold'
						]
						 , aes(
						 	x = time_step
						 	, y = price
						 	)
						 , size = 2
					, shape = 15
					, color = 'grey'
					, fill = 'grey'
					, alpha = 0.75
						 ) +
				theme_minimal()
			
			print(plo)
			
			cat('########## training dynamics was plotted after', private$iter, 'iterations ##########', '\n')
			
			invisible(self)
			
		}
		
		, run = function(
			test_mode = F
			, batch_size = 16L
			, discount_factor = 0.99
			, learn_rate = 0.001
			, max_iter = 5000L
			, min_trans_cost = 0.5
			, print_returns_every = 100
			, magic_const = 32
		)
		{
			
			require(data.table)
			require(magrittr)
			
			if(test_mode == F)
			{
				
				private$features <- Dat$features
				
				private$Rb <- Rb
				
				private$Nn1 <- Nn
				
				private$Nn2 <- Nn2
				
				private$feature_names <- Dat$feature_names
				
				private$old_pos_price <- Dat$features[1, dat]
				
				private$lstm_seq_length <- Nn$lstm_seq_length
				
				private$buffer_size <- Rb$buffer_size
				
				private$returns_data <- data.table()
				
				private$SAR <- c(
					unlist(
						Rb$rb[
							.N
							, paste0(
								c(
									Dat$feature_names
									, 'buy'
									, 'sell'
									, 'hold'
									, 'floating'
								)
								, '_next'
							)
							, with = F]
					)
					, 0,0,0,0
				) %>%
					magrittr::set_names(
						c(
							private$feature_names
							, 'buy'
							, 'sell'
							, 'hold'
							, 'floating'
							, "act_buy" 
							, "act_sell" 
							, "act_hold" 
							, "reward"
						)
					)
				
				private$iter <- private$lstm_seq_length + 1L
				
				if(
					max_iter > nrow(private$features)
				)
				{
					max_iter <- nrow(private$features)
				}
				
				
				## training cycle
				
				while(private$iter < max_iter)
				{
					
					## Make state vector
					
					dt <- private$features[private$iter]
					
					private$new_pos_price <- dt[, dat]
					
					fl <- ifelse(
						all(private$old_pos == c(1,0,0))
						, private$new_pos_price - private$old_pos_price - min_trans_cost
						, ifelse(
							all(private$old_pos == c(0,1,0))
							, private$old_pos_price - private$new_pos_price - min_trans_cost
							, 0
						)
					) * magic_const
					
					private$S <- 
						c(
						unlist(dt[, private$feature_names, with = F])
						, private$old_pos
						, fl
						) %>%
						magrittr::set_names(
							c(
								private$feature_names
								, 'buy'
								, 'sell'
								, 'hold'
								, 'floating'
							)
						)
					
					
					## Update replay buffer (one private$iteration lagged)
					
					SAR <- private$SAR
					
					S <- 
						private$S %>%
						magrittr::set_names(
							paste0(c(
								private$feature_names
								, 'buy'
								, 'sell'
								, 'hold'
								, 'floating'
							)
							, '_next'
						)
					)
					
					private$Rb$update_rb(c(SAR, S))
					

					## Make action
					
					q_vals <- self$act()

					private$q_val_tracker <-
						rbind(
							private$q_val_tracker
							, as.data.table(t(q_vals))
						)

					private$A <- as.integer(c(1,2,3) == which.max(q_vals))
					
					
					## Get reward
					
					private$R <- self$reward(fl)
					
					private$reward_buffer <- 
						c(
							private$reward_buffer
							, private$R
						)
					
					private$SAR <-
						c(
							private$S #env
							, private$A #actions
							, private$R #reward
						) %>%
						magrittr::set_names(
							c(
								private$feature_names
								, 'buy'
								, 'sell'
								, 'hold'
								, 'floating'
								, "act_buy" 
								, "act_sell" 
								, "act_hold" 
								, "reward"
							)
						)
					
					
					## Update model
					
					if(
						nrow(private$Rb$rb) == private$buffer_size
					   )
						{
						self$update_nn(
							batch_size
							, learn_rate
							, discount_factor
						)
					}
					
					
					## Track dynamics
					
					self$dynamics(fl)
					
					
					## Update old values
					
					private$old_pos_price <- private$new_pos_price
					
					private$old_pos <- private$A
					
					
					## Print stats
					
					if (
						private$deal_count > 100 & 
						private$iter %% print_returns_every == 0 & 
						length(private$critic_loss_tracker) > 100 & 
						nrow(private$Rb$rb) == private$buffer_size
					)
					{

						self$stats_train()
						
					}
					
					
					## Update index
					
					private$iter <- private$iter + 1
					
				}
				
				
				## Analyze results
				
				self$analyze()
				
				
			} else {
				
				NULL
				
			}
			
		}
		
	)
)
