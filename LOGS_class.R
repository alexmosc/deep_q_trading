
library(R6)

Logs <- R6::R6Class(
	'Data'
	, public = list(
		q_val_tracker = NULL
		, critic_loss_tracker = NULL
		, returns_data = NULL
		, reward_buffer = NULL
		, deal_count = NULL
		
		, initialize = function()
		{
			
			require(data.table)
			
			self$q_val_tracker <- data.table()
			self$critic_loss_tracker <- numeric()
			self$returns_data <- data.table()
			self$reward_buffer <- numeric()
			self$deal_count <- 0
			
			cat('########## loggin was initialized ##########', '\n')
			
			invisible(self)
			
		}
		
		, stats_train = function(
			iter
			, features
		)
		{
			
			require(TTR)
			
			par(mfrow = c(3, 2), oma=c(0,0,2,0))
			
			plot(
				log(TTR::SMA(self$critic_loss_tracker, 100)[!is.na(TTR::SMA(self$critic_loss_tracker, 100))])
				, type = 'l'
				, main = '100-average critic loss logarithm'
				, xlab = 'timestep'
				, ylab = ''
			)
			
			plot(self$returns_data[deal == 1, TTR::SMA(return, 100)][!is.na(self$returns_data[deal == 1, TTR::SMA(return, 100)])],
				type = 'l',
				main = '100-deal-average return plot'
				, xlab = 'episode'
				, ylab = ''
			)
			
			plot(self$returns_data[deal == 1, TTR::runSum(return, cumulative = T)],
				type = 'l',
				main = 'cumulative return plot'
				, xlab = 'episode'
				, ylab = ''
			)
			
			plot(tail(self$reward_buffer, 100),
				type = 'l',
				main = 'last 100 rewards'
				, xlab = 'timestep'
				, ylab = ''
			)
			
			plot(self$q_val_tracker[(nrow(self$q_val_tracker) - 100):(nrow(self$q_val_tracker) - 1), V1]
				, type = 'l'
				, col = 'red'
				, main = 'last 100 q values'
				, xlab = 'timestep'
				, ylab = ''
				, ylim = c(
					min(unlist(self$q_val_tracker[(nrow(self$q_val_tracker) - 99):nrow(self$q_val_tracker), .(V1, V2, V3)]))
					, max(unlist(self$q_val_tracker[(nrow(self$q_val_tracker) - 99):nrow(self$q_val_tracker), .(V1, V2, V3)]))
				)
			)
			
			lines(self$q_val_tracker[(nrow(self$q_val_tracker) - 100):(nrow(self$q_val_tracker) - 1), V2]
				 , type = 'l'
				 , col = 'black'
			)
			
			lines(self$q_val_tracker[(nrow(self$q_val_tracker) - 100):(nrow(self$q_val_tracker) - 1), V3]
				 , type = 'l'
				 , col = 'green'
			)
			
			plot(features[(iter - 99):iter, dat]
				, type = 'l'
				, main = 'last 100 quotes'
				, xlab = 'timestep'
				, ylab = ''
			)
			
			mtext(paste0('timestep = ', iter), line=0, side=3, outer=TRUE, cex=2)
			
			cat('########## intermediate stats were displayed for iteration', iter, '##########', '\n')
			
			invisible(self)
			
		}
		
		, dynamics = function(
			fl
			, new_pos_price
			, iter
			, old_pos
			, A
		)
		{
			
			# closing buy trade -----------------------
			
			if(
				all(old_pos == c(1,0,0)) == T
				& all(A == old_pos) == F
			)
			{
				
				self$returns_data <- rbind(
					self$returns_data,
					data.table(
						new_state = c('buy','sell','hold')[A == 1]
						, old_state = c('buy','sell','hold')[old_pos == 1]
						, return = fl
						, time_step = iter
						, price = new_pos_price
						, deal = 1
					)
				)
				
				self$deal_count <- self$deal_count + 1L
				
			}
			
			
			## closing sell trade -----------------------
			
			else if(
				all(old_pos == c(0,1,0)) == T
				& all(A == old_pos) == F
			)
			{
				
				self$returns_data <- rbind(
					self$returns_data,
					data.table(
						new_state = c('buy','sell','hold')[A == 1]
						, old_state = c('buy','sell','hold')[old_pos == 1]
						, return = fl
						, time_step = iter
						, price = new_pos_price
						, deal = 1
					)
				)
				
				self$deal_count <- self$deal_count + 1L
				
			}
			
			
			## opening trade from hold -----------------------
			
			else if(
				all(old_pos == c(0,0,1)) == T
				& all(A == old_pos) == F
			)
			{
				
				self$returns_data <- rbind(
					self$returns_data,
					data.table(
						new_state = c('buy','sell','hold')[A == 1]
						, old_state = c('buy','sell','hold')[old_pos == 1]
						, return = 0
						, time_step = iter
						, price = new_pos_price
						, deal = 0
					)
				)
				
			}
			
			
			## continuing hold or trade -----------------------
			
			else 
			{
				
				self$returns_data <- rbind(
					self$returns_data,
					data.table(
						new_state = c('buy','sell','hold')[A == 1]
						, old_state = c('buy','sell','hold')[old_pos == 1]
						, return = 0
						, time_step = iter
						, price = new_pos_price
						, deal = 0
					)
				)
				
			}
			
			invisible(self)
			
		}
		
		, analyze = function(
			round_factor = 100
		)
		{
			
			require(ggplot2)
			
			each_period <- round(nrow(self$returns_data) / round_factor / 12)
			
			train_tise_analyze <- copy(self$returns_data)
			
			train_tise_analyze[, round_index:= round(time_step / round_factor)]
			
			train_tise_analyze[, round_index_residual := round_index %% each_period]
			
			plo <- ggplot() +
				facet_wrap(
					~ round_index
					, ncol = 4
					, scales = 'free'
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
			
			cat('########## training dynamics was plotted ##########', '\n')
			
			invisible(self)
			
		}
	)
)
