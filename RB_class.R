
library(R6)

RB = R6::R6Class(
	'RB'
	, public = list(
		rb = NULL
		, buffer_size = NULL
		, priority_alpha = 0.25
		, priority_vec = NULL
		
		, initialize = function(
			buffer_size
			, priority_alpha
		)
		{
			self$buffer_size <- buffer_size
			
			self$priority_alpha <- priority_alpha
		}
		
		, init_rb = function()
		{
			
			require(data.table)
			require(magrittr)
			
			rb_left <-
				data.table(
					Dat$features[1:Nn$lstm_seq_length, Dat$feature_names, with = F]
					, data.table(
						buy = integer(Nn$lstm_seq_length)
						, sell = integer(Nn$lstm_seq_length)
						, hold = integer(Nn$lstm_seq_length) + 1
						, floating = integer(Nn$lstm_seq_length)
						, act_buy = integer(Nn$lstm_seq_length)
						, act_sell = integer(Nn$lstm_seq_length)
						, act_hold = integer(Nn$lstm_seq_length)
						, reward = integer(Nn$lstm_seq_length)
					)
				)
			
			rb_right <-
				rb_left[2:.N, 1:(length(Dat$feature_names) + 4), with = F]
			
			colnames(rb_right) <-
				paste0(
					colnames(rb_left)[1:(length(Dat$feature_names) + 4)]
					, '_next'
				)
			
			rb_left <-
				rb_left[-.N]
			
			self$rb <- cbind(rb_left, rb_right)
			
			cat('########## replay buffer was initialized ##########', '\n')
			
			invisible(self)
			
		}
		
		, update_rb = function(sars)
		{
			
			require(data.table)
			require(magrittr)
			
			self$rb <-
				rbind(
					self$rb
					, as.data.table(t(sars))
				)
			
			if(
				nrow(self$rb) == self$buffer_size
			)
			{
				
				## Make RB priority
				
				self$priority_vec <- self$rb[, abs(reward)]
				
				cat('########## replay buffer is prefilled ##########', '\n')
				
			}
			
			if(
				nrow(self$rb) > self$buffer_size
			)
			{
				
				self$rb <- self$rb[-1]
				
				self$priority_vec <- 
					c(
						self$priority_vec[-1]
						, 0
					)
				
			}
			
			invisible(self)
			
		}
		
		, rb_priority = function()
		{
			
			priority_rank <- rank(-self$priority_vec, ties.method = 'random')
			
			p <- 1 / priority_rank
			
			p ^ self$priority_alpha / sum(p ^ self$priority_alpha)
			
		}
		
	)
)