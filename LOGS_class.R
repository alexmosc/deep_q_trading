
library(R6)

Logs <- R6::R6Class(
	'Data'
	, public = list(
		q_val_tracker = NULL
		, critic_loss_tracker = NULL
		, returns_data = NULL
		, reward_buffer = NULL
		
		, initialize = function()
		{
			
			require(data.table)
			
			self$q_val_tracker <- data.table()
			self$critic_loss_tracker <- numeric()
			self$returns_data <- data.table()
			self$reward_buffer <- numeric()
			
			cat('########## loggin was initialized ##########', '\n')
			
			invisible(self)
			
		}
	)
)
