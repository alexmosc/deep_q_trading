
library(R6)

Logs <- R6::R6Class(
	'Data'
	, public = list(
		q_val_tracker = data.table()
		, critic_loss_tracker = numeric()
		, returns_data = data.table()
		, reward_buffer = numeric()
		
		, initialize = function()
		{
			
			cat('########## loggin was initialized ##########', '\n')
			
			invisible(self)
			
		}
	)
)
