
library(R6)

Data <- R6::R6Class(
	'Data'
	, public = list(
		dat_vec = NULL
		, features = NULL
		, feature_names = NULL
		
		, stock_data = function(
			symbol_name = 'AAPL'
			, from = "2000-01-01"
			, to = "2020-03-01"
			, dat.env = new.env()
		)
		{

			require(data.table)
			require(quantmod)
			require(magrittr)
			
			quantmod::getSymbols.yahoo(
				symbol_name
				, from = Sys.Date() - 5000
				, to = Sys.Date() - 1
				, env = dat.env
			)
			
			self$dat_vec <-
				dat.env %>%
				as.list %>%
				as.data.table %>%
				`[[`(1)
			
			plot(
				self$dat_vec[1:100]
				, type = 'l'
				, main = 'Example of your data'
			)
			
			cat('########## stock data were downloaded ##########', '\n')
			
			invisible(self)
			
		}
		
		, synthetic_signal = function(
			stepsize = 0.1
			, noise_sd = 0.1
			, noise_sd2 = 0.1
			, n = 10000
		)
		{
			
			set.seed(0)
			
			tise <- sin(seq(stepsize, n*stepsize, stepsize)) + 
				rnorm(n, 0, noise_sd) + 
				cumsum(rnorm(n, 0, noise_sd2))
			
			self$dat_vec <- tise + abs(min(tise)) + 0.1
			
			plot(
				self$dat_vec[1:100]
				, type = 'l'
				, main = 'Example of your data'
			)
			
			cat('########## synthetic signal was generated ##########', '\n')
			
			invisible(self)
			
		}
		
		, synthetic_noise = function(
			noise_sd = 0.1
			, n = 10000
		)
		{
			
			set.seed(0)
			
			self$dat_vec <- cumsum(rnorm(n, 0, noise_sd))
			
			plot(
				self$dat_vec[1:100]
				, type = 'l'
				, main = 'Example of your data'
			)
			
			cat('########## synthetic noise was generated ##########', '\n')
			
			invisible(self)
			
		}
		
		, make_features = function(
			max_lag_power = 8
			)
		{
			
			require(data.table)
			require(magrittr)
			
			# difference of prices
			
			lags <- unique(round(2 ^ seq(0, max_lag_power, 0.5)))
			
			dt <- 
				data.table(
					dat = self$dat_vec
					) %>%
				.[
				, (paste('lag_', lags, sep = '')) := lapply(
						lags
						, function(x) log(dat / shift(dat, x))
					)
				] %>%
				na.omit
			
			self$features <- dt
			
			self$feature_names <- colnames(dt)[colnames(dt) != 'dat']
			
			cat('########## data.table with input features was created ##########', '\n')
			
			invisible(self)
			
		}
	)
)