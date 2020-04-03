
library(R6)

NN <- R6::R6Class(
	'NN'
	, public = list(
		nn = NULL
		, shape_output = 3L
		, shape_input = NULL
		, lstm_seq_length = NULL
		
		, initialize = function(lstm_seq_length = 4L)
		{
			
			self$shape_input <- length(Dat$feature_names) + 4L
			
			self$lstm_seq_length <- lstm_seq_length
			
		}
		
		, compile_nn = function(
			loss = 'mse'
			, metrics = 'mse'
			, optimizer = 'adam'
		)
		{
			
			require(keras)
			require(magrittr)
			
			a_CustomLayer <- R6::R6Class(
				"CustomLayer"
				, inherit = KerasLayer
				, public = list(
					
					call = function(x, mask = NULL) {
						x - k_mean(x, axis = 2, keepdims = T)
					}
					
				)
			)
			
			a_normalize_layer <- function(object) {
				create_layer(a_CustomLayer, object, list(name = 'a_normalize_layer'))
			}
			
			v_CustomLayer <- R6::R6Class(
				"CustomLayer"
				, inherit = KerasLayer
				, public = list(
					
					call = function(x, mask = NULL) {
						k_concatenate(list(x, x, x), axis = 2)
					}
					
					, compute_output_shape = function(input_shape) {
						
						output_shape = input_shape
						output_shape[[2]] <- input_shape[[2]] * 3L
						
						output_shape
					}
				)
			)
			
			v_normalize_layer <- function(object) {
				create_layer(v_CustomLayer, object, list(name = 'v_normalize_layer'))
			}
			
			noise_CustomLayer <- R6::R6Class(
				"CustomLayer"
				, inherit = KerasLayer
				, lock_objects = FALSE
				, public = list(
					
					initialize = function(output_dim) {
						self$output_dim <- output_dim
					}
					
					, build = function(input_shape) {
						
						self$input_dim <- input_shape[[2]]
						
						sqr_inputs <- self$input_dim ** (1/2)
						
						self$sigma_initializer <- initializer_constant(.5 / sqr_inputs)
						
						self$mu_initializer <- initializer_random_uniform(minval = (-1 / sqr_inputs), maxval = (1 / sqr_inputs))
						
						self$mu_weight <- self$add_weight(
							name = 'mu_weight', 
							shape = list(self$input_dim, self$output_dim),
							initializer = self$mu_initializer,
							trainable = TRUE
						)
						
						self$sigma_weight <- self$add_weight(
							name = 'sigma_weight', 
							shape = list(self$input_dim, self$output_dim),
							initializer = self$sigma_initializer,
							trainable = TRUE
						)
						
						self$mu_bias <- self$add_weight(
							name = 'mu_bias', 
							shape = list(self$output_dim),
							initializer = self$mu_initializer,
							trainable = TRUE
						)
						
						self$sigma_bias <- self$add_weight(
							name = 'sigma_bias', 
							shape = list(self$output_dim),
							initializer = self$sigma_initializer,
							trainable = TRUE
						)
						
					}
					
					, call = function(x, mask = NULL) {
						
						#sample from noise distribution
						
						e_i = k_random_normal(shape = list(self$input_dim, self$output_dim))
						e_j = k_random_normal(shape = list(self$output_dim))
						
						
						#We use the factorized Gaussian noise variant from Section 3 of Fortunato et al.
						
						eW = k_sign(e_i) * (k_sqrt(k_abs(e_i))) * k_sign(e_j) * (k_sqrt(k_abs(e_j)))
						eB = k_sign(e_j) * (k_abs(e_j) ** (1/2))
						
						
						#See section 3 of Fortunato et al.
						
						noise_injected_weights = k_dot(x, self$mu_weight + (self$sigma_weight * eW))
						noise_injected_bias = self$mu_bias + (self$sigma_bias * eB)
						output = k_bias_add(noise_injected_weights, noise_injected_bias)
						
						output
						
					}
					
					, compute_output_shape = function(input_shape) {
						
						output_shape <- input_shape
						output_shape[[2]] <- self$output_dim
						
						output_shape
						
					}
				)
			)
			
			noise_add_layer <- function(object, output_dim) {
				create_layer(
					noise_CustomLayer
					, object
					, list(
						name = 'noise_add_layer'
						, output_dim = as.integer(output_dim)
						, trainable = T
					)
				)
			}
			
			critic_input <- layer_input(
				shape = list(self$lstm_seq_length, self$shape_input)
				, name = 'critic_input'
			)
			
			batch_norm <- layer_batch_normalization(name = 'batch_norm')
			
			common_lstm_layer <- layer_lstm(
				units = self$lstm_seq_length * 2L
				, activation = "tanh"
				, recurrent_activation = "hard_sigmoid"
				, use_bias = T
				, return_sequences = F
				, stateful = F
				, name = 'lstm1'
			)
			
			critic_layer_dense_v_1 <- layer_dense(
				units = 16
				, activation = "tanh"
			)
			
			critic_layer_dense_v_2 <- layer_dense(
				units = 8
				, activation = "tanh"
			)
			
			critic_layer_dense_v_3 <- layer_dense(
				units = 1
				, activation = 'linear'
				, name = 'critic_layer_dense_v_3'
			)
			
			critic_layer_dense_a_1 <- layer_dense(
				units = 16
				, activation = "tanh"
			)
			
			critic_layer_dense_a_2 <- layer_dense(
				units = 8
				, activation = "tanh"
			)
			
			critic_layer_dense_a_3 <- layer_dense(
				units = self$shape_output
				, activation = 'linear'
				, name = 'critic_layer_dense_a_3'
			)
			
			critic_model_v <-
				critic_input %>%
				#batch_norm %>%
				common_lstm_layer %>%
				critic_layer_dense_v_1 %>%
				critic_layer_dense_v_2 %>%
				critic_layer_dense_v_3 %>%
				v_normalize_layer
			
			critic_model_a <-
				critic_input %>%
				#batch_norm %>%
				common_lstm_layer %>%
				critic_layer_dense_a_1 %>%
				noise_add_layer(output_dim = 16) %>%
				critic_layer_dense_a_2 %>%
				critic_layer_dense_a_3 %>%
				a_normalize_layer
			
			critic_output <-
				layer_add(
					list(
						critic_model_v
						, critic_model_a
					)
					, name = 'critic_output'
				)
			
			critic_model  <- keras_model(
				inputs = critic_input
				, outputs = critic_output
			)
			
			self$nn <- 
				critic_model %>%
				keras::compile(
					optimizer = optimizer
					, loss = loss
					, metrics = metrics
				)
			
			cat('########## neural network was compiled ##########', '\n')
			
			invisible(self)
			
		}
		
		, save = function()
		{
			
			## save NN model
			
			save_model_hdf5(
				object = self$nn
				, filepath = 'primary_nn_model.h5'
				, overwrite = TRUE
				, include_optimizer = TRUE
			)
			
		}
		
	)
)
