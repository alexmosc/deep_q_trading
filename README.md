# deep_q_trading
Welcome to an R language OOP implementation of deep Q learning to simulate stock trading.

This project is an end-to-end prototype that will make you familiar with an approach and allow to build a neural network model that can "trade" timeseries.

Packages you need to install before running the script:

R6,
data.table,
magrittr,
ggplot2,
keras,
quantmod,
TTR.

with the command like the following: install.packages('TTR', dependencies = T).

Installing keras for R is a little different, but should typically be easy; refer to a manual: https://keras.rstudio.com/.

This is it. After you are done with the packages, run main.R script. Default parameters will use a generated signal timeseries and training will last for 5000 iterations.

If you wish to change parameters or load real stock data, you have to tune the main.R script.

It is best to run the thing in RStudio for full control over the training. You will also see charts popping up.
