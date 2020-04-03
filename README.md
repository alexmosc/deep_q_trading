# deep_q_trading
R language OOP implementation of deep Q learning to simulate stock trading

Packages you need to install before running the script:

R6,
data.table,
magrittr,
keras,
quantmod,
TTR.

with the command install.packages('TTR', dependencies = T).

Installing keras for R is a litle different but should also be easy; refer to a manual: https://keras.rstudio.com/.

After you are done with the packages, run main.R script. Default parameters will use a generated signal timeseries and training will last for 5000 iterations.

If you wish to change parameters or load real stock data, you have to tune the main.R script.

It is best to run the thing in RStudio for full control over the training. You will also see charts popping up.
