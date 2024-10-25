phi <- 1/3               # MA coefficient (not used directly but relates to the filter)
n   <- 10000             # Number of observations
w   <- rnorm(n, sd = 1)  # White noise with mean 0 and variance 1


V   <- stats::filter(w, filter = rep(phi, 3), method = 'convolution', sides = 2)
plot.ts(V)

# Remove NA values
V_clean <- na.omit(V)

cor_1 <- cor(V_clean[1:(n-1)], V_clean[2:n])     
cor_2 <- cor(V_clean[1:(n-2)], V_clean[3:(n)])  
save.cor <- c()

for(h in 1:20) {
  save.cor <- c(save.cor, cor(V_clean[1:(length(V_clean)-h)], V_clean[(h+1):length(V_clean)]))
}
plot(1:20, save.cor, type = "b", pch = 19, col = "blue",
     xlab = "Lag", ylab = "Autocorrelation",
     main = "Empirical Autocorrelations for MA(2) Model")



# Using the default acf function to compute autocorrelation
acf.v <- acf(V_clean, lag.max = 20, plot = FALSE)

plot(acf.v$lag, acf.v$acf, type = "h", lwd = 2, col = "black",
     main = "ACF vs Empirical Autocorrelations for MA(2) Model",
     xlab = "Lag", ylab = "Autocorrelation")

points(1:20, save.cor, col = "blue", pch = 19)


# Add the theoretical autocorrelation for MA(2)
h <- 1:20
true_cor <- c(2/3, 1/3, rep(0, 18))  # Theoretical ACF for MA(2)

# Plot the theoretical autocorrelations
lines(1:20, true_cor, col = "red", lwd = 2, type = "b", pch = 17)


# Approximate test of the null hypothesis: autocorrelation=0
abline(h = 2/sqrt(n), col = "green", lwd = 2, lty = 2)
abline(h = -2/sqrt(n), col = "green", lwd = 2, lty = 2)

legend("topright",
       legend = c("ACF Function", "Empirical (Loop)", "Theoretical"),
       col = c("black", "blue", "red"),
       pch = c(NA, 19, 17),
       lty = c(1, 1, 1),
       lwd = c(2, 1, 2))


