data {
  int<lower=1> N;                      // number of observations
  int<lower=1> K;                      // number of predictors
  matrix[N, K] X;                      // predictor matrix
  array[N] int<lower=0> y;             // response variable
  vector[N] log_kms;                   // offset (e.g., log(kms))
}
parameters {
  real alpha;                          // intercept
  vector[K] beta;                      // coefficients
}
model {
  // Priors
  alpha ~ normal(0, 10);
  beta ~ normal(0, 10);

  // Likelihood with offset
  y ~ poisson_log(log_kms + alpha + X * beta);
}
