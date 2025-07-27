data {
  int<lower=0> N;                 // number of observations
  int<lower=0> K;                 // number of predictors (2)
  matrix[N, K] X;                 // predictor matrix
  array[N] int<lower=0> y;        // response variable
}

parameters {
  real alpha;                     // intercept
  vector[K] beta;                 // coefficients
}

model {
  // Priors
  alpha ~ normal(0, 5);
  beta ~ normal(0, 2);

  // Likelihood
  y ~ poisson_log(alpha + X * beta);
}
