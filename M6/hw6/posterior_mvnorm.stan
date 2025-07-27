data {
  int<lower=1> N;               // number of observations
  int<lower=1> D;               // dimensionality
  matrix[N, D] y;               // observed data
  vector[D] mu0;
  real<lower=0> kappa0;
  real<lower=D-1> nu0;
  matrix[D, D] Lambda0;
}
parameters {
  vector[D] mu;
  cov_matrix[D] Sigma;
}
model {
  Sigma ~ inv_wishart(nu0, Lambda0);
  mu ~ multi_normal(mu0, Sigma / kappa0);
  for (n in 1:N)
    y[n] ~ multi_normal(mu, Sigma);
}
generated quantities {
  array[N] vector[D] y_rep;
  for (n in 1:N)
    y_rep[n] = multi_normal_rng(mu, Sigma);
}