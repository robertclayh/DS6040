data {
  int<lower=0> N;
  array[N] int<lower=0> y;  // modern syntax
}

parameters {
  real mu;
  real<lower=0> sigma;
  vector[N] log_lambda;
}

model {
  mu ~ normal(0, 10);
  sigma ~ cauchy(0, 5);
  log_lambda ~ normal(mu, sigma);
  y ~ poisson_log(log_lambda);  // vectorized
}