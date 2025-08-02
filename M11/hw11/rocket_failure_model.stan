data {
  int<lower=1> J;
  array[J] int<lower=0> y;
  array[J] int<lower=0> n;
}

parameters {
  real log_alpha;
  real log_beta;
  vector[J] z_raw;  // non-centered z
}

transformed parameters {
  real<lower=0> alpha = exp(log_alpha);
  real<lower=0> beta = exp(log_beta);
  vector<lower=0,upper=1>[J] z;

  for (j in 1:J)
    z[j] = inv_logit(z_raw[j]);  // map raw values to (0,1)
}

model {
  // Priors
  log_alpha ~ normal(0, 2);
  log_beta ~ normal(0, 2);

  z_raw ~ normal(0, 1);  // non-centered

  for (j in 1:J)
    y[j] ~ binomial(n[j], z[j]);
}

generated quantities {
  real z_new;
  z_new = beta_rng(alpha, beta);
}