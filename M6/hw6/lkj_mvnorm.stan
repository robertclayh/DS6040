data {
  int<lower=1> N;
  int<lower=1> D;
  matrix[N, D] y;
}

parameters {
  vector[D] mu;
  vector<lower=0>[D] sigma;
  cholesky_factor_corr[D] L;
}

model {
  mu ~ normal(0, 1);
  sigma ~ normal(0, 1) T[0, ];
  L ~ lkj_corr_cholesky(2.0);
  for (n in 1:N)
    y[n] ~ multi_normal_cholesky(mu, diag_pre_multiply(sigma, L));
}

generated quantities {
  array[N] vector[D] y_rep;
  for (n in 1:N)
    y_rep[n] = multi_normal_cholesky_rng(mu, diag_pre_multiply(sigma, L));
}