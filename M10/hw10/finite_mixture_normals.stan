data {
  int<lower=0> N;         // number of data points
  int<lower=1> D;         // number of features
  int<lower=1> H;         // number of clusters
  matrix[N, D] y;         // data matrix
}

parameters {
  simplex[H] lambda;                    // mixture weights
  array[H] ordered[D] mu;              // ordered cluster means
  array[H] vector<lower=0>[D] tau;     // per-cluster std deviations
  array[H] cov_matrix[D] Sigma;        // full covariance matrix for each cluster
}

model {
  vector[H] log_ps;

  // Mixture weights
  lambda ~ dirichlet(rep_vector(1.0, H));

  for (h in 1:H) {
    mu[h] ~ normal(0, 3);                         // Regularized prior on means
    tau[h] ~ normal(0, 1);                        // Tighter prior than Cauchy
    Sigma[h] ~ inv_wishart(D + 2, diag_matrix(rep_vector(1.0, D))); // weakly informative IW prior
  }

  // Likelihood
  for (n in 1:N) {
    for (h in 1:H) {
      log_ps[h] = log(lambda[h]) + multi_normal_lpdf(y[n] | mu[h], Sigma[h]);
    }
    target += log_sum_exp(log_ps);
  }
}

generated quantities {
  matrix[N, H] label_prob;

  for (n in 1:N) {
    vector[H] log_ps;
    for (h in 1:H) {
      log_ps[h] = log(lambda[h]) + multi_normal_lpdf(y[n] | mu[h], Sigma[h]);
    }
    label_prob[n] = softmax(log_ps)';
  }
}