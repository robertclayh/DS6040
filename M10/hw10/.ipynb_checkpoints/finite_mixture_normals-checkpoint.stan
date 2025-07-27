data {
  int<lower=1> N;                     // Number of observations
  int<lower=1> D;                     // Number of features
  int<lower=1> H;                     // Number of clusters
  matrix[N, D] y;                     // Data matrix
  array[H] int<lower=D+1> nu;         // Degrees of freedom for Inv-Wishart
  array[H] matrix[D, D] Psi;          // Scale matrices for Inv-Wishart
}

parameters {
  simplex[H] lambda;                 // Mixture weights
  array[H] vector[D] mu;            // Cluster means
  array[H] cov_matrix[D] Sigma;     // Cluster covariance matrices
}

model {
  vector[H] log_ps;

  lambda ~ dirichlet(rep_vector(1.0, H));

  for (h in 1:H) {
    mu[h] ~ normal(0, 5);
    Sigma[h] ~ inv_wishart(nu[h], Psi[h]);
  }

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