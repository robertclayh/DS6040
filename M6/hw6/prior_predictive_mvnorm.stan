data {
  int<lower=1> N;                   // Number of samples to simulate
  int<lower=2> D;                   // Dimension
  vector<lower=0>[D] diag_cov;      // Prior covariance diagonals
}

generated quantities {
  matrix[N, D] y;                   // Simulated dataset
  vector[D] mu;
  cov_matrix[D] Sigma;

  Sigma = inv_wishart_rng(10, diag_matrix(diag_cov));         // Prior on Sigma
  mu = multi_normal_rng(rep_vector(0, D), Sigma);             // Prior on mu
  for (n in 1:N)
    y[n] = to_row_vector(multi_normal_rng(mu, Sigma));        // Simulate each y[n]
}