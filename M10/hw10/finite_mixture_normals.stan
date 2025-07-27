data {
  int<lower=0> N;         // number of data points
  int<lower=1> D;         // dimensionality of each data point
  matrix[N, D] y;         // data matrix (each row is a data point)
}

parameters {
  simplex[3] lambda;        // mixture weights
  array[3] vector[D] mu;    // a mean vector for each cluster
  array[3] cov_matrix[D] Sigma;
}

model {
  array[3] real ps;

  // prior!
  vector[3] alpha = rep_vector(1.0, 3); 
  lambda ~ dirichlet(alpha);
  for (k in 1:3) {
    mu[k] ~ normal(0, 10);            
    Sigma[k] ~ inv_wishart(D + 1, diag_matrix(rep_vector(1, D)));  
  }

  // observed-data likelihood
  for (n in 1:N) {
    for (k in 1:3) {
      ps[k] = log(lambda[k]) + multi_normal_lpdf(y[n] | mu[k], Sigma[k]);
    }
    target += log_sum_exp(ps);
  }
}


generated quantities {
  matrix[N, 3] label_prob;   // probabilities of each row's label
  
  for (n in 1:N) {
    array[3] real log_ps;
    real max_log_ps;
    
    for (k in 1:3) {
      log_ps[k] = log(lambda[k]) + multi_normal_lpdf(y[n] | mu[k], Sigma[k]); 
    }
    
    max_log_ps = max(log_ps);
    
    for (k in 1:3) {
      label_prob[n, k] = exp(log_ps[k] - max_log_ps);
    }
    
    // Normalize to get probabilities
    label_prob[n] /= sum(label_prob[n]);
  }
}
