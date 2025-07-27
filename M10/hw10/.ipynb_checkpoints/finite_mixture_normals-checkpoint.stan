data {
  int<lower=1> N;           // number of observations
  int<lower=1> D;           // number of features/dimensions
  int<lower=1> K;           // number of mixture components
  matrix[N, D] x;           // predictor matrix
}

parameters {
  simplex[K] lambda;                  // mixing proportions
  array[K] vector[D] mu;              // component means
  array[K] vector<lower=1e-3>[D] sigma;  // component std devs (bounded away from zero)
}

model {
  // Informative priors based on your empirical summary statistics
  mu[1] ~ normal([-0.004, -0.42, -0.41,  0.29, -0.33], 0.3);   // cluster 0
  mu[2] ~ normal([ 0.02,  1.93,  1.87, -1.33,  1.51], 0.3);     // cluster 1

  sigma[1] ~ normal([1.02, 0.33, 0.49, 0.70, 0.71], 0.1);       // cluster 0 stds
  sigma[2] ~ normal([0.92, 0.77, 0.48, 0.74, 0.74], 0.1);       // cluster 1 stds

  lambda ~ dirichlet(rep_vector(2.0, K));  // symmetric weak prior

  for (n in 1:N) {
    vector[K] log_probs;
    for (k in 1:K) {
      log_probs[k] = log(lambda[k]);
      for (d in 1:D)
        log_probs[k] += normal_lpdf(x[n, d] | mu[k][d], sigma[k][d]);
    }
    target += log_sum_exp(log_probs);
  }
}

generated quantities {
  array[N] vector[K] label_prob;

  for (n in 1:N) {
    vector[K] log_probs;
    for (k in 1:K) {
      log_probs[k] = log(lambda[k]);
      for (d in 1:D)
        log_probs[k] += normal_lpdf(x[n, d] | mu[k][d], sigma[k][d]);
    }
    label_prob[n] = softmax(log_probs);
  }
}