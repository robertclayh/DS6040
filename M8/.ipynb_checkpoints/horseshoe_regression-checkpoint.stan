data {
  int<lower=0> N;              // number of observations
  int<lower=0> K;              // number of predictors
  matrix[N, K] X;              // predictor matrix
  vector[N] y;                 // log-price response
}

parameters {
  real alpha;                  // intercept
  real<lower=0> sigma;         // noise scale
  vector[K] z;                 // standard normal base for coefficients
  real<lower=0> tau;           // global shrinkage
  vector<lower=0>[K] lambda;   // local shrinkage
}

transformed parameters {
  real epsilon = 1e-8;
  vector[K] safe_lambda = lambda + epsilon;
  real safe_tau = tau + epsilon;

  vector[K] raw_beta = z .* safe_lambda * safe_tau;

  // Clip beta to avoid Inf or NaN values
  vector[K] beta;
  for (k in 1:K)
    beta[k] = fmin(fmax(raw_beta[k], -1e6), 1e6);
}

model {
  // Priors
  z ~ normal(0, 1);
  lambda ~ student_t(3, 0, 1); // more stable than half-Cauchy
  tau ~ student_t(3, 0, 1);
  alpha ~ normal(0, 5);
  sigma ~ normal(0, 2);

  // Linear predictor and likelihood
  vector[N] mu = alpha + X * beta;

  // Clip mu to avoid -inf/nan
  for (n in 1:N)
    mu[n] = fmin(fmax(mu[n], -1e6), 1e6);

  y ~ normal(mu, sigma);
}
