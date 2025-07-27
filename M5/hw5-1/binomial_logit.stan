data {
  int<lower=0> m;       // number of data points
  int<lower=0> n;       // number of trials
  array[m] int<lower=0, upper=n> y;  // number of successes
}
parameters {
  real theta;  // unconstrained
}
transformed parameters {
  real<lower=0, upper=1> eta;
  eta = inv_logit(theta);  // map theta to (0,1)
}
model {
  // prior on theta (unconstrained)
  theta ~ normal(0, 5);

  // likelihood
  y ~ binomial(n, eta);
}