data {
  int<lower=0> N;
  array[N] int<lower=0> y;  // modern array declaration
}

parameters {
  real theta;  // unconstrained parameter
}

model {
  theta ~ normal(4.5, 1.5);       // weak prior centered around log(85–100)
  y ~ poisson(exp(theta));       // likelihood uses exp(theta) as the Poisson rate
}

generated quantities {
  real lambda;
  lambda = exp(theta);           // derived positive rate
}