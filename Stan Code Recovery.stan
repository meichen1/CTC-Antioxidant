//
// Stan model for RECOVERY outcomes (e.g., symptom alleviation, functional recovery).
// Intercept prior is centered at 0 (logit(0.5)), reflecting the high baseline
// recovery rate in this population, in contrast to the hospitalization model
// which uses alpha ~ student_t(1, -3.48, 2.5).
//
// All other priors are identical to "Stan Code.stan".
//

data {
  // Meta data
  int<lower=1> N; // number of observations
  int<lower=1> J; // number of interventions (excluding control)
  int<lower=1> M; // number of columns in the design matrix
  // Data
  array[N] int<lower=0, upper=1> y; // recovery status (y=1 recovered by day 14)
  matrix<lower=0, upper=1>[N, J] Z; // design matrix of treatment assignments
  matrix[N, M] X; // design matrix of covariates
}
parameters {
  real alpha;
  vector[J] theta;
  vector[M] bbeta;
}
transformed parameters {
  vector[N] mu;
  mu = alpha + Z*theta + X*bbeta;
}
model {
  // Intercept prior centered at 0 (logit(0.5) = 0) for recovery outcomes
  alpha ~ student_t(1, 0, 2.5);
  theta ~ student_t(1, 0, 2.5);
  bbeta ~ student_t(1, 0, 2.5);
  // Likelihood
  y ~ bernoulli_logit(mu);
}
