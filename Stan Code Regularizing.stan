//
// Stan model with more regularizing normal(0,1) priors on treatment effect (theta)
// and covariates (bbeta), as a sensitivity analysis.
//
// The intercept prior location (alpha_loc) is passed as data so it can be set
// to the outcome-appropriate value:
//   - Hospitalization: alpha_loc = -3.48  (logit(0.03))
//   - Recovery:        alpha_loc =  0.0   (logit(0.50))
//
// This allows evaluation of how robust estimates are to tighter priors on
// treatment and covariate coefficients versus the base student_t(1,0,2.5).
//

data {
  // Meta data
  int<lower=1> N; // number of observations
  int<lower=1> J; // number of interventions (excluding control)
  int<lower=1> M; // number of columns in the design matrix
  // Data
  array[N] int<lower=0, upper=1> y;
  matrix<lower=0, upper=1>[N, J] Z;
  matrix[N, M] X;
  // Prior hyperparameter
  real alpha_loc; // outcome-specific intercept prior location
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
  // Outcome-specific intercept prior (same width as base model)
  alpha ~ student_t(1, alpha_loc, 2.5);
  // Regularizing normal(0,1) priors on treatment and covariates
  theta ~ normal(0, 1);
  bbeta ~ normal(0, 1);
  // Likelihood
  y ~ bernoulli_logit(mu);
}
