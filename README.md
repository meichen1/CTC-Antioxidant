
# CanTreatCOVID Antioxidant Analyses

This repo contains analysis scripts for the antioxidant arm of CanTreatCOVID, focused on primary outcomes and secondary binary recovery outcomes.

## Main Scripts

- Primary analysis: antiox_primary_2603.py
- Secondary binary recovery analysis: antiox_secd_bin_2603.py
- Shared helper functions: src_helper.py
- Stan logistic model: Stan Code.stan

## Data Input

Both scripts read the REDCap export:

- CSV_RCC_Data_Export_ALL_Final_2025-05-15-Antiox.csv

Data preparation steps include:
- extraction of Randomization, Baseline, Follow-up, and Diary events
- combination of pdd_ and fpp_ diary fields where applicable
- derivation of recovery timing and binary endpoint variables

## Primary Analysis

Script: antiox_primary.py

Main outcomes:
- hospitalization within 28 days
- early sustained recovery by day 14

Approach:
- PANORAMIC-style Bayesian logistic model in Stan
- fallback to frequentist logistic regression when Stan is unavailable
- covariate adjustment using age, vaccination status, and comorbidity
- additional subgroup analysis (age, comorbidity, vaccination)

Outputs:
- antiox_primary_analysis_results.csv
- antiox_primary_subgroup.csv

## Secondary Binary Analysis

Script: antiox_secd_bin.py

Purpose:
- for each recovery-related diary metric, create a binary endpoint recovered by day 14
- estimate treatment effect as odds ratio with uncertainty and probability of superiority

Recovery metrics analyzed include:
- first and sustained recovery
- first and sustained symptom alleviation
- return to health/activity (first and sustained)

Approach:
- Bayesian logistic model via Stan
- fallback crude/frequentist OR when Stan fails
- subgroup analyses for selected demographic and clinical strata
- markdown report per outcome


## Run

From repository root:

python antiox_primary_2603.py  
python antiox_secd_bin_2603.py

## Environment

Core dependencies are listed in requirements.txt, including:
- pandas, numpy, scikit-learn
- cmdstanpy
- statsmodels

If Stan is not configured, scripts continue using fallback frequentist methods where implemented.
