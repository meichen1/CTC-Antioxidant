"""
Primary outcome analysis for CanTreatCOVID Antioxidant Study

Primary outcomes:
- Hospitalization or death within 28 days of randomization
- Early sustained recovery within 14 days of randomization

Follow up at 21 and 28 days captures: hospitalization, death, recovery status: 0 for all
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Prior hyperparameter constants (outcome-specific intercept location)
ALPHA_LOC_HOSP = -3.48      # logit(0.03) ≈ -3.48, appropriate for ~3% hospitalization rate
ALPHA_LOC_RECOVERY = 0.0    # logit(0.50) = 0, broad neutral prior for high recovery rates

try:
    from ctc_antiox.src_helper import first_change_to_1, ret_first_alleviation, ret_sustain_alleviation, sustain_change_to_1
except ModuleNotFoundError:
    from src_helper import first_change_to_1, ret_first_alleviation, ret_sustain_alleviation, sustain_change_to_1


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def replace_empty_with_none(df):
    """Replace empty strings in the DataFrame with None."""
    for col_name in df.columns:
        df[col_name] = df[col_name].replace(['', 'NA', 'N/A', 'na', 'NaN'], None)
    return df


# ============================================================================
# DATA LOADING AND CLEANING FUNCTIONS
# ============================================================================

def load_raw_data(csv_path):
    """Load raw antiox data from CSV file."""
    return pd.read_csv(csv_path)


def remove_rcc_ids(df):
    """Remove specific RCC IDs from the dataset."""
    rcc_ids_in_antiox = (
        df[df['redcap_event_name'] == 'Randomization']
        .sort_values('rand_date')
        .head(4)['participant_id'].tolist()
    )
    # rcc_ids_in_antiox.pop(3)
    print(f"Removing RCC IDs: {rcc_ids_in_antiox}")
    return df[~df['participant_id'].isin(rcc_ids_in_antiox)]


# def filter_study_completion(df):
#     """Filter participants based on study completion reasons."""
#     id_to_remove = (
#         df[df['end_study_reason'].isin([4, 6])]['participant_id'].to_list()
#     )
#     print(f"Removing {len(id_to_remove)} participants (lost to follow-up or withdrew)")
#     return df[~df['participant_id'].isin(id_to_remove)]


def load_and_clean_raw_data(csv_path):
    """Load and perform initial cleaning of raw data."""
    df = load_raw_data(csv_path)
    df = remove_rcc_ids(df)
    # df = filter_study_completion(df)
    print(f"Final dataset shape: {df.shape}")
    return df


def extract_randomization_data(df):
    """Extract and clean randomization data."""
    antiox_random = df[df['redcap_event_name'] == 'Randomization']
    antiox_random = antiox_random.dropna(axis=1, how='all')
    antiox_random = replace_empty_with_none(antiox_random)
    print(f"Randomization data shape: {antiox_random.shape}")
    return antiox_random


def extract_baseline_data(df, antiox_random):
    """Extract and clean baseline data."""
    antiox_baseline = df[df['redcap_event_name'] == 'Baseline']
    antiox_baseline = antiox_baseline.dropna(axis=1, how='all')
    antiox_baseline = replace_empty_with_none(antiox_baseline)
    
    # Merge with randomization data
    antiox_baseline = antiox_baseline.merge(
        antiox_random[['participant_id', 'rand_group', 'symp_onset_date', 'rand_date']], 
        on='participant_id', 
        how='left'
    )
    
    # Create comorbidity flag
    antiox_baseline['dem_comorb'] = (
        (antiox_baseline['chronic_disease'] > 0) | 
        (antiox_baseline['diagnose_addict'] > 0)
    ).astype(int)
    
    # Calculate age for those with missing values
    antiox_baseline.loc[antiox_baseline['dem_age_calc'] == 0, 'dem_age_calc'] = (
        (pd.to_datetime(antiox_baseline.loc[antiox_baseline['dem_age_calc'] == 0, 'rand_date']) - 
         pd.to_datetime(antiox_baseline.loc[antiox_baseline['dem_age_calc'] == 0, 'dem_dob'])).dt.days // 365.2425
    )
    
    print(f"Baseline data shape: {antiox_baseline.shape}")
    return antiox_baseline





def extract_followup_data(df):
    """Extract and aggregate follow-up data across all visit days."""
    followup_events = ['Day 21', 'Day 28']
    antiox_followup = df[df['redcap_event_name'].isin(followup_events)]
    antiox_followup = antiox_followup.dropna(axis=1, how='all')
    antiox_followup = replace_empty_with_none(antiox_followup)
    
    # Aggregate multiple follow-up events into single row per participant
    agg_dict = {col: 'first' for col in antiox_followup.columns if col != 'participant_id'}
    antiox_followup = antiox_followup.groupby('participant_id').agg(agg_dict).reset_index()
    
    print(f"Follow-up data shape: {antiox_followup.shape}")
    return antiox_followup



def extract_diary_data(df):
    """Extract and process daily e-diary data."""
    antiox_dd = df[df['redcap_event_name'].str.contains('Diar', na=False)]
    antiox_dd = antiox_dd.dropna(axis=1, how='all')
    antiox_dd = replace_empty_with_none(antiox_dd)
    
    cols_to_agg = ['pdd_recover', 'pdd_return_health', 'pdd_return_activity', 'pdd_feel_today', 
                   'pdd_fam_dr', 'pdd_walkin', 'pdd_tel_health', 'pdd_er', 'pdd_other_healthcare', 
                   'pdd_hospital', 'fpp_recover', 'fpp_return_health', 'fpp_return_activity', 
                   'fpp_feel_today', 'fpp_fam_dr', 'fpp_walkin', 'fpp_tel_health', 'fpp_er', 
                   'fpp_other_healthcare', 'fpp_hospital']
    cols_to_keep = ['participant_id', 'redcap_event_name'] + cols_to_agg
    antiox_dd = antiox_dd[cols_to_keep]
    
    # Combine pdd_ and fpp_ columns
    pdd_columns = [col for col in antiox_dd.columns if col.startswith('pdd_')]
    fpp_columns = [col for col in antiox_dd.columns if col.startswith('fpp_')]
    
    column_mapping = {}
    for pdd_col in pdd_columns:
        suffix = pdd_col[4:]
        fpp_col = f'fpp_{suffix}'
        if fpp_col in fpp_columns:
            column_mapping[pdd_col] = fpp_col
    
    for pdd_col, fpp_col in column_mapping.items():
        antiox_dd[pdd_col] = antiox_dd[pdd_col].combine_first(antiox_dd[fpp_col])
    
    antiox_dd = antiox_dd.drop(columns=list(column_mapping.values()))
    
    print(f"Diary data shape: {antiox_dd.shape}")
    return antiox_dd, column_mapping.keys()



def aggregate_diary_data(antiox_dd, pdd_col_names):
    """Aggregate diary data by participant."""
    cols_to_agg = list(pdd_col_names)
    
    cols_return_first = ['pdd_recover', 'pdd_return_health', 'pdd_return_activity']
    col_return_alleviation = ['pdd_feel_today']
    cols_return_sum = [col for col in cols_to_agg if col not in cols_return_first + col_return_alleviation]
    
    antiox_dd_pandas = antiox_dd.sort_values(['participant_id', 'redcap_event_name'])
    
    # Create aggregation dictionary
    agg_dict = {}
    for col in cols_return_first:
        if col in antiox_dd_pandas.columns:
            agg_dict[col] = [first_change_to_1, sustain_change_to_1]
    
    for col in col_return_alleviation:
        if col in antiox_dd_pandas.columns:
            agg_dict[col] = [ret_first_alleviation, ret_sustain_alleviation]
    
    for col in cols_return_sum:
        if col in antiox_dd_pandas.columns:
            agg_dict[col] = 'sum'
    
    antiox_dd_agg = antiox_dd_pandas.groupby('participant_id').agg(agg_dict).reset_index()
    
    # Flatten MultiIndex columns
    antiox_dd_agg.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col 
                               for col in antiox_dd_agg.columns]
    antiox_dd_agg = antiox_dd_agg.rename(columns={'participant_id_': 'participant_id'})
    
    # Create binary hospitalization outcome
    antiox_dd_agg['pdd_hospital_binary'] = (antiox_dd_agg['pdd_hospital_sum'] > 0).astype(int)
    antiox_dd_agg['pdd_er_binary'] = (antiox_dd_agg['pdd_er_sum'] > 0).astype(int)
    
    print(f"Aggregated diary data shape: {antiox_dd_agg.shape}")
    return antiox_dd_agg


#############################################################
###############################################################


def prepare_primary_outcome_dataset(antiox_dd_agg, antiox_followup, antiox_random, antiox_baseline):
    """Prepare dataset for primary outcome analysis (hospitalization)."""
    
    antiox_fup_primary = antiox_followup.copy()
    # Identify relevant outcome columns in follow-up data
    hosp_columns = [col for col in antiox_fup_primary.columns 
                    if col.startswith('fup') and col.endswith('_cov_hosp')]
    print(f"Hospitalization columns: {hosp_columns}")
    er_columns = [col for col in antiox_fup_primary.columns 
                  if col.startswith('fup') and col.endswith('_cov_er_visit')]
    print(f"ER visit columns: {er_columns}")
    
    # Create follow-up outcome variables
    antiox_fup_primary['fup_hosp_28d'] = 0
    antiox_fup_primary['fup_er_28d'] = 0
    
    for col in hosp_columns:
        antiox_fup_primary['fup_hosp_28d'] = antiox_fup_primary['fup_hosp_28d'] + (antiox_fup_primary[col] == 1)
    for col in er_columns:
        antiox_fup_primary['fup_er_28d'] = antiox_fup_primary['fup_er_28d'] + (antiox_fup_primary[col] == 1)
    
    # Start with diary data and pdd_hospital_binary
    antiox_primary = antiox_dd_agg[['participant_id', 'pdd_hospital_binary', 'pdd_er_binary']].copy()
    antiox_primary = antiox_primary.merge(
        antiox_fup_primary[['participant_id', 'fup_hosp_28d', 'fup_er_28d']],
        on='participant_id',
        how='left'
    )
    
    antiox_primary['pdd_hospital_binary'] = ((antiox_primary['pdd_hospital_binary'] == 1) | (antiox_primary['fup_hosp_28d'] > 0)).astype(int)
    antiox_primary['pdd_er_binary'] = ((antiox_primary['pdd_er_binary'] == 1) | (antiox_primary['fup_er_28d'] > 0)).astype(int)

    # Merge with randomization data
    antiox_primary = antiox_primary.merge(
        antiox_random[['participant_id', 'rand_group']], 
        on='participant_id', 
        how='left'
    )
    # Merge with baseline data (age, vaccination, comorbidity)
    antiox_primary = antiox_primary.merge(
        antiox_baseline[['participant_id', 'dem_age_calc', 'dem_vaccination_status', 'dem_comorb']], 
        on='participant_id', 
        how='left'
    )
    
    # Clean and prepare variables
    antiox_primary['dem_vac_orig'] = antiox_primary['dem_vaccination_status']
    antiox_primary['dem_vaccination_status'] = (antiox_primary['dem_vaccination_status'] > 0).astype(int)
    antiox_primary['rand_group'] = antiox_primary['rand_group'].map({'Antioxidant': 1, 'Usual Care': 0})
    
    print(f"Primary outcome dataset shape: {antiox_primary.shape}")
    return antiox_primary



def prepare_early_recovery_dataset(antiox_dd_agg, antiox_random, antiox_baseline):
    """Prepare dataset for early sustained recovery analysis."""
    # Start with diary data and recovery outcome
    antiox_earlysus = antiox_dd_agg[['participant_id', 'pdd_recover_sustain_change_to_1']].copy()
    
    # Merge with randomization data
    antiox_earlysus = antiox_earlysus.merge(
        antiox_random[['participant_id', 'rand_group']], 
        on='participant_id', 
        how='left'
    )
    
    # Merge with baseline data
    antiox_earlysus = antiox_earlysus.merge(
        antiox_baseline[['participant_id', 'dem_age_calc', 'dem_vaccination_status', 'dem_comorb']], 
        on='participant_id', 
        how='left'
    )
    
    # Clean and prepare variables
    antiox_earlysus['dem_vaccination_status'] = (antiox_earlysus['dem_vaccination_status'] > 0).astype(int)
    antiox_earlysus['rand_group'] = antiox_earlysus['rand_group'].map({'Antioxidant': 1, 'Usual Care': 0})
    antiox_earlysus['pdd_recover_sustain_binary'] = (antiox_earlysus['pdd_recover_sustain_change_to_1'] <= 14).astype(int)
    
    # # Remove any rows with missing values
    # antiox_earlysus = antiox_earlysus.dropna(axis=0, how='any')
    
    print(f"Early recovery dataset shape: {antiox_earlysus.shape}")
    return antiox_earlysus


def calculate_mean_recovery_rate(antiox_dd_agg, recovery_cols=None):
    """Calculate pooled binary recovery-by-day-14 rate for each recovery endpoint.
    Useful for setting endpoint-specific intercept priors (logit of the mean rate).
    """
    if recovery_cols is None:
        recovery_cols = [
            'pdd_recover_first_change_to_1', 'pdd_recover_sustain_change_to_1',
            'pdd_feel_today_ret_first_alleviation', 'pdd_feel_today_ret_sustain_alleviation',
            'pdd_return_health_first_change_to_1', 'pdd_return_health_sustain_change_to_1',
            'pdd_return_activity_first_change_to_1', 'pdd_return_activity_sustain_change_to_1',
        ]
    print("\n=== Mean Recovery Rates by Day 14 (Overall Sample) ===")
    rates = {}
    for col in recovery_cols:
        if col not in antiox_dd_agg.columns:
            continue
        valid_mask = antiox_dd_agg[col].notna()
        n_valid = valid_mask.sum()
        n_recovered = (antiox_dd_agg.loc[valid_mask, col] <= 14).sum()
        rate = n_recovered / n_valid if n_valid > 0 else np.nan
        logit_rate = np.log(rate / (1 - rate)) if (not np.isnan(rate) and 0 < rate < 1) else np.nan
        rates[col] = {'n': int(n_valid), 'n_recovered': int(n_recovered), 'rate': rate, 'logit_rate': logit_rate}
        print(f"  {col}: {n_recovered}/{n_valid} ({rate*100:.1f}%), logit = {logit_rate:.3f}")
    return rates


# ============================================================================
# ADDITIONAL IMPORTS FOR ANALYSIS
# ============================================================================

from cmdstanpy import CmdStanModel
# from sklearn.linear_model import LogisticRegression



# ============================================================================
# ANALYSIS FUNCTIONS
# ============================================================================

def compile_stan_model(stan_file):
    """Compile Stan model with error handling."""
    try:
        model = CmdStanModel(stan_file=stan_file)
        print("Stan model compiled successfully.\n")
        return model, True
    except Exception as e:
        print(f"Warning: Stan model compilation failed: {e}")
        print("Fallback to frequentist analysis will be used.\n")
        return None, False


def prepare_analysis_data(antiox_dataset, outcome_col):
    """
    Prepare data for analysis.
    """
    
    analysis_data = antiox_dataset[antiox_dataset[outcome_col].notna() & antiox_dataset['rand_group'].notna()].copy()
    
    print(f"Analysis dataset shape: {analysis_data.shape}")
    print(f"Outcome distribution: {analysis_data[outcome_col].sum()} / {len(analysis_data)} ({analysis_data[outcome_col].mean()*100:.1f}%)")
    print(f"Treatment distribution: Antioxidant={analysis_data['rand_group'].sum()} / {len(analysis_data)}")
    
    return analysis_data


def standardize_covariates(X1, X2, X3, use_continuous_age=True):
    """Standardize covariates for PANORAMIC-style analysis."""
    if use_continuous_age:
        X1_std = (X1 - np.mean(X1)) / (4 * np.var(X1, ddof=0)) ** 0.5
    else:
        X1_std = X1 - np.mean(X1)
    
    X2_std = X2 - np.mean(X2)
    X3_std = X3 - np.mean(X3)
    
    return X1_std, X2_std, X3_std


def run_stan_analysis(stan_data, model, superiority_direction='lo', model_name="Bayesian"):
    """Run Stan analysis with error handling and fallback."""
    try:
        fit = model.sample(
            data=stan_data,
            iter_sampling=10000,
            iter_warmup=10000,
            thin=10,
            chains=4,
            seed=1,
            show_console=False
        )
        
        theta_samples = fit.draws_pd()['theta[1]']
        theta_mean = theta_samples.mean()
        theta_ci_low = theta_samples.quantile(0.025)
        theta_ci_high = theta_samples.quantile(0.975)
        
        return {
            'theta_mean': theta_mean,
            'ci_low': theta_ci_low,
            'ci_high': theta_ci_high,
            'theta_samples': theta_samples,
            'probability_superiority': (theta_samples < 0).mean() if superiority_direction == 'lo' else (theta_samples > 0).mean(),
            'method': 'Bayesian'
        }
    except Exception as e:
        print(f"Error in {model_name}: {e}")
        return None


def run_frequentist_fallback(Z, X1_std, X2_std, X3_std, y):
    """Unpenalized frequentist fallback using statsmodels logistic regression."""
    try:
        import statsmodels.api as sm
        X_lr = np.column_stack([np.ones(len(y)), Z.flatten(), X1_std, X2_std, X3_std])
        logit_model = sm.Logit(y, X_lr)
        result = logit_model.fit(disp=0, maxiter=200)
        theta_mean = float(result.params[1])
        theta_se = float(result.bse[1])
        ci_low = theta_mean - 1.96 * theta_se
        ci_high = theta_mean + 1.96 * theta_se
        return {
            'theta_mean': theta_mean,
            'ci_low': ci_low,
            'ci_high': ci_high,
            'theta_samples': None,
            'probability_superiority': np.nan,
            'method': 'Frequentist (Unpenalized LogReg)'
        }
    except Exception as e:
        print(f"Frequentist fallback failed: {e}")
        return None


def panoramic_analysis_hospitalization(antiox_primary, model, stan_model_available, model_reg=None):
    """Run PANORAMIC-style analysis for hospitalization outcome."""
    print("\n" + "="*80)
    print("PANORAMIC-STYLE ANALYSIS - HOSPITALIZATION")
    print("="*80)
    
    analysis_data = prepare_analysis_data(antiox_primary, 'pdd_hospital_binary')
    
    y = analysis_data['pdd_hospital_binary'].astype(int).values
    Z = analysis_data['rand_group'].astype(int).values.reshape(-1, 1)
    X1 = analysis_data['dem_age_calc'].astype(float).values
    X2 = analysis_data['dem_vaccination_status'].astype(int).values
    X3 = analysis_data['dem_comorb'].astype(int).values
    ## impute missing values with mean (for continuous) or mode (for binary) - only for covariates, not outcome or treatment
    X1 = np.where(np.isnan(X1), np.nanmean(X1), X1)
    X2 = np.where(np.isnan(X2), np.round(np.nanmean(X2)), X2)
    X3 = np.where(np.isnan(X3), np.round(np.nanmean(X3)), X3)
    
    X1_std, X2_std, X3_std = standardize_covariates(X1, X2, X3, use_continuous_age=True)
    
    stan_data = {
        'N': len(analysis_data),
        'J': 1,
        'M': 3,
        'y': y.tolist(),
        'Z': Z.reshape(-1, 1).tolist(),
        'X': np.column_stack([X1_std, X2_std, X3_std]).tolist()
    }
    
    if stan_model_available:
        results = run_stan_analysis(stan_data, model, 'lo', "PANORAMIC Hospitalization")
    else:
        results = run_frequentist_fallback(Z, X1_std, X2_std, X3_std, y)
    
    if results:
        print(f"Treatment effect (log-OR): {results['theta_mean']:.3f} [{results['ci_low']:.3f}, {results['ci_high']:.3f}]")
        print(f"Odds Ratio: {np.exp(results['theta_mean']):.3f}")
        print(f"Method: {results['method']}")
        
        crosstab = pd.crosstab(analysis_data['rand_group'], analysis_data['pdd_hospital_binary'], margins=True)
        print(f"\nCross-tabulation:\n{crosstab}")

    # Sensitivity analysis: regularizing normal(0,1) priors
    results_reg = None
    if model_reg is not None and stan_model_available:
        print(f"\n[Sensitivity - Regularizing Priors (alpha_loc={ALPHA_LOC_HOSP})]")
        stan_data_reg = {**stan_data, 'alpha_loc': ALPHA_LOC_HOSP}
        results_reg = run_stan_analysis(stan_data_reg, model_reg, 'lo', "Regularizing Hospitalization")
        if results_reg:
            print(f"  Reg. OR: {np.exp(results_reg['theta_mean']):.3f} [{np.exp(results_reg['ci_low']):.3f}, {np.exp(results_reg['ci_high']):.3f}]")

    return analysis_data, results, results_reg


# def cantreaycovid_analysis_hospitalization(antiox_primary, model, stan_model_available):
#     """Run CanTreatCOVID-style analysis for hospitalization outcome."""
#     print("\n" + "="*80)
#     print("CANTREATCOVID-STYLE ANALYSIS - HOSPITALIZATION")
#     print("="*80)
    
#     analysis_data = prepare_analysis_data(antiox_primary, 'pdd_hospital_binary')
    
#     y = analysis_data['pdd_hospital_binary'].astype(int).values
#     Z = analysis_data['rand_group'].astype(int).values.reshape(-1, 1)
#     X1 = analysis_data['dem_age_calc'].astype(float).values
#     X2 = analysis_data['dem_vaccination_status'].astype(int).values
#     X3 = analysis_data['dem_comorb'].astype(int).values
#     # Age dichotomized at 65
#     X1_ctc = (X1 >= 65).astype(int)
    
#     ## impute missing values with mean (for continuous) or mode (for binary) - only for covariates, not outcome or treatment
#     X1_ctc = np.where(np.isnan(X1_ctc), np.round(np.nanmean(X1_ctc)), X1_ctc)
#     X2 = np.where(np.isnan(X2), np.round(np.nanmean(X2)), X2)
#     X3 = np.where(np.isnan(X3), np.round(np.nanmean(X3)), X3)
    
#     X1_ctc_std, X2_std, _ = standardize_covariates(X1_ctc, X2, X3, use_continuous_age=False)
    
#     print(f"Age >=65: {X1_ctc.sum()} / {len(X1_ctc)} ({X1_ctc.mean()*100:.1f}%)")
    
#     stan_data = {
#         'N': len(analysis_data),
#         'J': 1,
#         'M': 2,
#         'y': y.tolist(),
#         'Z': Z.reshape(-1, 1).tolist(),
#         'X': np.column_stack([X1_ctc_std, X2_std]).tolist()
#     }
    
#     if stan_model_available:
#         results = run_stan_analysis(stan_data, model,'lo', "CanTreatCOVID Hospitalization")
#     else:
#         results = run_frequentist_fallback(Z, X1_ctc_std, X2_std, np.zeros_like(X2_std), y)
    
#     if results:
#         print(f"Treatment effect (log-OR): {results['theta_mean']:.3f} [{results['ci_low']:.3f}, {results['ci_high']:.3f}]")
#         print(f"Odds Ratio: {np.exp(results['theta_mean']):.3f}")
#         print(f"Method: {results['method']}")
    
#     return analysis_data, results


def early_recovery_analysis(antiox_earlysus, model, stan_model_available, model_reg=None):
    """Run analysis for early sustained recovery outcome."""
    print("\n" + "="*80)
    print("EARLY SUSTAINED RECOVERY ANALYSIS (Day 14)")
    print("="*80)
    
    analysis_data = prepare_analysis_data(antiox_earlysus, 'pdd_recover_sustain_binary')
    
    y = analysis_data['pdd_recover_sustain_binary'].astype(int).values
    Z = analysis_data['rand_group'].astype(int).values.reshape(-1, 1)
    X1 = analysis_data['dem_age_calc'].astype(float).values
    X2 = analysis_data['dem_vaccination_status'].astype(int).values
    X3 = analysis_data['dem_comorb'].astype(int).values
    
    ## impute missing values with mean (for continuous) or mode (for binary) - only for covariates, not outcome or treatment
    X1 = np.where(np.isnan(X1), np.nanmean(X1), X1)
    X2 = np.where(np.isnan(X2), np.round(np.nanmean(X2)), X2)
    X3 = np.where(np.isnan(X3), np.round(np.nanmean(X3)), X3)
    
    X1_std, X2_std, X3_std = standardize_covariates(X1, X2, X3, use_continuous_age=True)
    
    stan_data = {
        'N': len(analysis_data),
        'J': 1,
        'M': 3,
        'y': y.tolist(),
        'Z': Z.reshape(-1, 1).tolist(),
        'X': np.column_stack([X1_std, X2_std, X3_std]).tolist()
    }
    
    if stan_model_available:
        results = run_stan_analysis(stan_data, model, 'hi', "Early Recovery")
    else:
        results = run_frequentist_fallback(Z, X1_std, X2_std, X3_std, y)
    
    if results:
        print(f"Treatment effect (log-OR): {results['theta_mean']:.3f} [{results['ci_low']:.3f}, {results['ci_high']:.3f}]")
        print(f"Odds Ratio: {np.exp(results['theta_mean']):.3f}")
        print(f"Probability of superiority (OR > 1): {(results['theta_samples'] > 0).mean() if results['theta_samples'] is not None else 'N/A'}")
        print(f"Method: {results['method']}")
        
        crosstab = pd.crosstab(analysis_data['rand_group'], analysis_data['pdd_recover_sustain_binary'], margins=True)
        print(f"\nCross-tabulation:\n{crosstab}")

    # Sensitivity analysis: regularizing normal(0,1) priors
    results_reg = None
    if model_reg is not None and stan_model_available:
        print(f"\n[Sensitivity - Regularizing Priors (alpha_loc={ALPHA_LOC_RECOVERY})]")
        stan_data_reg = {**stan_data, 'alpha_loc': ALPHA_LOC_RECOVERY}
        results_reg = run_stan_analysis(stan_data_reg, model_reg, 'hi', "Regularizing Recovery")
        if results_reg:
            print(f"  Reg. OR: {np.exp(results_reg['theta_mean']):.3f} [{np.exp(results_reg['ci_low']):.3f}, {np.exp(results_reg['ci_high']):.3f}]")

    return analysis_data, results, results_reg



def subgroup_analysis(antiox_primary_subgroup, model, stan_model_available):
    """Run subgroup analyses."""
    print("\n" + "="*80)
    print("SUBGROUP ANALYSIS - HOSPITALIZATION OUTCOME")
    print("="*80)
    
    subgroups = [
        # ('age_<=50', antiox_primary_subgroup['dem_age_calc'] <= 50),
        ('age_>50', antiox_primary_subgroup['dem_age_calc'] > 50),
        ('age_>65', antiox_primary_subgroup['dem_age_calc'] > 65),
        # ('sex_male', antiox_primary_subgroup['dem_sex'] == 1),
        # ('sex_female', antiox_primary_subgroup['dem_sex'] == 2),
        # ('race_white', antiox_primary_subgroup['dem_race'] == 1),
        # ('race_nonwhite', antiox_primary_subgroup['dem_race'] != 1),
        # ('income_<=5', antiox_primary_subgroup['dem_house_income'] <= 5),
        # ('income_>5', antiox_primary_subgroup['dem_house_income'] > 5),
        ('comorb_yes', antiox_primary_subgroup['dem_comorb'] == 1),
        # ('comorb_no', antiox_primary_subgroup['dem_comorb'] == 0),
        ('vax_==2', antiox_primary_subgroup['dem_vac_orig'] == 2),
        # ('vax_<2', antiox_primary_subgroup['dem_vac_orig'] < 2)
    ]
    
    subgroup_results = []
    
    for subgroup_name, subgroup_mask in subgroups:
        subgroup_data = antiox_primary_subgroup[subgroup_mask].copy()
        analysis_subgroup = subgroup_data[
            subgroup_data['pdd_hospital_binary'].notna()
        ].copy()
        
        if len(analysis_subgroup) < 10:
            print(f"\n{subgroup_name}: Skipped (n={len(analysis_subgroup)} too small)")
            continue
        
        print(f"\n{subgroup_name}: n={len(analysis_subgroup)}")
        
        y_sub = analysis_subgroup['pdd_hospital_binary'].astype(int).values
        Z_sub = analysis_subgroup['rand_group'].astype(int).values.reshape(-1, 1)
        X1_sub = analysis_subgroup['dem_age_calc'].astype(float).values
        X2_sub = analysis_subgroup['dem_vaccination_status'].astype(int).values
        X3_sub = analysis_subgroup['dem_comorb'].astype(int).values
        
        ## impute missing values with mean (for continuous) or mode (for binary) - only for covariates, not outcome or treatment
        X1_sub = np.where(np.isnan(X1_sub), np.nanmean(X1_sub), X1_sub)
        X2_sub = np.where(np.isnan(X2_sub), np.round(np.nanmean(X2_sub)), X2_sub)
        X3_sub = np.where(np.isnan(X3_sub), np.round(np.nanmean(X3_sub)), X3_sub)
        X1_std_sub, X2_std_sub, X3_std_sub = standardize_covariates(X1_sub, X2_sub, X3_sub)
        
        stan_data_sub = {
            'N': len(analysis_subgroup),
            'J': 1,
            'M': 3,
            'y': y_sub.tolist(),
            'Z': Z_sub.reshape(-1, 1).tolist(),
            'X': np.column_stack([X1_std_sub, X2_std_sub, X3_std_sub]).tolist()
        }
        
        if stan_model_available:
            results = run_stan_analysis(stan_data_sub, model, 'lo', f"Subgroup {subgroup_name}")
        else:
            results = run_frequentist_fallback(Z_sub, X1_std_sub, X2_std_sub, X3_std_sub, y_sub)
        
        if results:
            print(f"  Events: {y_sub.sum()}/{len(y_sub)} ({y_sub.mean()*100:.1f}%)")
            print(f"  OR: {np.exp(results['theta_mean']):.3f}")
            if results['theta_samples'] is not None:
                print(f"  Prob(OR < 1): {(results['theta_samples'] < 0).mean():.3f}")
            
            subgroup_results.append({
                'subgroup': subgroup_name,
                # 'n': len(analysis_subgroup),
                # 'n_events': y_sub.sum(),
                'n_antiox': Z_sub.sum(),
                'n_events_antiox': y_sub[Z_sub.flatten() == 1].sum(),
                'n_control': len(Z_sub) - Z_sub.sum(),
                'n_events_control': y_sub[Z_sub.flatten() == 0].sum(),
                'event_rate_antiox': y_sub[Z_sub.flatten() == 1].mean(),
                'event_rate_control': y_sub[Z_sub.flatten() == 0].mean(),
                'log_or': results['theta_mean'],
                'or': np.exp(results['theta_mean']),
                'ci_lower': np.exp(results['ci_low']) if not np.isnan(results['ci_low']) else np.nan,
                'ci_upper': np.exp(results['ci_high']) if not np.isnan(results['ci_high']) else np.nan,
                'probability_superiority': results['probability_superiority'],
                'method': results['method']
            })
    
    return subgroup_results



if __name__ == '__main__':
    print("="*80)
    print("CanTreatCOVID ANTIOXIDANT ARM - PRIMARY ANALYSIS")
    print("="*80)
    
    # ========================================================================
    # DATA LOADING AND CLEANING
    # ========================================================================
    
    df_antiox = load_and_clean_raw_data("/workspaces/CTC_covid/data/CSV_RCC_Data_Export_ALL_Final_2025-05-15-Antiox.csv")
    
    # Extract key datasets
    antiox_random = extract_randomization_data(df_antiox)
    antiox_baseline = extract_baseline_data(df_antiox, antiox_random)
    antiox_followup = extract_followup_data(df_antiox)
    antiox_dd, pdd_col_names = extract_diary_data(df_antiox)
    antiox_dd_agg = aggregate_diary_data(antiox_dd, pdd_col_names)
    calculate_mean_recovery_rate(antiox_dd_agg)
    
    # Prepare analysis datasets
    antiox_primary = prepare_primary_outcome_dataset(antiox_dd_agg, antiox_followup, antiox_random, antiox_baseline)
    antiox_earlysus = prepare_early_recovery_dataset(antiox_dd_agg, antiox_random, antiox_baseline)
    
    # ========================================================================
    # COMPILE STAN MODEL
    # ========================================================================
    
    model_hosp, stan_model_available = compile_stan_model('/workspaces/CTC_covid/py_src/ctc_antiox/Stan Code.stan')
    model_recovery, _ = compile_stan_model('/workspaces/CTC_covid/py_src/ctc_antiox/Stan Code Recovery.stan')
    model_reg, _ = compile_stan_model('/workspaces/CTC_covid/py_src/ctc_antiox/Stan Code Regularizing.stan')
    
    # ========================================================================
    # PRIMARY OUTCOMES ANALYSIS
    # ========================================================================
    
    # Hospitalization (PANORAMIC-style)
    hosp_pano_data, hosp_pano_results, hosp_reg_results = panoramic_analysis_hospitalization(
        antiox_primary, model_hosp, stan_model_available, model_reg=model_reg
    )
    
    # # Hospitalization (CanTreatCOVID-style)
    # hosp_ctc_data, hosp_ctc_results = cantreaycovid_analysis_hospitalization(
    #     antiox_primary, model, stan_model_available
    # )
    
    # Early Sustained Recovery
    recovery_data, recovery_results, recovery_reg_results = early_recovery_analysis(
        antiox_earlysus, model_recovery, stan_model_available, model_reg=model_reg
    )
    
    
    # ========================================================================
    # SAVE PRIMARY RESULTS
    # ========================================================================
    
    print("\n" + "="*80)
    print("SAVING PRIMARY RESULTS")
    print("="*80)
    
    res_rows = []
    # Primary analysis results
    if hosp_pano_results and recovery_results:
        res_rows.append({
            'analysis_type': 'PANORAMIC_hosp',
            'n_participants': len(hosp_pano_data),
            'n_events_antiox': hosp_pano_data[hosp_pano_data['rand_group'] == 1]['pdd_hospital_binary'].sum(),  
            'n_events_control': hosp_pano_data[hosp_pano_data['rand_group'] == 0]['pdd_hospital_binary'].sum(),
            'n_antiox': hosp_pano_data['rand_group'].sum(),
            'n_control': len(hosp_pano_data) - hosp_pano_data['rand_group'].sum(),
            'log_or': hosp_pano_results['theta_mean'],
            'odds_ratio': np.exp(hosp_pano_results['theta_mean']),
            'ci_lower': np.exp(hosp_pano_results['ci_low']),
            'ci_upper': np.exp(hosp_pano_results['ci_high']),
            'probability_superiority': hosp_pano_results['probability_superiority'],
            'method': hosp_pano_results['method']
        })
        res_rows.append({
            'analysis_type': 'EarlySustainedRecovery',
            'n_participants': len(recovery_data),
            'n_events_antiox': recovery_data[recovery_data['rand_group'] == 1]['pdd_recover_sustain_binary'].sum(),  
            'n_events_control': recovery_data[recovery_data['rand_group'] == 0]['pdd_recover_sustain_binary'].sum(),
            'n_antiox': recovery_data['rand_group'].sum(),
            'n_control': len(recovery_data) - recovery_data['rand_group'].sum(),
            'log_or': recovery_results['theta_mean'],
            'odds_ratio': np.exp(recovery_results['theta_mean']),
            'ci_lower': np.exp(recovery_results['ci_low']),
            'ci_upper': np.exp(recovery_results['ci_high']),
            'probability_superiority': recovery_results['probability_superiority'],
            'method': recovery_results['method']
        })
        
    ## Save regularizing sensitivity results # prior using normal(0,1) 
    if hosp_reg_results:
        res_rows.append({
            'analysis_type': 'PANORAMIC_hosp_regularizing',
            'log_or': hosp_reg_results['theta_mean'],
            'odds_ratio': np.exp(hosp_reg_results['theta_mean']),
            'ci_lower': np.exp(hosp_reg_results['ci_low']),
            'ci_upper': np.exp(hosp_reg_results['ci_high']),
            'probability_superiority': hosp_reg_results['probability_superiority'],
            'method': hosp_reg_results['method']
        })
    if recovery_reg_results:
        res_rows.append({
            'analysis_type': 'EarlySustainedRecovery_regularizing',
            'log_or': recovery_reg_results['theta_mean'],
            'odds_ratio': np.exp(recovery_reg_results['theta_mean']),
            'ci_lower': np.exp(recovery_reg_results['ci_low']),
            'ci_upper': np.exp(recovery_reg_results['ci_high']),
            'probability_superiority': recovery_reg_results['probability_superiority'],
            'method': recovery_reg_results['method']
        })
    if res_rows:
        results_df = pd.DataFrame(res_rows)
        results_df.to_csv('/workspaces/CTC_covid/py_src/results_antiox/antiox_primary_analysis_results.csv', index=False, float_format='%.3f')
        print(f"✓ Primary analysis results saved")

    
    
    # ========================================================================
    # SUBGROUP ANALYSIS
    # ========================================================================
    
    antiox_primary_subgroup = antiox_primary.merge(
        antiox_baseline[['participant_id', 'dem_sex', 'dem_race', 'dem_house_income']], 
        on='participant_id', 
        how='left'
    )
    subgroup_results_list = subgroup_analysis(antiox_primary_subgroup, model_hosp, stan_model_available)
    
    ## Subgroup analysis results
    if subgroup_results_list:
        subgroup_df = pd.DataFrame(subgroup_results_list)
        subgroup_df.to_csv('/workspaces/CTC_covid/py_src/results_antiox/antiox_primary_subgroup.csv', index=False, float_format='%.3f')
        print(f"✓ Subgroup analysis results saved")
    
















































