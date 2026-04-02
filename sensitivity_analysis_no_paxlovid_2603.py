"""
Sensitivity Analysis - Excluding Paxlovid Takers (Addition 20260215)

This script performs a sensitivity analysis by removing participants who took Paxlovid
from both antioxidant and usual care groups, then re-runs primary and secondary analyses.

"""

import os
import sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to handle imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from py_src.ctc_antiox.antiox_primary_2603 import (
    load_and_clean_raw_data, 
    extract_randomization_data, 
    extract_baseline_data, 
    extract_followup_data, 
    extract_diary_data, 
    aggregate_diary_data,
    prepare_primary_outcome_dataset,
    prepare_early_recovery_dataset,
    compile_stan_model,
    prepare_analysis_data,
    standardize_covariates,
    run_stan_analysis,
    run_frequentist_fallback
)

from cmdstanpy import CmdStanModel


# List of participant IDs who took Paxlovid
# Antioxidant group: n=5 (1-1057, 1-1041, 1-1038, 1-1024, 1-1021)
# Usual Care group: n=2 (1-1054, 1-1028)
PAXLOVID_TAKERS = ['1-1021', '1-1024', '1-1028', '1-1038', '1-1041', '1-1054', '1-1057']

def secondary_binary_analysis_no_subgroups(antiox_dd_agg, antiox_baseline, antiox_random, 
                                           timerecovery_col, max_time=14, output_dir='results_antiox'):
    """
    Binary logistic regression analysis for recovery outcomes by day 14 WITHOUT subgroup analyses
    Matches the comprehensive reporting format of antiox_secd_bin_251007.py
    
    Parameters:
    -----------
    antiox_dd_agg : DataFrame
        Aggregated daily diary data
    antiox_baseline : DataFrame
        Baseline characteristics data
    antiox_random : DataFrame  
        Randomization data
    timerecovery_col : str
        Column name for time to recovery outcome
    max_time : int
        Cut-off time for binary outcome (default 14 days)
    output_dir : str
        Output directory for results
        
    Returns:
    --------
    dict : Dictionary with analysis results (NO subgroup results)
    """
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Binary Analysis for {timerecovery_col} ===")
    
    # Merge datasets
    antiox_secondary = antiox_dd_agg[['participant_id', timerecovery_col]].merge(
        antiox_baseline[['participant_id', 'dem_age_calc', 'dem_vaccination_status', 'dem_comorb']], 
        on='participant_id', how='left'
    ).merge(
        antiox_random[['participant_id', 'rand_group']], 
        on='participant_id', how='left'
    )
    
    # Convert treatment group: 1 for antioxidant, 0 for control
    antiox_secondary['treatment'] = (antiox_secondary['rand_group'] == 'Antioxidant').astype(int)
    
    # Create binary outcome: recovered by day max_time (1=yes, 0=no)
    antiox_secondary['recovered_by_day14'] = (antiox_secondary[timerecovery_col] <= max_time).astype(int)
    
    # Clean data - remove missing values
    analysis_data = antiox_secondary[antiox_secondary[timerecovery_col].notna()].copy()
    
    print(f"Analysis dataset size: {len(analysis_data)} participants")
    print(f"Treatment group: {analysis_data['treatment'].sum()} participants")
    print(f"Control group: {len(analysis_data) - analysis_data['treatment'].sum()} participants")
    
    if len(analysis_data) == 0:
        print("No valid data for analysis")
        return {}
    
    # Calculate descriptive statistics by treatment group
    def calculate_descriptive_stats(df, outcome_col, time_col):
        stats = {}
        for group in ['Antioxidant', 'Usual Care']:
            group_data = df[df['rand_group'] == group]
            
            # Binary outcome statistics
            n_recovered = group_data['recovered_by_day14'].sum()
            n_total = len(group_data)
            pct_recovered = (n_recovered / n_total * 100) if n_total > 0 else 0
            
            # Time statistics (for those who recovered)
            recovered_times = group_data[group_data['recovered_by_day14'] == 1][time_col]
            if len(recovered_times) > 0:
                median_time = np.median(recovered_times)
                q25_time = np.percentile(recovered_times, 25)
                q75_time = np.percentile(recovered_times, 75)
            else:
                median_time = q25_time = q75_time = np.nan
                
            # All times (including censored)
            all_times = group_data[time_col]
            median_all = np.median(all_times)
            q25_all = np.percentile(all_times, 25) 
            q75_all = np.percentile(all_times, 75)
            
            stats[group] = {
                'n_total': n_total,
                'n_recovered': n_recovered,
                'pct_recovered': pct_recovered,
                'median_time_recovered': median_time,
                'q25_time_recovered': q25_time,
                'q75_time_recovered': q75_time,
                'median_time_all': median_all,
                'q25_time_all': q25_all,
                'q75_time_all': q75_all
            }
        return stats
    
    descriptive_stats = calculate_descriptive_stats(analysis_data, 'recovered_by_day14', timerecovery_col)
    
    # Prepare variables for Stan analysis
    y = analysis_data['recovered_by_day14'].values.astype(int)
    treatment = analysis_data['treatment'].values.astype(int)
    age = analysis_data['dem_age_calc'].values.astype(float)
    vaccination_status = analysis_data['dem_vaccination_status'].values.astype(int)
    comorb = analysis_data['dem_comorb'].values.astype(int)
    
    # Handle missing values for age, vaccination status, and comorbidity by imputation (mean for age, mode for categorical)
    age = np.where(np.isnan(age), np.nanmean(age), age)
    vaccination_status = np.where(np.isnan(vaccination_status), np.round(np.nanmean(vaccination_status)), vaccination_status)
    comorb = np.where(np.isnan(comorb), np.round(np.nanmean(comorb)), comorb)
    
    # PANORAMIC-style analysis (continuous age + standardization)
    age_std, vaccination_status_std, comorb_std = standardize_covariates(age, vaccination_status, comorb)
    
    results = {}
    try:
        # First check if Stan is available
        stan_file = "/workspaces/CTC_covid/py_src/ctc_antiox/Stan Code.stan"
        model = CmdStanModel(stan_file=stan_file)
        
        # PANORAMIC analysis
        stan_data_pan = {
            'N': len(analysis_data),
            'J': 1,  # number of interventions
            'M': 3,  # number of covariates
            'y': y.tolist(),
            'Z': treatment.reshape(-1, 1).tolist(),
            'X': np.column_stack([age_std, vaccination_status_std, comorb_std]).tolist()
        }
        
        # Run PANORAMIC model
        print("Running PANORAMIC-style analysis...")
        fit_pan = model.sample(data=stan_data_pan, chains=4, iter_sampling=20000, iter_warmup=10000, seed=42)
        draws_pan = fit_pan.draws_pd()
        
        # Extract PANORAMIC results
        theta_pan = draws_pan['theta[1]']
        or_samples_pan = np.exp(theta_pan)
        or_mean_pan = np.mean(or_samples_pan)
        or_ci_lower_pan = np.percentile(or_samples_pan, 2.5)
        or_ci_upper_pan = np.percentile(or_samples_pan, 97.5)
        prob_superiority_pan = np.mean(or_samples_pan > 1)
        
        results = {
            'panoramic': {
                'or_mean': or_mean_pan,
                'or_ci_lower': or_ci_lower_pan,
                'or_ci_upper': or_ci_upper_pan,
                'prob_superiority': prob_superiority_pan
            }
        }
        
        print("Stan analysis completed successfully")
        
    except Exception as e:
        print(f"Stan analysis failed: {e}")
        print("Using fallback frequentist logistic regression analysis...")
        
        from scipy import stats
        import warnings
        warnings.filterwarnings('ignore')
        
        try:
            treat_success = np.sum((treatment == 1) & (y == 1))
            treat_total = np.sum(treatment == 1)
            treat_fail = treat_total - treat_success
            
            control_success = np.sum((treatment == 0) & (y == 1))
            control_total = np.sum(treatment == 0)
            control_fail = control_total - control_success
            
            if treat_fail > 0 and control_fail > 0 and treat_success > 0 and control_success > 0:
                or_crude = (treat_success * control_fail) / (treat_fail * control_success)
                log_or_se = np.sqrt(1/treat_success + 1/treat_fail + 1/control_success + 1/control_fail)
                log_or = np.log(or_crude)
                or_ci_lower = np.exp(log_or - 1.96 * log_or_se)
                or_ci_upper = np.exp(log_or + 1.96 * log_or_se)
                prob_superiority = 1 - stats.norm.cdf(0, log_or, log_or_se) if or_crude > 1 else stats.norm.cdf(0, log_or, log_or_se)
                
                results = {
                    'frequentist': {
                        'or_mean': or_crude,
                        'or_ci_lower': or_ci_lower,
                        'or_ci_upper': or_ci_upper,
                        'prob_superiority': prob_superiority,
                        'method': 'crude_or'
                    }
                }
                print(f"Frequentist analysis completed: OR = {or_crude:.3f} (95% CI: {or_ci_lower:.3f}-{or_ci_upper:.3f})")
            else:
                or_crude = ((treat_success + 0.5) * (control_fail + 0.5)) / ((treat_fail + 0.5) * (control_success + 0.5))
                results = {
                    'frequentist': {
                        'or_mean': or_crude,
                        'or_ci_lower': np.nan,
                        'or_ci_upper': np.nan,
                        'prob_superiority': 0.5,
                        'method': 'crude_or_corrected'
                    }
                }
                print(f"Frequentist analysis with continuity correction: OR = {or_crude:.3f}")
                
        except Exception as e2:
            print(f"Fallback analysis also failed: {e2}")
            results = {'analysis_error': f"Both Stan and fallback failed: {str(e)}; {str(e2)}"}
    
    
    # Create comprehensive markdown report matching antiox_secd_bin format
    md_file = os.path.join(output_dir, f'sensitivity_secd_report_{timerecovery_col}.md')
    
    with open(md_file, 'w') as f:
        f.write(f"# Sensitivity Analysis Report: {timerecovery_col}\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Outcome:** Recovered by Day {max_time} (Yes/No)\n")
        f.write(f"**Note:** Sensitivity analysis excluding Paxlovid takers - NO subgroup analyses\n\n")
        
        # Dataset summary
        f.write("## Dataset Summary\n\n")
        f.write(f"- **Total participants:** {len(analysis_data)}\n")
        f.write(f"- **Treatment group:** {analysis_data['treatment'].sum()}\n")
        f.write(f"- **Control group:** {len(analysis_data) - analysis_data['treatment'].sum()}\n\n")
        
        # Descriptive statistics
        f.write("## Descriptive Statistics\n\n")
        f.write("| Treatment Group | N | Recovered by Day 14 | Percentage | Median Time (IQR) - Recovered Only | Median Time (IQR) - All |\n")
        f.write("|---|---|---|---|---|---|\n")
        
        for group in ['Antioxidant', 'Usual Care']:
            stats = descriptive_stats[group]
            recovered_iqr = f"{stats['median_time_recovered']:.1f} ({stats['q25_time_recovered']:.1f}-{stats['q75_time_recovered']:.1f})"
            if np.isnan(stats['median_time_recovered']):
                recovered_iqr = "N/A"
            
            all_iqr = f"{stats['median_time_all']:.1f} ({stats['q25_time_all']:.1f}-{stats['q75_time_all']:.1f})"
            
            f.write(f"| {group} | {stats['n_total']} | {stats['n_recovered']} | {stats['pct_recovered']:.1f}% | {recovered_iqr} | {all_iqr} |\n")
        
        # Statistical analysis results
        if 'panoramic' in results:
            f.write("\n## Statistical Analysis Results (Bayesian)\n\n")
            f.write("### PANORAMIC-style Analysis\n")
            f.write(f"- **Odds Ratio:** {results['panoramic']['or_mean']:.3f} (95% CI: {results['panoramic']['or_ci_lower']:.3f}-{results['panoramic']['or_ci_upper']:.3f})\n")
            f.write(f"- **Probability of Superiority:** {results['panoramic']['prob_superiority']:.3f}\n\n")
        
        elif 'frequentist' in results:
            f.write("\n## Statistical Analysis Results (Frequentist)\n\n")
            f.write(f"- **Odds Ratio:** {results['frequentist']['or_mean']:.3f}")
            if not np.isnan(results['frequentist']['or_ci_lower']):
                f.write(f" (95% CI: {results['frequentist']['or_ci_lower']:.3f}-{results['frequentist']['or_ci_upper']:.3f})")
            f.write(f"\n- **Probability of Superiority:** {results['frequentist']['prob_superiority']:.3f}")
            f.write(f"\n- **Method:** {results['frequentist']['method']}\n\n")
        
        elif 'analysis_error' in results:
            f.write(f"\n## Statistical Analysis\n\nAnalysis failed: {results['analysis_error']}\n\n")
            
        # Cross-tabulation
        f.write("## Cross-tabulation\n\n")
        crosstab = pd.crosstab(analysis_data['rand_group'], analysis_data['recovered_by_day14'], margins=True)
        f.write(crosstab.to_markdown())
        f.write("\n\n")
        
        # Interpretation
        f.write("## Interpretation\n\n")
        if 'panoramic' in results:
            or_pan = results['panoramic']['or_mean']
            prob_pan = results['panoramic']['prob_superiority']
            if or_pan > 1:
                f.write(f"The antioxidant treatment shows a {((or_pan-1)*100):.1f}% increase in odds of recovery by day {max_time} ")
                f.write(f"compared to usual care (OR={or_pan:.3f}). ")
            else:
                f.write(f"The antioxidant treatment shows a {((1-or_pan)*100):.1f}% decrease in odds of recovery by day {max_time} ")
                f.write(f"compared to usual care (OR={or_pan:.3f}). ")
            f.write(f"The probability that antioxidant is superior to usual care is {prob_pan:.3f}.\n\n")
        
        elif 'frequentist' in results:
            or_freq = results['frequentist']['or_mean']
            prob_freq = results['frequentist']['prob_superiority']
            if or_freq > 1:
                f.write(f"The antioxidant treatment shows a {((or_freq-1)*100):.1f}% increase in odds of recovery by day {max_time} ")
                f.write(f"compared to usual care (OR={or_freq:.3f}). ")
            else:
                f.write(f"The antioxidant treatment shows a {((1-or_freq)*100):.1f}% decrease in odds of recovery by day {max_time} ")
                f.write(f"compared to usual care (OR={or_freq:.3f}). ")
            f.write(f"The probability that antioxidant is superior to usual care is {prob_freq:.3f}.\n\n")
        
        f.write("---\n\n")
        f.write("**Note:** Subgroup analyses are excluded from this sensitivity analysis.\n")

    print(f"Binary analysis report saved: {md_file}")
    
    # Add descriptive stats to results
    results['descriptive_stats'] = descriptive_stats
    results['outcome_col'] = timerecovery_col
    results['max_time'] = max_time
    
    return results


def exclude_paxlovid_takers(df, paxlovid_ids=PAXLOVID_TAKERS):
    """Exclude participants who took Paxlovid from the dataset."""
    n_before = df['participant_id'].nunique()
    df_filtered = df[~df['participant_id'].isin(paxlovid_ids)]
    n_after = df_filtered['participant_id'].nunique()
    print(f"Excluded {n_before - n_after} Paxlovid takers: {n_before} -> {n_after} participants")
    return df_filtered


def sensitivity_primary_analysis(antiox_primary, model, stan_model_available, output_dir='results_sensitivity'):
    """Run primary outcome analysis excluding Paxlovid takers."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS - PRIMARY OUTCOME (HOSPITALIZATION)")
    print("Excluding Paxlovid takers")
    print("="*80)
    
    # Exclude Paxlovid takers
    antiox_primary_sens = exclude_paxlovid_takers(antiox_primary)
    
    # PANORAMIC-style analysis
    analysis_data = prepare_analysis_data(antiox_primary_sens, 'pdd_hospital_binary')
    
    y = analysis_data['pdd_hospital_binary'].astype(int).values
    Z = analysis_data['rand_group'].astype(int).values.reshape(-1, 1)
    X1 = analysis_data['dem_age_calc'].astype(float).values
    X2 = analysis_data['dem_vaccination_status'].astype(int).values
    X3 = analysis_data['dem_comorb'].astype(int).values
    
    ## impute missing values with mean (for continuous) or mode (for binary)
    X1 = np.where(np.isnan(X1), np.nanmean(X1), X1)
    X2 = np.where(np.isnan(X2), np.round(np.nanmean(X2)), X2)
    X3 = np.where(np.isnan(X3), np.round(np.nanmean(X3)), X3)
    
    X1_std, X2_std, X3_std = standardize_covariates(X1, X2, X3, use_continuous_age=True)
    
    stan_data_pan = {
        'N': len(analysis_data),
        'J': 1,
        'M': 3,
        'y': y.tolist(),
        'Z': Z.reshape(-1, 1).tolist(),
        'X': np.column_stack([X1_std, X2_std, X3_std]).tolist()
    }
    
    results_pan = {}
    if stan_model_available:
        results_pan = run_stan_analysis(stan_data_pan, model, superiority_direction='lo', model_name="PANORAMIC")
    else:
        results_pan = run_frequentist_fallback(Z, X1_std, X2_std, X3_std, y)
    
    # CanTreatCOVID-style analysis
    X1_group = (X1 >= 65).astype(int)
    X1_group_std, X2_std_ctc, _ = standardize_covariates(X1_group, X2, X2, use_continuous_age=False)
    
    stan_data_ctc = {
        'N': len(analysis_data),
        'J': 1,
        'M': 2,
        'y': y.tolist(),
        'Z': Z.reshape(-1, 1).tolist(),
        'X': np.column_stack([X1_group_std, X2_std_ctc]).tolist()
    }
    
    results_ctc = {}
    if stan_model_available:
        results_ctc = run_stan_analysis(stan_data_ctc, model, superiority_direction='lo', model_name="CanTreatCOVID")
    else:
        results_ctc = run_frequentist_fallback(Z, X1_group_std, X2_std_ctc, None, y)
    
    # Generate report
    report_file = os.path.join(output_dir, 'sensitivity_primary_hospitalization_report.md')
    with open(report_file, 'w') as f:
        f.write("# Sensitivity Analysis - Primary Outcome (Hospitalization)\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Excluded Participants (Paxlovid takers):** {len(PAXLOVID_TAKERS)}\n")
        f.write(f"- Antioxidant: n=5 (1-1057, 1-1041, 1-1038, 1-1024, 1-1021)\n")
        f.write(f"- Usual Care: n=2 (1-1054, 1-1028)\n\n")
        
        f.write("## Dataset Summary\n\n")
        f.write(f"- **Total participants (after exclusion):** {len(analysis_data)}\n")
        f.write(f"- **Treatment group:** {(analysis_data['rand_group'] == 1).sum()}\n")
        f.write(f"- **Control group:** {(analysis_data['rand_group'] == 0).sum()}\n")
        f.write(f"- **Hospitalizations:** {y.sum()}\n\n")
        
        # Descriptive statistics by treatment group
        f.write("## Descriptive Statistics\n\n")
        f.write("| Treatment Group | N | Hospitalizations | Percentage |\n")
        f.write("|---|---|---|---|\n")
        
        for group in [1, 0]:
            group_data = analysis_data[analysis_data['rand_group'] == group]
            n_total = len(group_data)
            n_hosp = (group_data['pdd_hospital_binary'] == 1).sum()
            pct_hosp = (n_hosp / n_total * 100) if n_total > 0 else 0
            f.write(f"| {group} | {n_total} | {n_hosp} | {pct_hosp:.1f}% |\n")
        
        f.write("\n## Results\n\n")
        f.write("### PANORAMIC-style Analysis (Continuous Age)\n\n")
        if results_pan and 'theta_mean' in results_pan:
            f.write(f"- **Odds Ratio:** {np.exp(results_pan['theta_mean']):.3f}\n")
            f.write(f"- **95% CI:** ({np.exp(results_pan['ci_low']):.3f}, {np.exp(results_pan['ci_high']):.3f})\n")
            f.write(f"- **Probability OR < 1:** {results_pan.get('probability_superiority', 0):.3f}\n\n")
        
        f.write("### CanTreatCOVID-style Analysis (Age ≥65)\n\n")
        if results_ctc and 'theta_mean' in results_ctc:
            f.write(f"- **Odds Ratio:** {np.exp(results_ctc['theta_mean']):.3f}\n")
            f.write(f"- **95% CI:** ({np.exp(results_ctc['ci_low']):.3f}, {np.exp(results_ctc['ci_high']):.3f})\n")
            f.write(f"- **Probability OR < 1:** {results_ctc.get('probability_superiority', 0):.3f}\n\n")
    
    print(f"Sensitivity analysis report saved: {report_file}")
    return results_pan, results_ctc




def sensitivity_secondary_analysis(antiox_dd_agg, antiox_baseline, antiox_random, output_dir='results_sensitivity'):
    """Run secondary outcome analyses excluding Paxlovid takers."""
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS - SECONDARY OUTCOMES")
    print("Excluding Paxlovid takers")
    print("="*80)
    
    # Exclude Paxlovid takers from all datasets
    antiox_dd_agg_sens = exclude_paxlovid_takers(antiox_dd_agg)
    antiox_baseline_sens = exclude_paxlovid_takers(antiox_baseline)
    antiox_random_sens = exclude_paxlovid_takers(antiox_random)
    
    recovery_columns = [
        'pdd_recover_first_change_to_1',
        'pdd_recover_sustain_change_to_1', 
        'pdd_feel_today_ret_first_alleviation',
        'pdd_feel_today_ret_sustain_alleviation',
        'pdd_return_health_first_change_to_1',
        'pdd_return_health_sustain_change_to_1',
        'pdd_return_activity_first_change_to_1',
        'pdd_return_activity_sustain_change_to_1',
    ]
    
    all_results = {}
    
    for col in recovery_columns:
        print(f"\n{'='*60}")
        print(f"Processing: {col}")
        print('='*60)
        
        try:
            results = secondary_binary_analysis_no_subgroups(
                antiox_dd_agg_sens, 
                antiox_baseline_sens, 
                antiox_random_sens, 
                col, 
                max_time=14, 
                output_dir=output_dir
            )
            all_results[col] = results
        except Exception as e:
            print(f"Error processing {col}: {e}")
            all_results[col] = {'error': str(e)}
    
    return all_results


def main():
    """Main execution function."""
    
    # Create results directory
    output_dir = '/workspaces/CTC_covid/py_src/results_antiox/results_sensitivity_no_paxlovid'
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("SENSITIVITY ANALYSIS - EXCLUDING PAXLOVID TAKERS")
    print("="*80)
    
    # Load and prepare data
    print("\nLoading data...")
    df_antiox = load_and_clean_raw_data("/workspaces/CTC_covid/data/CSV_RCC_Data_Export_ALL_Final_2025-05-15-Antiox.csv")
    
    antiox_random = extract_randomization_data(df_antiox)
    antiox_baseline = extract_baseline_data(df_antiox, antiox_random)
    antiox_followup = extract_followup_data(df_antiox)
    antiox_dd, pdd_col_names = extract_diary_data(df_antiox)
    antiox_dd_agg = aggregate_diary_data(antiox_dd, pdd_col_names)
    
    # Prepare datasets for analysis
    antiox_primary = prepare_primary_outcome_dataset(antiox_dd_agg, antiox_followup, antiox_random, antiox_baseline)
    antiox_earlysus = prepare_early_recovery_dataset(antiox_dd_agg, antiox_random, antiox_baseline)
    
    # Try to compile Stan model
    stan_file = "/workspaces/CTC_covid/py_src/ctc_antiox/Stan Code.stan"
    model = None
    stan_model_available = False
    
    try:
        model = CmdStanModel(stan_file=stan_file)
        stan_model_available = True
        print("Stan model compiled successfully")
    except Exception as e:
        print(f"Could not compile Stan model: {e}")
        print("Will use frequentist fallback")
    
    # Run sensitivity analyses
    print("\n" + "="*80)
    print("RUNNING PRIMARY OUTCOME ANALYSES")
    print("="*80)
    
    results_primary = sensitivity_primary_analysis(
        antiox_primary, 
        model, 
        stan_model_available, 
        output_dir=output_dir
    )
    
    # results_recovery = sensitivity_early_recovery_analysis(
    #     antiox_earlysus, 
    #     model, 
    #     stan_model_available, 
    #     output_dir=output_dir
    # )
    
    print("\n" + "="*80)
    print("RUNNING SECONDARY OUTCOME ANALYSES")
    print("="*80)
    
    results_secondary = sensitivity_secondary_analysis(
        antiox_dd_agg, 
        antiox_baseline, 
        antiox_random, 
        output_dir=output_dir
    )

if __name__ == '__main__':
    main()
