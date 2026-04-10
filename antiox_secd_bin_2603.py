
### secondary analysis using stan binary logistic regression models
# For each recovery outcome, create binary variable: recovered by day 14 (yes=1, no=0)
# Report: numbers (percentage), median time (IQR), odds ratio (95%CI), probability of superiority

import pandas as pd
import numpy as np
from cmdstanpy import CmdStanModel
import os
import warnings
warnings.filterwarnings('ignore')

# from src_helper import first_change_to_1, ret_first_alleviation, ret_sustain_alleviation, sustain_change_to_1, ret_series, ret_indexof1
from antiox_primary_2603 import load_and_clean_raw_data, extract_randomization_data, extract_baseline_data, extract_followup_data, extract_diary_data, aggregate_diary_data, standardize_covariates
STAN_FILE_RECOVERY = "/workspaces/CTC_covid/py_src/ctc_antiox/Stan Code Recovery.stan"
STAN_FILE_REGULARIZING = "/workspaces/CTC_covid/py_src/ctc_antiox/Stan Code Regularizing.stan"

def secondary_binary_analysis(antiox_dd_agg, 
                              antiox_baseline, 
                              antiox_random, 
                              timerecovery_col, 
                              max_time=14,
                              subgroup_analysis=True, 
                              output_dir='results_antiox'):
    """
    Binary logistic regression analysis for recovery outcomes by day 14 with subgroup analyses
    
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
    subgroup_analysis : bool
        Whether to perform subgroup analyses (default True)
        
    Returns:
    --------
    dict : Dictionary with analysis results including subgroup results
    """
    
    # Create output directory if specified
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n=== Binary Analysis for {timerecovery_col} ===")
    
    # Merge datasets
    antiox_secondary = antiox_dd_agg[['participant_id', timerecovery_col]].merge(
        antiox_baseline[['participant_id', 'dem_age_calc', 'dem_vaccination_status', 'dem_comorb', 'dem_sex', 'dem_race', 'dem_house_income']], 
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
    def calculate_descriptive_stats(df, time_col):
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
    
    results = {}
    descriptive_stats = calculate_descriptive_stats(analysis_data, timerecovery_col)
    # Add descriptive stats to results
    results['descriptive_stats'] = descriptive_stats
    results['outcome_col'] = timerecovery_col
    results['max_time'] = max_time
    
    
    # Prepare variables for Stan analysis
    y = analysis_data['recovered_by_day14'].values.astype(int)
    treatment = analysis_data['treatment'].values.astype(int)
    age = analysis_data['dem_age_calc'].values.astype(float)
    vaccination_status = analysis_data['dem_vaccination_status'].values.astype(int)
    comorb = analysis_data['dem_comorb'].values.astype(int)
    
    # Handle missing values for age vaccination status, and comorbidity by imputation (mean for age, mode for categorical)
    age = np.where(np.isnan(age), np.nanmean(age), age)
    vaccination_status = np.where(np.isnan(vaccination_status), np.round(np.nanmean(vaccination_status)), vaccination_status)
    comorb = np.where(np.isnan(comorb), np.round(np.nanmean(comorb)), comorb)
    
    # PANORAMIC-style analysis (continuous age + standardization)
    age_std, vaccination_status_std, comorb_std = standardize_covariates(age, vaccination_status, comorb)
    
    # # CanTreatCOVID-style analysis (age dichotomized at 65)
    # age_group_std = age_group - np.mean(age_group)
    # vaccination_status_ctc_std = vaccination_status - np.mean(vaccination_status)
    
    
    try:
        # First check if Stan is available
        model = CmdStanModel(stan_file=STAN_FILE_RECOVERY)
        
        # PANORAMIC analysis
        stan_data_pan = {
            'N': len(analysis_data),
            'J': 1,  # number of interventions
            'M': 3,  # number of covariates
            'y': y.tolist(),
            'Z': treatment.reshape(-1, 1).tolist(),
            'X': np.column_stack([age_std, vaccination_status_std, comorb_std]).tolist()
        }
        
        # # CanTreatCOVID analysis
        # stan_data_can = {
        #     'N': len(analysis_data),
        #     'J': 1,  # number of interventions  
        #     'M': 2,  # number of covariates
        #     'y': y.tolist(),
        #     'Z': treatment.reshape(-1, 1).tolist(),
        #     'X': np.column_stack([age_group_std, vaccination_status_ctc_std]).tolist()
        # }
        
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
        
        # # Run CanTreatCOVID model  
        # print("Running CanTreatCOVID-style analysis...")
        # fit_can = model.sample(data=stan_data_can, chains=4, iter_sampling=20000, iter_warmup=10000, seed=42)
        # draws_can = fit_can.draws_pd()
        
        # # Extract CanTreatCOVID results
        # theta_can = draws_can['theta[1]']
        # or_samples_can = np.exp(theta_can)
        # or_mean_can = np.mean(or_samples_can)
        # or_ci_lower_can = np.percentile(or_samples_can, 2.5)
        # or_ci_upper_can = np.percentile(or_samples_can, 97.5)
        # prob_superiority_can = np.mean(or_samples_can > 1)
        
        results = {
            'panoramic': {
                'or_mean': or_mean_pan,
                'or_ci_lower': or_ci_lower_pan,
                'or_ci_upper': or_ci_upper_pan,
                'prob_superiority': prob_superiority_pan
            },
            # 'cantreatcovid': {
            #     'or_mean': or_mean_can,
            #     'or_ci_lower': or_ci_lower_can, 
            #     'or_ci_upper': or_ci_upper_can,
            #     'prob_superiority': prob_superiority_can
            # }
        }
        
        print("Stan analysis completed successfully")

        # Regularizing prior sensitivity analysis (normal(0,1) for theta and bbeta)
        try:
            model_reg = CmdStanModel(stan_file=STAN_FILE_REGULARIZING)
            stan_data_reg = {**stan_data_pan, 'alpha_loc': 0.0}
            print("Running regularizing prior sensitivity analysis...")
            fit_reg = model_reg.sample(data=stan_data_reg, 
                                       chains=4, 
                                       iter_sampling=20000, 
                                       iter_warmup=10000, 
                                       seed=42)
            
            draws_reg = fit_reg.draws_pd()
            theta_reg = draws_reg['theta[1]']
            or_samples_reg = np.exp(theta_reg)
            results['regularizing'] = {
                'or_mean': float(np.mean(or_samples_reg)),
                'or_ci_lower': float(np.percentile(or_samples_reg, 2.5)),
                'or_ci_upper': float(np.percentile(or_samples_reg, 97.5)),
                'prob_superiority': float(np.mean(or_samples_reg > 1))
            }
            print("Regularizing sensitivity completed successfully")
        except Exception as e_reg:
            print(f"Regularizing sensitivity failed: {e_reg}")
            
        
    except Exception as e:
        print(f"Stan analysis failed: {e}")
        print("Using fallback frequentist logistic regression analysis...")
        
        # Fallback to scipy logistic regression with bootstrap for confidence intervals
        from scipy import stats
        
        try:
            # Frequentist analysis as fallback
            # PANORAMIC-style model
            X_pan = np.column_stack([treatment, age_std, vaccination_status_std, comorb_std])
            
            # Simple cross-tabulation for odds ratio calculation  
            # Create 2x2 table for treatment vs outcome
            treat_success = np.sum((treatment == 1) & (y == 1))
            treat_total = np.sum(treatment == 1)
            treat_fail = treat_total - treat_success
            
            control_success = np.sum((treatment == 0) & (y == 1))  
            control_total = np.sum(treatment == 0)
            control_fail = control_total - control_success
            
            # Calculate crude odds ratio with confidence interval
            if treat_fail > 0 and control_fail > 0 and treat_success > 0 and control_success > 0:
                or_crude = (treat_success * control_fail) / (treat_fail * control_success)
                
                # Log OR standard error for confidence interval
                log_or_se = np.sqrt(1/treat_success + 1/treat_fail + 1/control_success + 1/control_fail)
                log_or = np.log(or_crude)
                
                # 95% CI for OR
                or_ci_lower = np.exp(log_or - 1.96 * log_or_se)
                or_ci_upper = np.exp(log_or + 1.96 * log_or_se)
                
                # Simple probability of superiority (assuming normal approximation)
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
                # Handle zero cells with continuity correction
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
    
    # # Save analysis dataset
    # analysis_data.to_csv(os.path.join(output_dir, f'binary_analysis_dataset_{timerecovery_col}.csv'), index=False)

    # ---------------------------------------
    # Perform subgroup analysis
    # ---------------------------------------
    if subgroup_analysis:
        model = CmdStanModel(stan_file=STAN_FILE_RECOVERY)
        subgroup_results = subgroup_binary_analysis(analysis_data, timerecovery_col, 'recovered_by_day14', model)
    
    # ---------------------------------------
    # Create comprehensive markdown report
    # ---------------------------------------
    md_file = os.path.join(output_dir, f'binary_secd_report_{timerecovery_col}.md')
    
    with open(md_file, 'w') as f:
        f.write(f"# Binary Analysis Report: {timerecovery_col}\n\n")
        f.write(f"**Analysis Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Outcome:** Recovered by Day {max_time} (Yes/No)\n\n")
        
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

            if 'regularizing' in results:
                f.write("### Sensitivity Analysis (Regularizing Priors: normal(0,1))\n")
                f.write(f"- **Odds Ratio:** {results['regularizing']['or_mean']:.3f} (95% CI: {results['regularizing']['or_ci_lower']:.3f}-{results['regularizing']['or_ci_upper']:.3f})\n")
                f.write(f"- **Probability of Superiority:** {results['regularizing']['prob_superiority']:.3f}\n\n")
            
            # f.write("### CanTreatCOVID-style Analysis\n")
            # f.write(f"- **Odds Ratio:** {results['cantreatcovid']['or_mean']:.3f} (95% CI: {results['cantreatcovid']['or_ci_lower']:.3f}-{results['cantreatcovid']['or_ci_upper']:.3f})\n")
            # f.write(f"- **Probability of Superiority:** {results['cantreatcovid']['prob_superiority']:.3f}\n\n")
        
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
        
        
        # # Subgroup results to md
        # if subgroup_analysis and subgroup_results:
        #     f.write("## Subgroup Analyses\n\n")
        #     f.write("| Subgroup | N | Antioxidant | Usual Care | antiox Recovered | % antiox Recovered | uc Recovered | % uc Recovered | Log-OR | OR | 95% CI | Prob(OR>1) | Notes |\n")
        #     f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
            
        #     for subgroup_result in subgroup_results:
        #         n = subgroup_result['n']
        #         n_ant = subgroup_result['n_antioxidant']
        #         n_uc = subgroup_result['n_usual_care']
        #         antiox_rec = int(subgroup_result['antiox_recovered'])
        #         antiox_pct = subgroup_result['antiox_pct_recovered']
        #         uc_rec = int(subgroup_result['uc_recovered'])
        #         uc_pct = subgroup_result['uc_pct_recovered']
        #         log_or_val = subgroup_result['log_or']
        #         or_val = subgroup_result['or']
        #         ci_lower = subgroup_result['ci_lower']
        #         ci_upper = subgroup_result['ci_upper']
        #         prob_sup = subgroup_result['prob_superiority']
        #         note = subgroup_result.get('note', '')
                
        #         # Format values
        #         log_or_str = f"{log_or_val:.3f}" if not np.isnan(log_or_val) else "N/A"
        #         or_str = f"{or_val:.3f}" if not np.isnan(or_val) else "N/A"
        #         ci_str = f"({ci_lower:.3f}-{ci_upper:.3f})" if not np.isnan(ci_lower) and not np.isnan(ci_upper) else "N/A"
        #         prob_str = f"{prob_sup:.3f}" if not np.isnan(prob_sup) else "N/A"

        #         f.write(f"| {subgroup_result['subgroup']} | {n} | {n_ant} | {n_uc} | {antiox_rec} | {antiox_pct:.2f}% | {uc_rec} | {uc_pct:.2f}% | {log_or_str} | {or_str} | {ci_str} | {prob_str} | {note} |\n")
    
    return results, subgroup_results if subgroup_analysis else None


def subgroup_binary_analysis(antiox_secondary, timerecovery_col, outcome_col, stan_file):
    """
    Perform subgroup analyses for secondary outcomes
    
    Parameters:
    -----------
    antiox_secondary : DataFrame
        Dataset with all required variables for analysis
    timerecovery_col : str
        Column name for time to recovery
    outcome_col : str
        Binary outcome column name
    stan_file : str
        Path to Stan model file
        
    Returns:
    --------
    list : List of subgroup result dictionaries
    """
    
    subgroup_results = []
    
    # Define subgroups
    subgroups = {
        # 'Age <= 50': antiox_secondary['dem_age_calc'] <= 50,
        'Age > 50': antiox_secondary['dem_age_calc'] > 50,
        'Age > 65': antiox_secondary['dem_age_calc'] > 65,
        # 'Sex = 1 (Male)': antiox_secondary['dem_sex'] == 1,
        # 'Sex = 2 (Female)': antiox_secondary['dem_sex'] == 2,
        # 'Race = 1 (White)': antiox_secondary['dem_race'] == 1,
        # 'Race != 1 (Non-White)': antiox_secondary['dem_race'] != 1,
        # 'Income <= 5': antiox_secondary['dem_house_income'] <= 5,
        # 'Income > 5': antiox_secondary['dem_house_income'] > 5,
        'Comorbidity = 1 (Yes)': antiox_secondary['dem_comorb'] == 1,
        # 'Comorbidity = 0 (No)': antiox_secondary['dem_comorb'] == 0,
        'Vaccination >= 2': antiox_secondary['dem_vaccination_status'] >= 2,
        # 'Vaccination < 2': antiox_secondary['dem_vaccination_status'] < 2,
    }
    
    try:
        model = CmdStanModel(stan_file=stan_file)
    except Exception as e:
        print(f"Warning: Could not load Stan model for subgroup analysis: {e}")
        model = None
    
    for subgroup_name, subgroup_mask in subgroups.items():
        print(f"Analyzing subgroup: {subgroup_name}")
        subgroup_data = antiox_secondary[subgroup_mask].dropna(subset=[outcome_col, 'treatment'])
        
        if len(subgroup_data) < 10 or subgroup_data['treatment'].nunique() < 2:
            subgroup_results.append({
                'subgroup': subgroup_name,
                'n': len(subgroup_data),
                'n_antioxidant': len(subgroup_data[subgroup_data['treatment'] == 1]),
                'n_usual_care': len(subgroup_data[subgroup_data['treatment'] == 0]),
                'antiox_recovered': 0,
                'antiox_pct_recovered': 0,
                'uc_recovered': 0,
                'uc_pct_recovered': 0,
                'log_or': np.nan,
                'or': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'prob_superiority': np.nan,
                'note': 'Insufficient data or missing treatment arm'
            })
            continue
        
        # Split by treatment
        treat_data = subgroup_data[subgroup_data['treatment'] == 1]
        control_data = subgroup_data[subgroup_data['treatment'] == 0]
        
        y_sub = subgroup_data[outcome_col].values.astype(int)
        treatment_sub = subgroup_data['treatment'].values.astype(int)
        age_sub = subgroup_data['dem_age_calc'].values.astype(float)
        vax_sub = subgroup_data['dem_vaccination_status'].values.astype(int)
        comorb_sub = subgroup_data['dem_comorb'].values.astype(int)
        
        # Standardize covariates
        age_mean = np.nanmean(age_sub)
        age_sub = np.where(np.isnan(age_sub), age_mean, age_sub)
        age_std_sub = (age_sub - np.mean(age_sub)) / (4 * np.var(age_sub))**0.5 if np.var(age_sub) > 0 else (age_sub - np.mean(age_sub))
        vax_std_sub = vax_sub - np.mean(vax_sub)
        comorb_std_sub = comorb_sub - np.mean(comorb_sub)
        
        result_entry = {
            'timerecovery_col': timerecovery_col,   
            'subgroup': subgroup_name,
            'n': len(subgroup_data),
            'n_antioxidant': len(treat_data),
            'n_usual_care': len(control_data),
            'antiox_recovered': treat_data[outcome_col].sum(),
            'antiox_pct_recovered': (treat_data[outcome_col].sum() / len(treat_data) * 100) if len(treat_data) > 0 else 0,
            'uc_recovered': control_data[outcome_col].sum(),
            'uc_pct_recovered': (control_data[outcome_col].sum() / len(control_data) * 100) if len(control_data) > 0 else 0,
        }
        
        # Try Stan analysis
        if model is not None and len(subgroup_data) > 10:
            try:
                stan_data_sub = {
                    'N': len(subgroup_data),
                    'J': 1,
                    'M': 3,
                    'y': y_sub.tolist(),
                    'Z': treatment_sub.reshape(-1, 1).tolist(),
                    'X': np.column_stack([age_std_sub, vax_std_sub, comorb_std_sub]).tolist()
                }
                
                fit_sub = model.sample(data=stan_data_sub, chains=4, iter_sampling=10000, iter_warmup=5000, seed=42, show_console=False)
                draws_sub = fit_sub.draws_pd()
                theta_sub = draws_sub['theta[1]']
                log_or_sub = theta_sub.mean()
                or_sub = np.exp(log_or_sub)
                ci_lower_sub = np.exp(theta_sub.quantile(0.025))
                ci_upper_sub = np.exp(theta_sub.quantile(0.975))
                prob_sup_sub = (theta_sub > 0).mean()
                
                result_entry.update({
                    'log_or': log_or_sub,
                    'or': or_sub,
                    'ci_lower': ci_lower_sub,
                    'ci_upper': ci_upper_sub,
                    'prob_superiority': prob_sup_sub
                })
            except Exception as e:
                # Fallback to frequentist
                result_entry.update({
                    'log_or': np.nan,
                    'or': np.nan,
                    'ci_lower': np.nan,
                    'ci_upper': np.nan,
                    'prob_superiority': np.nan,
                    'note': f'Stan failed, using frequentist fallback'
                })
        else:
            result_entry.update({
                'log_or': np.nan,
                'or': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'prob_superiority': np.nan,
                'note': 'Insufficient data for Stan'
            })
        
        subgroup_results.append(result_entry)
    
    return subgroup_results



if __name__ == '__main__':
    
    
    # List of time-to-recovery columns to analyze
    recovery_columns = [
        'pdd_recover_first_change_to_1',
        'pdd_recover_sustain_change_to_1', 
        'pdd_feel_today_ret_first_alleviation',
        'pdd_feel_today_ret_sustain_alleviation',
        'pdd_return_health_first_change_to_1',
        'pdd_return_health_sustain_change_to_1',
        'pdd_return_activity_first_change_to_1',
        'pdd_return_activity_sustain_change_to_1'
    ]
    
    # Run analysis for each recovery column
    all_results = {}
    all_subgroup_results = []
    
    df_antiox = load_and_clean_raw_data("/workspaces/CTC_covid/data/CSV_RCC_Data_Export_ALL_Final_2025-05-15-Antiox.csv")
    # Extract key datasets
    antiox_random = extract_randomization_data(df_antiox)
    antiox_baseline = extract_baseline_data(df_antiox, antiox_random)
    antiox_followup = extract_followup_data(df_antiox)
    antiox_dd, pdd_col_names = extract_diary_data(df_antiox)
    antiox_dd_agg = aggregate_diary_data(antiox_dd, pdd_col_names)
    
    
    for col in recovery_columns:
        if col in antiox_dd_agg.columns:
            print(f"\n{'='*80}")
            try:
                col_results, col_subgroup_result = secondary_binary_analysis(
                    antiox_dd_agg, antiox_baseline, antiox_random, 
                    timerecovery_col=col, 
                    max_time=14, 
                    subgroup_analysis=True, 
                    output_dir='/workspaces/CTC_covid/py_src/results_antiox'
                )
                all_results[col] = col_results
                all_subgroup_results.extend(col_subgroup_result)
                
            except Exception as e:
                print(f"Error analyzing {col}: {e}")
                continue
        else:
            print(f"Column {col} not found in dataset")
            
    ## save subgroup results to csv
    if all_subgroup_results is not None:
        subgroup_df = pd.DataFrame(all_subgroup_results)
        subgroup_df.to_csv(f'/workspaces/CTC_covid/py_src/results_antiox/secd_subgroup_results_all.csv', index=False, float_format='%.3f')

    
