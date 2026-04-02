
import numpy as np
import pandas as pd


def replace_empty_with_none(df):
    """
    Replace empty strings in the DataFrame with None.
    """
    for col_name in df.columns:
        df[col_name] = df[col_name].replace(['', 'NA', 'N/A', 'na', 'NaN'], None)
    return df

# Read CSV directly with pandas
df_antiox = pd.read_csv("/workspaces/CTC_covid/data/CSV_RCC_Data_Export_ALL_Final_2025-05-15-Antiox.csv")

rcc_ids_in_antiox = (
    df_antiox[df_antiox['redcap_event_name'] == 'Randomization']
    .sort_values('rand_date')
    ['participant_id']
    .head(5)
    .tolist()
)
rcc_ids_in_antiox.pop(3)

# Drop all the rows with participant_id in rcc_ids_in_antiox
df_antiox = df_antiox[~df_antiox['participant_id'].isin(rcc_ids_in_antiox)]



# #### followup table to figure out follow completion: 68 participants
# antiox_followup_eligible = df_antiox[df_antiox['redcap_event_name'].isin(['Baseline','Day 1','Day 6 (Antioxidant only)', 'Day 10 (Antioxidant only)','Day 21','Day 28'])]
# antiox_followup_eligible = antiox_followup_eligible.dropna(axis=1, how='all')
# antiox_followup_eligible = replace_empty_with_none(antiox_followup_eligible)

# ## keep all columns end with _complete
# antiox_followup_eligible = antiox_followup_eligible[['participant_id',
#  'Follow - up at day 1_complete',
#  'Follow - up at day 6_complete',
#  'Follow - up at day 10_complete',
#  'Follow up Day 28_complete',
#  'Follow up Day 21_complete',
#  'Visit Date_complete']]

# antiox_followup_eligible_agg = antiox_followup_eligible.groupby('participant_id').agg('first').reset_index()

# id_to_remove = (
#     df_antiox[df_antiox['end_study_reason'].isin([4,6])]['participant_id'].to_list()
# )

# df_antiox = df_antiox[~df_antiox['participant_id'].isin(id_to_remove)]




#########################################################################
######################### randomization table #############################
##########################################################################


event_antiox = df_antiox['redcap_event_name'].unique().tolist()

antiox_random = df_antiox[df_antiox['redcap_event_name'] == 'Randomization']
antiox_random = antiox_random.dropna(axis=1, how='all')
antiox_random = replace_empty_with_none(antiox_random)
antiox_random.head()
## 
antiox_random.redcap_study.value_counts()


##############################################################
###############################################################
antiox_end = df_antiox[df_antiox['redcap_event_name'] == 'Study Completion/Termination']
antiox_end = antiox_end.dropna(axis=1, how='all')
antiox_end = replace_empty_with_none(antiox_end)

antiox_end.head()
antiox_end.end_study_reason.value_counts()



############################ summarise took 5,10 doses ###############
###################################################################

antiox_treatment = df_antiox[df_antiox['redcap_event_name']=='Study treatment']
antiox_treatment = antiox_treatment.dropna(axis=1, how='all')
antiox_treatment = replace_empty_with_none(antiox_treatment)
print(antiox_treatment.shape)

antiox_treatment.end_treat_yn.value_counts()
antiox_treatment.end_treat_reason.value_counts()


##########################
########################
#############################

antiox_dd = df_antiox[df_antiox['redcap_event_name'].str.contains('Diar', na=False)]
antiox_dd = antiox_dd.dropna(axis=1, how='all')
antiox_dd = replace_empty_with_none(antiox_dd)
print(antiox_dd.shape)
antiox_dd.head()


# Combine pdd_ and fpp_ columns using combine_first
# First identify matching column pairs
pdd_columns = [col for col in antiox_dd.columns if col.startswith('pdd_')]
fpp_columns = [col for col in antiox_dd.columns if col.startswith('fpp_')]

# Create mapping of pdd to fpp columns
column_mapping = {}
for pdd_col in pdd_columns:
    # Extract the suffix after 'pdd_'
    suffix = pdd_col[4:]  # Remove 'pdd_' prefix
    fpp_col = f'fpp_{suffix}'
    if fpp_col in fpp_columns:
        column_mapping[pdd_col] = fpp_col

# Combine columns using combine_first
for pdd_col, fpp_col in column_mapping.items():
    antiox_dd[pdd_col] = antiox_dd[pdd_col].combine_first(antiox_dd[fpp_col])

# Drop the fpp_ columns after combining
antiox_dd = antiox_dd.drop(columns=list(column_mapping.values()))


antiox_dd_agg = pd.DataFrame() 
antiox_dd_agg['num_med_doses'] = antiox_dd.groupby('participant_id')['pdd_study_meds'].agg(lambda x: (x==1).sum())

antiox_dd_agg['take_over_5doses'] = antiox_dd_agg['num_med_doses'] >= 5
antiox_dd_agg['take_over_10doses'] = antiox_dd_agg['num_med_doses'] >= 10


#########################################################################
######################### baseline table #############################
##########################################################################
antiox_baseline = df_antiox[df_antiox['redcap_event_name'] == 'Baseline']
antiox_baseline = antiox_baseline.dropna(axis=1, how='all')
antiox_baseline = replace_empty_with_none(antiox_baseline)

antiox_baseline = antiox_baseline.merge(antiox_random[['participant_id', 'rand_group','symp_onset_date','rand_date']], on='participant_id', how='left')

antiox_baseline = antiox_baseline.merge(antiox_dd_agg, on='participant_id', how='left')
# Drop columns start with wk or wend
antiox_baseline = antiox_baseline.loc[:, ~antiox_baseline.columns.str.startswith(('wk', 'wend'))]

antiox_baseline['symp_onset_date'] = pd.to_datetime(antiox_baseline['symp_onset_date'], errors='coerce')
antiox_baseline['visit_date'] = pd.to_datetime(antiox_baseline['visit_date'], errors='coerce')

antiox_baseline['duration_symp'] = antiox_baseline['visit_date']- antiox_baseline['symp_onset_date']
## change duration_symp to numbers
antiox_baseline['duration_symp'] = antiox_baseline['duration_symp'].dt.days
antiox_baseline.head()
antiox_baseline['dem_age_calc'].describe()
antiox_baseline[antiox_baseline['dem_age_calc']==0] ## participand_id: 1002
# print(antiox_baseline.loc[antiox_baseline['participant_id']=='1-1002','dem_age_calc'])

antiox_baseline.loc[antiox_baseline['dem_age_calc']==0, 'dem_age_calc'] =  (pd.to_datetime(antiox_baseline.loc[antiox_baseline['dem_age_calc']==0, 'rand_date'])  -  pd.to_datetime(antiox_baseline.loc[antiox_baseline['dem_age_calc']==0, 'dem_dob'])).dt.days//365.2425


## adjust ethnicity - map all ethnicity codes, combining Asian categories (6,7,8) into 4
antiox_baseline['dem_race'] = antiox_baseline['dem_race'].map({
    1:1,  # White
    2:2,  # Black
    3:3,  # Latino
    4:4,  # Asian
    5:5,  # Indigenous
    6:4,  # Map to Asian
    7:4,  # Map to Asian
    8:4,  # Map to Asian
    9:9,  # Mixed
    99:99 # Other
})

antiox_baseline['any_symptom_rated23'] = antiox_baseline[['covid_fever','covid_cough', 'covid_sob',  'covid_taste','covid_muscle_ache','covid_nausea',  'covid_fatigue1', 'covid_concetrate', 'covid_mood']].apply(lambda x: 1 if x.isin([2,3]).any() else (0 if x.notna().any() else None), axis=1)



def ret_any_one(row: pd.Series):
    if row.empty:
        return None
    elif 1 in row.values:
        return 1
    return 0

#### combine comorbidities
antiox_baseline['lung_disease'] = antiox_baseline[['chronic_disease_list___2','chronic_disease_list___9']].apply(ret_any_one, axis=1)

antiox_baseline['liver_disease'] = antiox_baseline[['chronic_disease_list___5','chronic_disease_list___15','chronic_disease_list___16']].apply(ret_any_one, axis=1)

antiox_baseline['neurological_disease'] = antiox_baseline[['chronic_disease_list___7','chronic_disease_list___8','chronic_disease_list___10','chronic_disease_list___12','chronic_disease_list___21']].apply(ret_any_one, axis=1)

antiox_baseline['heart_disease'] = antiox_baseline[['chronic_disease_list___26','chronic_disease_list___27','chronic_disease_list___28','chronic_disease_list___29','chronic_disease_list___18']].apply(ret_any_one, axis=1)

antiox_baseline['any_disease'] = antiox_baseline[['chronic_disease_list___1','chronic_disease_list___2','chronic_disease_list___3','chronic_disease_list___4','chronic_disease_list___5','chronic_disease_list___6','chronic_disease_list___7','chronic_disease_list___8','chronic_disease_list___9','chronic_disease_list___10','chronic_disease_list___11','chronic_disease_list___12','chronic_disease_list___13','chronic_disease_list___15','chronic_disease_list___16','chronic_disease_list___17','chronic_disease_list___18','chronic_disease_list___19','chronic_disease_list___20','chronic_disease_list___21','chronic_disease_list___22','chronic_disease_list___23','chronic_disease_list___24','chronic_disease_list___25','chronic_disease_list___26','chronic_disease_list___27','chronic_disease_list___28','chronic_disease_list___29','chronic_disease_list___99','diagnose_addict']].apply(ret_any_one, axis=1)
##  Diabetes
##  High blood pressure (Hypertension)
##  Kidney disease
##  Obesity





## summarize the baseline table: 
"""
Table 1: Baseline characteristics
Antioxidant therapy (N= ?)	Control (N=?)
Took at least 5 doses study treatment, n(%)
Missing, n(%)
"""


def summarize_baseline(df, out_csv='/workspaces/CTC_covid/py_src/results_antiox/antiox_baseline_table1.csv'):
    """Build Table 1-style summary for available baseline columns.

    Strategy: convert the pandas-on-spark DataFrame to pandas, autodetect likely
    column names from common candidates, compute per-rand_group counts and statistics,
    save a wide CSV with one row per metric and one column per group plus Overall.
    """
    ## df = antiox_baseline

    if 'rand_group' not in df.columns:
        raise RuntimeError('Expected column "rand_group" in merged baseline DataFrame')

    groups = df['rand_group'].fillna('Unknown').unique().tolist()

    # helper to compute n and percent for values
    def n_pct(sub, val=None):
        """Return string 'n (p%)' for a Series `sub`.

        - If `sub` is a boolean Series (e.g. produced by isnull()), count True as n and denom is length of series.
        - If `val` is provided, count (sub == val) with denom = non-missing entries in sub.
        - If `val` is None, return count of non-missing and percent of non-missing over total rows.
        """
        if sub is None:
            return ''
        try:
            s = sub.dropna() if hasattr(sub, 'dropna') else pd.Series(sub).dropna()
        except Exception:
            # fallback
            s = pd.Series(sub)

        # If original object is a boolean Series (we can detect via dtype or by checking values)
        if pd.api.types.is_bool_dtype(sub):
            # count True values; denom is total length (including False)
            n = int(sub.sum())
            denom = int(len(sub))
            pct = n / denom * 100 if denom > 0 else np.nan
            return f"{n} ({pct:.1f}%)" if not np.isnan(pct) else f"{n}"

        if val is None:
            # number non-missing and percent of total rows
            n = int(s.notna().sum()) if hasattr(s, 'notna') else int(s.count())
            denom = int(len(sub)) if hasattr(sub, '__len__') else int(len(s))
            pct = n / denom * 100 if denom > 0 else np.nan
            return f"{n} ({pct:.1f}%)" if not np.isnan(pct) else f"{n}"

        # val provided: count matches among non-missing values
        try:
            n = int((sub == val).sum())
            denom = int(sub.notna().sum()) if hasattr(sub, 'notna') else int(len(sub.dropna()))
            pct = n / denom * 100 if denom > 0 else np.nan
            return f"{n} ({pct:.1f}%)" if not np.isnan(pct) else f"{n}"
        except Exception:
            return ''

    # record which columns we used
    used = {
        'age': 'dem_age_calc',
        'sex': 'dem_sex',
        'ethnicity': 'dem_race',
        'duration_symptoms': 'duration_symp',
        'took_5_doses': 'take_over_5doses',
        'took_10_doses': 'take_over_10doses',
        'vax_doses': 'dem_vaccination_status',
        'bmi': 'bmi',
        # symptoms
        'fever': 'covid_fever',
        'cough': 'covid_cough',
        'sob': 'covid_sob',
        'loss_smell': 'covid_taste',
        'muscle_ache': 'covid_muscle_ache',
        'nausea': 'covid_nausea',
        'fatigue': 'covid_fatigue1',
        'concentration': 'covid_concetrate',
        'anxious': 'covid_mood',
        'any_symptom_rated_moderate_or_major': 'any_symptom_rated23',
        # comorbidities
        'lung disease': 'lung_disease',
        'liver disease': 'liver_disease',
        'neurological disease': 'neurological_disease',
        'heart disease': 'heart_disease',
        'diabetes': 'chronic_disease_list___11',
        'hypertension' : 'chronic_disease_list___19',
        'kidney disease': 'chronic_disease_list___20',
        'obesity': 'chronic_disease_list___22',
        'mental illness': 'diagnose_addict',
        'any disease':'any_disease'
    }

    # Build rows
    rows = []
    # N per group
    counts = df.groupby('rand_group')['participant_id'].nunique()
    counts = counts.reindex(groups).fillna(0).astype(int)
    overall_n = df['participant_id'].nunique()
    header = ['metric'] + [f"{g} (N={counts.loc[g]})" for g in groups] + [f"Overall (N={overall_n})"]

    def add_row_scalar(label, func):
        vals = []
        for g in groups:
            sub = df[df['rand_group'] == g]
            vals.append(func(sub))
        # overall
        vals.append(func(df))
        rows.append([label] + vals)

    # Age
    age_col = used.get('age')
    if age_col:
        def age_stat(sub):
            s = pd.to_numeric(sub.get(age_col, pd.Series(dtype=float)), errors='coerce').dropna()
            if len(s) == 0:
                return ''
            return f"{s.mean():.1f} ({s.std(ddof=0):.1f}) [{s.min():.0f},{s.max():.0f}]"
        add_row_scalar('Age, mean(SD) [min,max]', age_stat)
    else:
        rows.append(['Age, mean(SD) [min,max]'] + [''] * (len(groups) + 1))

    # Sex
    sex_col = used.get('sex')
    if sex_col:
        # female, male, other, missing
        def sex_counts_for(sub, label):
            s = sub.get(sex_col)
            if s is None:
                return ''
            s = s.fillna('Missing')
            return n_pct(s, label)
        
        sex_dict = {
            1: 'Male',
            2: 'Female',
            3: 'Intersex'
        }
        for label in sex_dict.keys():
            add_row_scalar(f"Sex: {sex_dict[label]}", lambda sub, lbl=label: sex_counts_for(sub, lbl))
        add_row_scalar("Sex: Missing", lambda sub: n_pct(sub.get(sex_col).isnull()))
    else:
        for label in sex_dict.keys():
            rows.append([f"Sex: {sex_dict[label]}"] + [''] * (len(groups) + 1))

    # Ethnicity - try to present top-level categories if available
    eth_col = used.get('ethnicity')
    if eth_col:
        
        present = {1: 'White', 4: 'Asian', 2: 'Black', 5: 'Indigeneous', 9: 'Mixed', 99: 'Other'} # 3: 'Latino',
       
        for cat in present.keys():
            add_row_scalar(f"Ethnicity: {present[cat]}", lambda sub, cat=cat: n_pct(sub.get(eth_col), cat))

        add_row_scalar("Ethnicity: Missing", lambda sub: n_pct(sub.get(eth_col).isnull()))
    else:
        rows.append(["Ethnicity: (no column found)"] + [''] * (len(groups) + 1))

    # Duration symptoms: mean(SD) and median(IQR)
    dur_col = used.get('duration_symptoms')
    if dur_col:
        def dur_mean_sd(sub):
            s = pd.to_numeric(sub.get(dur_col), errors='coerce').dropna()
            if s.empty:
                return ''
            return f"{s.mean():.1f} ({s.std(ddof=0):.1f})"
        def dur_median_iqr(sub):
            s = pd.to_numeric(sub.get(dur_col), errors='coerce').dropna()
            if s.empty:
                return ''
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            return f"{s.median():.1f} ({q1:.1f},{q3:.1f})"
        add_row_scalar('Duration symptoms at baseline in days, mean(SD)', dur_mean_sd)
        add_row_scalar('Duration symptoms at baseline in days, median(IQR)', dur_median_iqr)
    else:
        rows.append(['Duration symptoms at baseline in days, mean(SD)'] + [''] * (len(groups) + 1))
        rows.append(['Duration symptoms at baseline in days, median(IQR)'] + [''] * (len(groups) + 1))


    # Vaccine doses (categorical)
    vax_col = used.get('vax_doses')
    if vax_col:

        vax_dict = {0:'None', 1:'Less than 2 doses', 2:'2 doses or more'}
        for label in vax_dict.keys():
            add_row_scalar(f"Number of vaccine doses: {vax_dict[label]}", lambda sub, lbl=label: n_pct(sub.get(vax_col), lbl))
        add_row_scalar("Number of vaccine doses: Missing", lambda sub: n_pct(sub.get(vax_col).isnull()))
    else:
        for label in ['None', 'Less than 2', '2 or more', 'Missing']:
            rows.append([f"Number of vaccine doses: {label}"] + [''] * (len(groups) + 1))

    # BMI
    bmi_col = used.get('bmi')
    if bmi_col:
        def bmi_median_iqr(sub):
            s = pd.to_numeric(sub.get(bmi_col), errors='coerce').dropna()
            if s.empty:
                return ''
            q1 = s.quantile(0.25)
            q3 = s.quantile(0.75)
            return f"{s.median():.1f} ({q1:.1f},{q3:.1f})"
        add_row_scalar('Body Mass Index, median(IQR)', bmi_median_iqr)
    else:
        rows.append(['Body Mass Index, median(IQR)'] + [''] * (len(groups) + 1))

    # Baseline symptoms - iterate through some common symptoms
    symptom_keys = ['fever', 
                    'cough', 
                    'sob', 
                    'loss_smell',
                    'muscle_ache', 
                    'nausea', 
                    'fatigue', 
                    'concentration', 
                    'anxious']
    for key in symptom_keys:
        col = used.get(key)
        if col:
            # if categorical with levels, list levels

            levels = {0:'No problem', 1:'Mild problem', 2:'Moderate problem', 3:'Major problem'}
            
            for lev in levels.keys():
                add_row_scalar(f"{col}: {levels[lev]}", lambda sub, lev=lev, c=col: n_pct(sub.get(c), lev))
            
            add_row_scalar(f"{col}: Missing", lambda sub, c=col: n_pct(sub.get(c).isnull()))
        else:
            rows.append([f"{key}: (no column found)"] + [''] * (len(groups) + 1))

    # Comorbidities - show n(%) for each comorbidity and missing
    for key in ['took_5_doses',
                'took_10_doses',
                'any_symptom_rated_moderate_or_major',
                'lung disease',
                'liver disease',
                'neurological disease',
                'heart disease',
                'diabetes',
                'hypertension',
                'kidney disease',
                'obesity',
                'mental illness',
                'any disease'
                ]:
        col = used.get(key)
        print(col)
        if col:
            add_row_scalar(f"{key}, n(%)", lambda sub, c=col: n_pct(sub.get(c), True) if sub.get(c) is not None else '')
            add_row_scalar(f"{key}: Missing", lambda sub, c=col: n_pct(sub.get(c).isnull()))
        else:
            rows.append([f"{key}, n(%)"] + [''] * (len(groups) + 1))
            rows.append([f"{key}: Missing"] + [''] * (len(groups) + 1))

    
    # Compose DataFrame and save
    out_df = pd.DataFrame(rows, columns=header)
    out_df.to_csv(out_csv, index=False)

    # print mapping of used columns
    print('\nBaseline summary complete. Output saved to:', out_csv)
    print('Column mapping (detected -> column name or None):')
    for k, v in used.items():
        print(f"  {k}: {v}")

    return out_df


if __name__ == '__main__':
    # run the summarizer for the baseline dataframe if the script is executed directly
    try:
        summary = summarize_baseline(antiox_baseline)
    except Exception as e:
        print('Error while summarizing baseline:', str(e))



'''### chronic disease mapping:

'Arthritis (excluding fibromyalgia)': 'chronic_disease_list___1',
'Asthma': 'chronic_disease_list___2',
'Bladder disorders (e.g., incontinence, overactive bladder, cystitis)': 'chronic_disease_list___3',
'Bowel disease': 'chronic_disease_list___4',
'Liver disease or cirrhosis (non-hepatitis)': 'chronic_disease_list___5',
'Cancer (excluding non-melanoma skin cancer)': 'chronic_disease_list___6',
'Cerebrovascular diseases (e.g., stroke, TIA)': 'chronic_disease_list___7',
'Chronic pain (e.g., fibromyalgia, back pain, migraines)': 'chronic_disease_list___8',
'COPD, chronic bronchitis, emphysema': 'chronic_disease_list___9',
'Dementia (e.g., Alzheimer\'s disease)': 'chronic_disease_list___10',
'Diabetes': 'chronic_disease_list___11',
'Epilepsy/seizure disorder': 'chronic_disease_list___12',
'GERD/reflux/peptic ulcer': 'chronic_disease_list___13',
'Arrhythmia': 'chronic_disease_list___26',
'Angina': 'chronic_disease_list___27',
'Past heart attack': 'chronic_disease_list___28',
'Heart failure': 'chronic_disease_list___29',
'Hepatitis B': 'chronic_disease_list___15',
'Hepatitis C': 'chronic_disease_list___16',
'HIV / AIDS': 'chronic_disease_list___17',
'High cholesterol': 'chronic_disease_list___18',
'High blood pressure (Hypertension)': 'chronic_disease_list___19',
'Kidney disease': 'chronic_disease_list___20',
'Myotonic Dystrophy type 2 (DM2)': 'chronic_disease_list___21',
'Obesity': 'chronic_disease_list___22',
'Osteoporosis': 'chronic_disease_list___23',
'Sleep disorder (e.g., sleep apnea, insomnia)': 'chronic_disease_list___24',
'Thyroid disease (hyper/hypo)': 'chronic_disease_list___25',
'Other chronic diseases': 'chronic_disease_list___99',



'Arthritis (excluding fibromyalgia)',
'Asthma',
'Bladder disorders (e.g., incontinence, overactive bladder, cystitis)',
'Bowel disease',
'Liver disease or cirrhosis (non-hepatitis)',
'Cancer (excluding non-melanoma skin cancer)',
'Cerebrovascular diseases (e.g., stroke, TIA)',
'Chronic pain (e.g., fibromyalgia, back pain, migraines)',
'COPD, chronic bronchitis, emphysema',
'Dementia (e.g., Alzheimer\'s disease)',
'Diabetes',
'Epilepsy/seizure disorder',
'GERD/reflux/peptic ulcer',
'Arrhythmia',
'Angina',
'Past heart attack',
'Heart failure',
'Hepatitis B',
'Hepatitis C',
'HIV / AIDS',
'High cholesterol',
'High blood pressure (Hypertension)',
'Kidney disease',
'Myotonic Dystrophy type 2 (DM2)',
'Obesity',
'Osteoporosis',
'Sleep disorder (e.g., sleep apnea, insomnia)',
'Thyroid disease (hyper/hypo)',
'Other chronic diseases',
'mental illness'
'''