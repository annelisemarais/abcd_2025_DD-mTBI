## Property of Anne-Lise Marais, 2025 {maraisannelise98@gmail.com}

##This code extracts subjects from the ABCD study data 5.1 release. 
#Children are chosen upon three criteria
#1- Has all demographic data
#2- Developmental delay (DD). Determine if a child as a DD using the 90th percentile of our control sample.
#3- mild traumatic brain ijury (mTBI). Determine if a child had a mTBI before enrolment, never after enrolment, and is a unique trauma


##scripted on ipython


import pandas as pd
import numpy as np


###DEMOGRAPHICS
print('processing demographics data')

all_demo = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/abcd_p_demo.xlsx', header=0)


# Filter data for variables 'demo_brthdat_v2', 'demo_sex_v2' et 'demo_prnt_ed_v2'
all_demo_data = (
    all_demo
    .dropna(subset=['demo_brthdat_v2', 'demo_sex_v2', 'demo_prnt_ed_v2'])  # Supprimer les lignes avec des NaN 
    .drop_duplicates(subset=['src_subject_id'], keep='first')  # Garder la première occurrence par sujet
    [['src_subject_id', 'demo_brthdat_v2', 'demo_sex_v2', 'demo_prnt_ed_v2', 'demo_prtnr_ed_v2']]  # Ne garder que les colonnes nécessaires
)

#drop children with unexpected data (intersexe, doesn't know, doesn't want to answer)
demo = all_demo_data[all_demo_data['demo_sex_v2'].isin([1, 2])]
demo = demo.loc[demo['demo_prnt_ed_v2'] != 777]
demo = demo.loc[demo['demo_prtnr_ed_v2'] != 777]
demo = demo.loc[demo['demo_prtnr_ed_v2'] != 999]

#select parent education as the highest between the two partners
demo['prnt_educ'] = demo[['demo_prnt_ed_v2', 'demo_prtnr_ed_v2']].max(axis=1)

demo_id = demo['src_subject_id'].unique()

print('END processing demographics data')

### PHYSICAL HEALTH

print('processing physical health data')

ph = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/ph_p_dhx.xlsx', header=0)


##Extract developmental milestones (dm) questions
#roll_over : devhx_19a_p ; sit : devhx_19b_p ; walk : devhx_19c_p ; speak : devhx_19d_p
dm = ph[ph['eventname'] == 'baseline_year_1_arm_1'][['src_subject_id', 'devhx_19a_p', 'devhx_19b_p', 'devhx_19c_p', 'devhx_19d_p']]

# Drop children that did not complete the questions or miscompleted

dm_var = ['devhx_19a_p', 'devhx_19b_p', 'devhx_19c_p', 'devhx_19d_p']
dm[dm_var] = dm[dm_var].where(dm[dm_var] <= 72, np.nan) #
dm = dm.dropna(subset=dm_var, how='all')
dm_id = dm['src_subject_id'].unique()
demo = demo[demo['src_subject_id'].isin(dm_id)]

demo_dm = demo.merge(dm, on='src_subject_id', how='inner')

print('END processing physical health data')

### TRAUMATIC BRAIN INJURY

print('processing TBI data')

all_otbi = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/ph_p_otbi.xlsx', header=0)

otbi = all_otbi[all_otbi['src_subject_id'].isin(demo_dm['src_subject_id'].unique())]

# drop children that did not answer the questions at baseline
subjects_with_baseline = otbi[otbi['eventname'] == 'baseline_year_1_arm_1']['src_subject_id'].unique()
otbi_baseline = otbi[otbi['src_subject_id'].isin(subjects_with_baseline)]

# drop children with multiple TBI or were unconscious for other reasons
repeated_trauma = otbi_baseline.groupby('src_subject_id').apply(lambda group: (group['tbi_7a'] == 1).any() | (group['tbi_6o'] == 1).any() | (group['tbi_8g'] == 1).any())
valid_subs = repeated_trauma[~repeated_trauma].index  
otbi_norepeat = otbi_baseline[otbi_baseline['src_subject_id'].isin(valid_subs)]

# drop children with a new tbi after baseline
tbi_l_columns = ['tbi_1_l', 'tbi_2_l', 'tbi_3_l', 'tbi_4_l', 'tbi_5_l']
otbi_no_new = otbi_norepeat.groupby('src_subject_id')[tbi_l_columns].sum().sum(axis=1).eq(0)
otbi_filtered = otbi_norepeat[otbi_norepeat['src_subject_id'].isin(otbi_no_new[otbi_no_new].index)]


##Select control children

tbi_columns = ['tbi_1', 'tbi_2', 'tbi_3', 'tbi_4', 'tbi_5']
loc_columns = ['tbi_1b', 'tbi_2b', 'tbi_3b', 'tbi_4b', 'tbi_5b']
memory_columns = ['tbi_1c', 'tbi_2c', 'tbi_3c', 'tbi_4c', 'tbi_5c']
age_columns = ['tbi_1d', 'tbi_2d', 'tbi_3d', 'tbi_4d', 'tbi_5d']

# Identify subject with mTBI
tbi_subjects = otbi_filtered.groupby('src_subject_id').apply(
    lambda group: (
        (group.loc[group['eventname'] == 'baseline_year_1_arm_1', tbi_columns].sum().sum() > 0) &  # At lest one 1 at baseline.
        (group.loc[group['eventname'] != 'baseline_year_1_arm_1', tbi_columns].sum().sum() == 0)   # No other one elsewhere
    )
)

tbi_index = tbi_subjects[tbi_subjects].index  
mtbi_control = otbi_filtered[~otbi_filtered['src_subject_id'].isin(tbi_index)]
control = pd.DataFrame({'src_subject_id': mtbi_control['src_subject_id'].unique(), 'eventname':'baseline_year_1_arm_1', 'tbi_group':0, 'tbi_severity':0})



##Get percentils on mtbi_controls

control_ids = mtbi_control['src_subject_id'].unique()
control_ph = ph[ph['src_subject_id'].isin(control_ids)]

dm_control = control_ph[control_ph['eventname'] == 'baseline_year_1_arm_1'][['src_subject_id', 'devhx_19a_p', 'devhx_19b_p', 'devhx_19c_p', 'devhx_19d_p']]

percentiles = {var: dm_control[var].quantile(0.90) for var in dm_var}


##Filter children with exclusion criterion
tbi = otbi_filtered[otbi_filtered['src_subject_id'].isin(tbi_index)]

# drop children with moderate or severe TBI
high_severity = tbi.loc[(tbi[loc_columns + memory_columns] > 1).any(axis=1), 'src_subject_id'].unique()
tbi = tbi[~tbi['src_subject_id'].isin(high_severity)]

#drop children with a tbi before latest milestones
max_percentile = max(percentiles.values())/12
early_tbi_index = tbi.index[(tbi[age_columns] <= max_percentile).any(axis=1)].tolist()
early_tbi = tbi.loc[early_tbi_index, 'src_subject_id'].tolist()
tbi = tbi[~tbi['src_subject_id'].isin(early_tbi)]

print(f"IDs of dropped subjects : {early_tbi}")
print(f"Number of subs with a TBI before 18 month (90th perc) : {len(early_tbi)}")


tbi.to_excel('/Users/amarais/Documents/abcd/verif/tbi.xlsx')


## Drop children with 2 mTBI before baseline

mtbi_index = tbi['tbi_ss_worst_overall']>1
mtbi = tbi.loc[mtbi_index]

total_tbi = mtbi[tbi_columns].sum(axis=1)
multiple_yes = mtbi[total_tbi > 1]

def sort_multiple_yes(row):
    # Trouver les questions principales auxquelles le sujet a répondu "oui"
    questions_with_yes = [f"tbi_{i}" for i in range(1, 6) if row[f"tbi_{i}"] == 1]

    if len(questions_with_yes) > 1:
        # Récupérer les réponses aux sous-questions b, c et d
        bcd_responses = [tuple(row[f"{q}{s}"] for s in "bcd") for q in questions_with_yes]

        # Vérifier si toutes les réponses sont identiques (=> doublon) ou différentes (=> incohérent)
        if all(resp == bcd_responses[0] for resp in bcd_responses):
            return "duplicate"  # Réponses dupliquées
        else:
            return "inconsistent"  # Réponses incohérentes

    return "valid"  # Aucune anomalie

analysis_results = multiple_yes.apply(sort_multiple_yes, axis=1)

to_drop = analysis_results[analysis_results == "inconsistent"].index
duplicates = analysis_results[analysis_results == "duplicate"].index

mtbi = mtbi.drop(to_drop)

print(f"Subjects dropped due to inconsistent responses: {len(to_drop)}")
print(f"Subjects with duplicated responses: {len(duplicates)}")


mtbi.to_excel('/Users/amarais/Documents/abcd/verif/mtbi.xlsx')

print('END processing TBI data')


##Export useful data
def extract_data(row):
    columns_with_1 = [col for col in tbi_columns if row[col] == 1]
    
    if columns_with_1:
        cause = max(int(col.split('_')[1]) for col in columns_with_1)
    else:
        cause = 'NaN'

    b_col = f"tbi_{cause}b"  # Correction de la syntaxe pour récupérer b
    c_col = f"tbi_{cause}c"  # Correction de la syntaxe pour récupérer c

    if row[b_col] == 0 and row[c_col] == 0:
        severity = 'NaN'
    elif row[b_col] == 1 and row[c_col] == 0:
        severity = 1 #Lost consciousness
    elif row[b_col] == 0 and row[c_col] == 1:
        severity = 2 #Lost memory
    elif row[b_col] == 1 and row[c_col] == 1:
        severity = 3 #Lost conciousness + memory
    else:
        severity = 'NaN'  

    age = row[age_columns].dropna()
    age_value = age.iloc[0] if not age.empty else None

    return cause, severity, age_value  


mtbi_data = pd.DataFrame({
    'src_subject_id': mtbi['src_subject_id'].unique(),
    'eventname': 'baseline_year_1_arm_1',
    'tbi_group': 1,
    'tbi_cause': mtbi.apply(lambda row: extract_data(row)[0], axis=1),
    'tbi_severity': mtbi.apply(lambda row: extract_data(row)[1], axis=1),
    'tbi_age': mtbi.apply(lambda row: extract_data(row)[2] *12, axis=1),
    'tbi_age_y': mtbi.apply(lambda row: extract_data(row)[2], axis=1)
})

mtbi_data.to_excel('/Users/amarais/Documents/abcd/data/mtbi_r.xlsx')




N_mtbi = mtbi_data['src_subject_id'].nunique()
N_control = control['src_subject_id'].nunique()
print("Total mTBI :", N_mtbi)
print("Total control :", N_control)

my_sub_otbi = pd.concat([mtbi_data, control], ignore_index=True)
my_sub_otbi_id = my_sub_otbi['src_subject_id'].unique()
demo_dm_otbi = demo_dm[demo_dm['src_subject_id'].isin(my_sub_otbi_id)]
demo_dm_otbi = demo_dm_otbi.merge(my_sub_otbi, on='src_subject_id', how='left')


demo_dm_otbi.to_excel('/Users/amarais/Documents/abcd/data/all_sub_r.xlsx')

###PHYSICAL HEALTH 2 

##Get developmental milestones
dm_data = demo_dm_otbi.copy()  

for var in dm_var:
    dm_data[f'90th_{var}'] = dm_data[var] > percentiles[var]
group_totals = dm_data.groupby('tbi_group').size()  

results_90th = (
    dm_data
    .groupby('tbi_group')[[f'90th_{var}' for var in dm_var]]
    .agg(['sum', 'mean'])  
    .rename(columns={'sum': 'Count', 'mean': 'Proportion'})
)


for var in dm_var:
    results_90th[(f'90th_{var}', '90th_Percentile_Value')] = percentiles[var]

results_90th.insert(0, 'Total_Subjects', group_totals)  # Insérer le total en première colonne


print("Résultats par groupe pour le 90e percentile :")
print(results_90th)

output_path = "/Users/amarais/Documents/abcd/result/proportion_90th.xlsx"
results_90th.to_excel(output_path, engine='openpyxl') 


##Assign group with binary code: 0 has no dd; 1 has at least on dd
demo_dm_otbi['dd_group'] = (dm_data[[f'90th_{var}' for var in dm_var]].sum(axis=1) > 0).astype(int)
demo_dm_otbi['dd_severity'] = dm_data[[f'90th_{var}' for var in dm_var]].sum(axis=1)

#Create 4 groups depending on tbi group and dd group
def assign_group(row):
    if row['tbi_group'] == 0 and row['dd_group'] == 0:
        return 0, 'control'
    elif row['tbi_group'] == 0 and row['dd_group'] == 1:
        return 1, 'delay'
    elif row['tbi_group'] == 1 and row['dd_group'] == 0:
        return 2, 'mtbi'
    elif row['tbi_group'] == 1 and row['dd_group'] == 1:
        return 3, 'paired'
    else:
        return 'NaN'  

demo_dm_otbi[['group', 'group_nom']] = demo_dm_otbi.apply(
    lambda row: pd.Series(assign_group(row)), axis=1
)


###LONGITUDINAL DATA


## Re-assort rows in a longitudinal setting
new_rows = []

for index, row in demo_dm_otbi.iterrows():

    new_row_1 = row.copy()
    new_row_1['eventname'] = '2_year_follow_up_y_arm_1'
    new_rows.append(new_row_1)
    
    new_row_2 = row.copy()
    new_row_2['eventname'] = '4_year_follow_up_y_arm_1'
    new_rows.append(new_row_2)


new_rows_df = pd.DataFrame(new_rows)
abcd_sample = pd.concat([demo_dm_otbi, new_rows_df], ignore_index=True)
abcd_sample = abcd_sample.sort_values(by='src_subject_id').reset_index(drop=True)

##Add longitudinal data (age at follow-ups)

longitudinal = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/abcd_y_lt.xlsx', header=0)

abcd_sample = abcd_sample.merge(
    longitudinal[['src_subject_id', 'eventname', 'interview_age']], 
    on=['src_subject_id', 'eventname'], 
    how='left'
)


####---------------CBCL---------------

print("Adding CBCL")

cbcl = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/mh_p_cbcl.xlsx', header=0)
abcd_sample = abcd_sample.merge(cbcl[['src_subject_id', 'eventname','cbcl_scr_dsm5_adhd_t','cbcl_scr_dsm5_anxdisord_t','cbcl_scr_dsm5_conduct_t','cbcl_scr_dsm5_depress_t','cbcl_scr_dsm5_opposit_t','cbcl_scr_syn_external_t','cbcl_scr_syn_internal_t']], on=['src_subject_id', 'eventname'], how='left')
print(abcd_sample)

###------------------END CBCL-----------


###------------------Final sample

final_sample = abcd_sample[abcd_sample.groupby('src_subject_id')['src_subject_id'].transform('count') > 1]

final_sample['age0'] = final_sample['interview_age'] - final_sample['interview_age'].min()
final_sample['demo_sex_v2'] = final_sample['demo_sex_v2'] - 1
final_sample['age_from_tbi'] = final_sample['demo_brthdat_v2'] - final_sample['tbi_age']/12

map_event = {
    "baseline_year_1_arm_1": 0,
    "2_year_follow_up_y_arm_1": 1,
    "4_year_follow_up_y_arm_1": 2
}
final_sample['event0'] = final_sample['eventname'].map(map_event)

final_sample.to_excel('/Users/amarais/Documents/abcd/data/final_sample.xlsx')







print("Data are saved")

print("End script abcd-1_participant-selection")