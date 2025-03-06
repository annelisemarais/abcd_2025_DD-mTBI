##This code extracts subjects TBI using the otbi questionnaire in the ABCD cohort

## Property of Anne-Lise Marais, 2025 {maraisannelise98@gmail.com}

##scripted on ipython

##requirements
## pip install pandas


import pandas as pd
import scipy
from scipy.stats import chi2_contingency

#DEMOGRAPHICS

demo = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/abcd_p_demo.xlsx', header=0)

##Sexe, âge de l'enfant, education parent

# Filtrer les données pour 'demo_brthdat_v2', 'demo_sex_v2' et 'demo_prnt_ed_v2'
demo_data = (
    demo
    .dropna(subset=['demo_brthdat_v2', 'demo_sex_v2', 'demo_prnt_ed_v2'])  # Supprimer les lignes avec des NaN 
    .drop_duplicates(subset=['src_subject_id'], keep='first')  # Garder la première occurrence par sujet
    [['src_subject_id', 'demo_brthdat_v2', 'demo_sex_v2', 'demo_prnt_ed_v2', 'demo_prtnr_ed_v2']]  # Ne garder que les colonnes nécessaires
)


all_sub_r_demo = demo_data[demo_data['demo_sex_v2'].isin([1, 2])]
all_sub_r_demo = all_sub_r_demo.loc[all_sub_r_demo['demo_prnt_ed_v2'] != 777]
all_sub_r_demo = all_sub_r_demo.loc[all_sub_r_demo['demo_prtnr_ed_v2'] != 777]
all_sub_r_demo = all_sub_r_demo.loc[all_sub_r_demo['demo_prtnr_ed_v2'] != 999]


all_sub_r_demo['prnt_educ'] = all_sub_r_demo[['demo_prnt_ed_v2', 'demo_prtnr_ed_v2']].max(axis=1)





all_sub_r_demo_id = all_sub_r_demo['src_subject_id'].unique()

## Physical health

ph = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/ph_p_dhx.xlsx', header=0)


# Étape 2 : Filtrer 

#roll_over : devhx_19a_p ; sit : devhx_19b_p ; walk : devhx_19c_p ; speak : devhx_19d_p
dev_mil = ph[ph['eventname'] == 'baseline_year_1_arm_1'][['src_subject_id', 'devhx_19a_p', 'devhx_19b_p', 'devhx_19c_p', 'devhx_19d_p']]


#Calculer les N

# Liste des variables pour lesquelles effectuer les calculs
dev_mil_var = ['devhx_19a_p', 'devhx_19b_p', 'devhx_19c_p', 'devhx_19d_p']

dev_mil = dev_mil.dropna(subset=dev_mil_var, how='all')

dev_mil_id = dev_mil['src_subject_id'].unique()


all_sub_r_demo = all_sub_r_demo[all_sub_r_demo['src_subject_id'].isin(dev_mil_id)]


all_sub_r_demo_dm = all_sub_r_demo.merge(dev_mil, on='src_subject_id', how='inner')





otbi = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/ph_p_otbi.xlsx', header=0)

my_otbi = otbi[otbi['src_subject_id'].isin(all_sub_r_demo_dm['src_subject_id'].unique())]

### Retirer les sujets qui n'ont pas la baseline

# Identifier les sujets ayant une ligne avec 'baseline_year_1_arm_1'
subjects_with_baseline = my_otbi[my_otbi['eventname'] == 'baseline_year_1_arm_1']['src_subject_id'].unique()

# Filtrer pour garder uniquement les sujets ayant une baseline
otbi_baseline = my_otbi[my_otbi['src_subject_id'].isin(subjects_with_baseline)]



### Retirer les enfants qui ont de multiples TBI

# Retirer les sujets qui ont un trauma répété dans l'année ou une perte de conscience liée à un autre event
repeated_trauma = otbi_baseline.groupby('src_subject_id').apply(
    lambda group: (group['tbi_7a'] == 1).any() | (group['tbi_6o'] == 1).any() | (group['tbi_8g'] == 1).any()
)

valid_subs = repeated_trauma[~repeated_trauma].index  

otbi_norepeat = otbi_baseline[otbi_baseline['src_subject_id'].isin(valid_subs)]

# Liste des colonnes 'nouveau tbi depuis la dernière fois ?' (après la baseline)
tbi_l_columns = ['tbi_1_l', 'tbi_2_l', 'tbi_3_l', 'tbi_4_l', 'tbi_5_l']


# Identifier et filtrer les sujets où la somme des questions est strictement égale à 0 (pas de nouveau tbi)
otbi_newtbi = otbi_norepeat.groupby('src_subject_id')[tbi_l_columns].sum().sum(axis=1)

nonewtbi_index = otbi_newtbi[otbi_newtbi == 0].index

otbi_filtered = otbi_norepeat[otbi_norepeat['src_subject_id'].isin(nonewtbi_index)]




###Filtrer les enfants qui n'ont jamais eu de TBI et n'en auront jamais (control)


# Liste des colonnes "votre enfant s'est-il cogné la tête avant 9 ans" (baseline)
tbi_columns = ['tbi_1', 'tbi_2', 'tbi_3', 'tbi_4', 'tbi_5']
loc_columns = ['tbi_1b', 'tbi_2b', 'tbi_3b', 'tbi_4b', 'tbi_5b']
memory_columns = ['tbi_1c', 'tbi_2c', 'tbi_3c', 'tbi_4c', 'tbi_5c']
age_columns = ['tbi_1d', 'tbi_2d', 'tbi_3d', 'tbi_4d', 'tbi_5d']

# Identifier les sujets qui ont un TBI
tbi_subjects = otbi_filtered.groupby('src_subject_id').apply(
    lambda group: (
        (group.loc[group['eventname'] == 'baseline_year_1_arm_1', tbi_columns].sum().sum() > 0) &  # Il y a au moins un 1 dans baseline...
        (group.loc[group['eventname'] != 'baseline_year_1_arm_1', tbi_columns].sum().sum() == 0)   # Et aucun 1 ailleurs
    )
)

tbi_index = tbi_subjects[tbi_subjects].index  


# Créer un tableau ne contenant que les sujets qui n'ont jamais eu de TBI avant 9 ans
control_subjects = otbi_filtered[~otbi_filtered['src_subject_id'].isin(tbi_index)]


#Vérifier qu'il n'y a pas d'erreur
control_sums = control_subjects.groupby('src_subject_id')[tbi_columns + tbi_l_columns].sum().sum(axis=1)

control_error = control_sums[control_sums != 0]

if not control_error.empty:
    print("Erreur : Certains sujets 'control' ont une valeur de 1 ou plus signifiant un TBI.")
    print("Sujets concernés et leurs sommes :")
    print(control_error)
    print("fichier 'control' non mis à jour")
    #ajouter if tbiss overall == 1 & tbiss overall l .mean() == 1
else:
    print("Pas de TBI chez les 'control', mise à jour du fichier.")
    control = pd.DataFrame({'src_subject_id': control_subjects['src_subject_id'].unique(), 'eventname':'baseline_year_1_arm_1', 'tbi_group':0, 'tbi_severity':0})
    control_subjects.to_excel('/Users/amarais/Documents/abcd/verif/control.xlsx')
    control.to_excel('/Users/amarais/Documents/abcd/data/control_r.xlsx')


#Calculer les percentiles sur les control

control_ids = control['src_subject_id'].unique()


## Physical health


# Étape 2 : Filtrer 
control_ph = ph[ph['src_subject_id'].isin(control_ids)]

#roll_over : devhx_19a_p ; sit : devhx_19b_p ; walk : devhx_19c_p ; speak : devhx_19d_p
dev_mil_control = control_ph[control_ph['eventname'] == 'baseline_year_1_arm_1'][['src_subject_id', 'devhx_19a_p', 'devhx_19b_p', 'devhx_19c_p', 'devhx_19d_p']]

#Calculer les N

# Liste des variables pour lesquelles effectuer les calculs
dev_mil_var = ['devhx_19a_p', 'devhx_19b_p', 'devhx_19c_p', 'devhx_19d_p']

# Calcul des 90ᵉ percentiles pour chaque variable
percentiles = {var: dev_mil_control[var].quantile(0.90) for var in dev_mil_var}






###Filtrer les enfants qui ont déjà eu un TBI et qui n'en auront plus


# Créer le nouveau tableau ne contenant que les sujets avec TBI avant 9 ans
previous_injury = otbi_filtered[otbi_filtered['src_subject_id'].isin(tbi_index)]

# Supprimer les sujets qui ont un TBI moderate et severe
high_severity_prev = previous_injury.loc[(previous_injury[loc_columns + memory_columns] > 1).any(axis=1), 'src_subject_id'].unique()
previous_injury = previous_injury[~previous_injury['src_subject_id'].isin(high_severity_prev)]

#Supprimer les sujets qui ont eu un tbi avant les developmental milestone

max_percentile = max(percentiles.values())/12

age_columns = ['tbi_1d', 'tbi_2d', 'tbi_3d', 'tbi_4d', 'tbi_5d']

early_tbi_index = previous_injury.index[(previous_injury[age_columns] <= max_percentile).any(axis=1)].tolist()
early_tbi = previous_injury.loc[early_tbi_index, 'src_subject_id'].tolist()

previous_injury = previous_injury[~previous_injury['src_subject_id'].isin(early_tbi)]



# Affichage des résultats
print(f"Nombre de sujets ayant un TBI avant 18 mois (90 perc) : {len(early_tbi)}")
print(f"Noms (ou IDs) des sujets retirés : {early_tbi}")


print("mise à jour de 'previous_injury'")
previous_injury.to_excel('/Users/amarais/Documents/abcd/verif/previous_injury.xlsx')




all_mtbi_index = previous_injury['tbi_ss_worst_overall']>1

all_mtbi = previous_injury[all_mtbi_index]


# Supprimer les sujets avec deux mTBI avant 9 ans
to_delete = []

# Itérer sur chaque ligne pour vérifier si le total est supérieur à 1
for index, row in all_mtbi.iterrows():
    # Vérifier si la somme des colonnes tbi_columns est supérieure à 1
    if row[tbi_columns].sum() > 1:
        # Récupérer les colonnes où il y a un chiffre (valeurs > 0)
        columns_with_values = [col for col in tbi_columns if row[col] > 0]

        # Générer les colonnes b, c, d correspondantes
        colb = [col + 'b' for col in columns_with_values]
        colc = [col + 'c' for col in columns_with_values]
        cold = [col + 'd' for col in columns_with_values]

        # On va vérifier les paires de colonnes
        for i in range(len(columns_with_values)):
            # Vérifier s'il existe au moins une valeur 1 dans colb[i] ou colc[i]
            if row[colb[i]] == 1 or row[colc[i]] == 1:
                # On vérifie les colonnes suivantes si elles existent
                if i < len(columns_with_values) - 1:  # S'assurer que i + 1 est valide
                    # Vérifier au moins une valeur 1 dans les colonnes suivantes
                    if row[colb[i + 1]] == 1 or row[colc[i + 1]] == 1:
                        # Vérifier si les colonnes d sont différentes
                        if (row[cold[i]] != row[cold[i + 1]] or
                        row[colc[i]] != row[colc[i + 1]] or
                        row[colb[i]] != row[colb[i + 1]]):
                            to_delete.append(row['src_subject_id'])
                            break  # Sortir de la boucle si le sujet est marqué pour suppression

# Supprimer les sujets du DataFrame
all_mtbi = all_mtbi[~all_mtbi['src_subject_id'].isin(to_delete)]

# Afficher les résultats
print(f"Sujets à supprimer car au moins deux mTBI : {to_delete}")
print(f"Nombre total de sujets supprimés : {len(to_delete)}")










# Étape 1 : Calculer le total des réponses
total_tbi = all_mtbi[tbi_columns].sum(axis=1)

# Étape 2 : Identifier les sujets avec des réponses multiples
multiple_yes = all_mtbi[total_tbi > 1]

print(f"Nb ayant répondu plusieurs fois oui : {len(multiple_yes)}")

# Étape 3 : Fonction pour vérifier l'égalité des réponses aux sous-questions
def check_duplicates(row):
    # Obtenir les indices des questions principales où le sujet a répondu `1`
    questions_with_yes = [i for i in range(1, 6) if row[f"tbi_{i}"] == 1]
    
    # Extraire les réponses correspondantes pour les sous-questions b, c, et d
    bcd_responses = [
        (row[f"tbi_{i}b"], row[f"tbi_{i}c"], row[f"tbi_{i}d"])
        for i in questions_with_yes
    ]
    
    # Vérifier si toutes les réponses aux sous-questions sont identiques
    return all(resp == bcd_responses[0] for resp in bcd_responses)

# Étape 4 : Filtrer les sujets avec des réponses cohérentes
duplicates = multiple_yes[multiple_yes.apply(check_duplicates, axis=1)]

print(f"Nombre de sujets avec des réponses dupliquées : {len(duplicates)}")

print("Retirer les sujet qui ont 2 mTBI avant 9 ans")
# Étape 5 : Retirer les lignes de multiple_yes de mtbi
mtbi_cleaned = all_mtbi[~all_mtbi.index.isin(multiple_yes.index)]

# Étape 6 : Réinjecter les duplicates dans mtbi_cleaned
mtbi = pd.concat([mtbi_cleaned, duplicates]).sort_index()



print("mise à jour de mtbi")
mtbi.to_excel('/Users/amarais/Documents/abcd/verif/mtbi.xlsx')



def extract_data(row):
    ## CAUSE
    # Trouver les colonnes où il y a un 1
    columns_with_1 = [col for col in tbi_columns if row[col] == 1]
    
    # Extraire le suffixe des colonnes et récupérer le max (si applicable)
    if columns_with_1:
        cause = max(int(col.split('_')[1]) for col in columns_with_1)
    else:
        cause = 'NaN'

    ## SEVERITY
    # Construire les noms de colonnes pour b et c
    b_col = f"tbi_{cause}b"  # Correction de la syntaxe pour récupérer b
    c_col = f"tbi_{cause}c"  # Correction de la syntaxe pour récupérer c

    # Déterminer la sévérité 
    if row[b_col] == 0 and row[c_col] == 0:
        severity = 'NaN'
    elif row[b_col] == 1 and row[c_col] == 0:
        severity = 1 #Lost consciousness
    elif row[b_col] == 0 and row[c_col] == 1:
        severity = 2 #Lost memory
    elif row[b_col] == 1 and row[c_col] == 1:
        severity = 3 #Lost conciousness + memory
    else:
        severity = 'NaN'  # Cas théoriquement impossible

    ## AGE
    age = row[age_columns].dropna()
    age_value = age.iloc[0] if not age.empty else None

    return cause, severity, age_value  # Retourner les valeurs extraites



mtbi_r = pd.DataFrame({
    'src_subject_id': mtbi['src_subject_id'].unique(),
    'eventname': 'baseline_year_1_arm_1',
    'tbi_group': 1,
    'tbi_cause': mtbi.apply(lambda row: extract_data(row)[0], axis=1),
    'tbi_severity': mtbi.apply(lambda row: extract_data(row)[1], axis=1),
    'tbi_age': mtbi.apply(lambda row: extract_data(row)[2] *12, axis=1)
})

mtbi_r.to_excel('/Users/amarais/Documents/abcd/data/mtbi_r.xlsx')




N_mtbi = mtbi_r['src_subject_id'].nunique()
#244
N_control = control['src_subject_id'].nunique()
#7878

print("Total mTBI :", N_mtbi)
print("Total control :", N_control)

all_sub = pd.concat([mtbi, control_subjects], ignore_index=True)
all_sub.to_excel('/Users/amarais/Documents/abcd/verif/all_sub.xlsx')

all_sub_r = pd.concat([mtbi_r, control], ignore_index=True)


all_sub_r_id = all_sub_r['src_subject_id'].unique()


all_sub_r_demo_dm = all_sub_r_demo_dm[all_sub_r_demo_dm['src_subject_id'].isin(all_sub_r_id)]


all_sub_r_demo_dm_otbi = all_sub_r_demo_dm.merge(all_sub_r, on='src_subject_id', how='left')




all_sub_r_demo_dm_otbi.to_excel('/Users/amarais/Documents/abcd/data/all_sub_r.xlsx')





























dev_mil_data = all_sub_r_demo_dm_otbi.copy()  # Copier uniquement les colonnes concernées


for var in dev_mil_var:
    threshold = percentiles[var]
    dev_mil_data[f'90th_{var}'] = dev_mil_data[var] > threshold  # Création des nouvelles colonnes



# Étape 2 : Calculer le nombre et la proportion par groupe pour chaque variable
results_90th = {}

for var in dev_mil_var:
    column = f'90th_{var}'
    
    
    group_totals = dev_mil_data.groupby('tbi_group').size()  # Nombre total de sujets par groupe
    group_above_threshold = dev_mil_data.groupby('tbi_group')[column].sum()  # Nombre de sujets au-dessus du 90e percentile
    group_proportions = group_above_threshold / group_totals  # Proportion de sujets au-dessus du 90e percentile
    
    # Stocker les résultats dans un dictionnaire
    results_90th[var] = pd.DataFrame({
        'Total_Subjects': group_totals,
        f'{column}_Count': group_above_threshold,
        f'{column}_Proportion': group_proportions,
        '90th_Percentile_Value': percentiles[var]  # Ajouter la valeur du 90e percentile
    })

# Étape 3 : Afficher les résultats
print("Résultats par groupe pour le 90e percentile :")
for var, df in results_90th.items():
    print(f"\nRésultats pour {var} (90e percentile):")
    print(df)

# Enregistrer les résultats dans un fichier Excel avec plusieurs onglets
with pd.ExcelWriter('/Users/amarais/Documents/abcd/result/proportion_90th.xlsx') as writer:
    for var, df in results_90th.items():
        df.to_excel(writer, sheet_name=var, index=False)  # Un onglet par variable

all_sub_r_demo_dm_otbi['dm_group'] = (dev_mil_data[[f'90th_{var}' for var in dev_mil_var]].sum(axis=1) > 0).astype(int)
all_sub_r_demo_dm_otbi['dm_severity'] = dev_mil_data[[f'90th_{var}' for var in dev_mil_var]].sum(axis=1)

#Créer le groupe final
def assign_group(row):
    if row['tbi_group'] == 0 and row['dm_group'] == 0:
        return 0, 'control'
    elif row['tbi_group'] == 0 and row['dm_group'] == 1:
        return 1, 'delay'
    elif row['tbi_group'] == 1 and row['dm_group'] == 0:
        return 2, 'mtbi'
    elif row['tbi_group'] == 1 and row['dm_group'] == 1:
        return 3, 'paired'
    else:
        return 'NaN'  # Cas théoriquement impossible

# Application de la fonction sur chaque ligne
# Appliquer la fonction et extraire les deux valeurs
all_sub_r_demo_dm_otbi[['group', 'group_nom']] = all_sub_r_demo_dm_otbi.apply(
    lambda row: pd.Series(assign_group(row)), axis=1
)







# Créer un DataFrame vide pour les nouvelles lignes



new_rows = []

# Parcourir chaque ligne du DataFrame d'origine
for index, row in all_sub_r_demo_dm_otbi.iterrows():
    # Créer deux nouvelles lignes avec eventname modifié
    new_row_1 = row.copy()
    new_row_1['eventname'] = '2_year_follow_up_y_arm_1'
    new_rows.append(new_row_1)
    
    new_row_2 = row.copy()
    new_row_2['eventname'] = '4_year_follow_up_y_arm_1'
    new_rows.append(new_row_2)

# Créer un DataFrame à partir des nouvelles lignes
new_rows_df = pd.DataFrame(new_rows)

# Concaténer les nouveaux enregistrements au DataFrame d'origine
final_sample = pd.concat([all_sub_r_demo_dm_otbi, new_rows_df], ignore_index=True)

# Trier le DataFrame par src_subject_id
final_sample = final_sample.sort_values(by='src_subject_id').reset_index(drop=True)



longitudinal = pd.read_excel('/Users/amarais/Documents/abcd/abcd_data/abcd_y_lt.xlsx', header=0)

# Fusionner les deux DataFrames en fonction de 'src_subject_id' et 'eventname'
final_sample = final_sample.merge(
    longitudinal[['src_subject_id', 'eventname', 'interview_age']], 
    on=['src_subject_id', 'eventname'], 
    how='left'  # 'left' pour garder uniquement les sujets de final_sample
)




# Afficher le DataFrame trié
print(final_sample)



final_sample.to_excel('/Users/amarais/Documents/abcd/data/final_sample.xlsx', index_label='Group')













# Compter le nombre de sujets par combinaison de tbi_group et dm_group
N_final = all_sub_r_demo_dm_otbi.groupby(['group']).size()

# Afficher le tableau
print(N_final)

N_severity = all_sub_r_demo_dm_otbi.groupby(['tbi_severity', 'dm_severity']).size()


# Afficher le tableau
print(N_severity)

N_final.to_excel('/Users/amarais/Documents/abcd/result/N_final.xlsx')
N_severity.to_excel('/Users/amarais/Documents/abcd/result/N_severity.xlsx')








#Severité à la baseline

# Compter le nombre de chaque niveau de 'severity' par groupe
severity_counts = all_sub_r_demo_dm_otbi.groupby('tbi_group')['tbi_severity'].value_counts().unstack(fill_value=0)

# Vérifier le nombre de colonnes créées par unstack()
print("Colonnes dans severity_counts avant renommage :", severity_counts.columns)

# Adapter le renommage des colonnes en fonction du nombre de niveaux de severity
severity_counts.columns = [f'N{level}' for level in severity_counts.columns]

# Calculer le nombre total de sujets par groupe
total_subjects = severity_counts.sum(axis=1)

# Calculer la proportion de chaque niveau de 'severity' par groupe
for level in severity_counts.columns:
    severity_counts[f'P{level}'] = severity_counts[level] / total_subjects

column_names = {'N0': 'N Control','N1': 'N mTBI LOC', 'N2': 'N mTBI mem','N3': 'N mTBI LOC+mem', 'PN0': 'P Control', 'PN1': 'P mTBI LOC', 'PN2': 'P mTBI mem','PN3': 'P mTBI LOC+mem',}

# Renommer les colonnes
severity_counts.rename(columns=column_names, inplace=True)

# Afficher les résultats
print("Nombre et proportion de 'severity' par groupe :")
print(severity_counts)

severity_counts.to_excel('/Users/amarais/Documents/abcd/result/severity_tbi.xlsx', index_label='Group')







# Calculer la moyenne et la standard deviation de 'demo_brthdat_v2' pour chaque groupe
age_by_event = final_sample.groupby(['eventname'])['interview_age'].agg(['mean', 'std','min', 'max'])

age_by_group = final_sample.groupby(['group','eventname'])['interview_age'].agg(['mean', 'std','min', 'max'])


print("Age (M, SE, min, max) :")
print(age_by_event)
print("Age (M, SE, min, max) pour chaque groupe :")
print(age_by_group)

# Sauvegarder les résultats dans un fichier Excel
age_by_event.to_excel('/Users/amarais/Documents/abcd/result/age_by_event.xlsx', index_label='Group')

age_by_group.to_excel('/Users/amarais/Documents/abcd/result/age_by_group.xlsx', index_label='Group')


#------------------STATS AGE

#------------BASELINE

from scipy.stats import f_oneway

# Extraire les âges par groupe
ctrl_ages = all_sub_r_demo_dm_otbi['group']==0 ['demo_brthdat_v2']
dm_ages = all_sub_r_demo_dm_otbi['group']==1 ['demo_brthdat_v2']
mtbi_ages = all_sub_r_demo_dm_otbi['group']==2 ['demo_brthdat_v2']
both_ages = all_sub_r_demo_dm_otbi['group']==3 ['demo_brthdat_v2']

# ANOVA
anova_age = f_oneway(ctrl_ages, dm_ages, mtbi_ages, both_ages)

# Créer un résumé des résultats
anova_summary = (
    "Résultats de l'ANOVA :\n"
    f"F-statistic : {anova_age.statistic}\n"
    f"P-value : {anova_age.pvalue}\n"
)

# Interprétation
if anova_age.pvalue < 0.05:
    anova_summary += "Les moyennes d'âge sont significativement différentes entre les groupes (p < 0.05).\n"
else:
    anova_summary += "Les moyennes d'âge ne sont pas significativement différentes entre les groupes (p >= 0.05).\n"

# Exporter les résultats dans un fichier texte
output_path = "/Users/amarais/Documents/abcd/result/stats_age.txt"
with open(output_path, "w") as f:
    f.write(anova_summary)

print(anova_summary)

#------------YEAR 2





# Extraire les âges par groupe
final_2_ctrl = final_sample[(final_sample['group'] == 0) & (final_sample['eventname'] == '2_year_follow_up_y_arm_1')]['interview_age']
final_2_dm = final_sample[(final_sample['group'] == 1) & (final_sample['eventname'] == '2_year_follow_up_y_arm_1')]['interview_age']
final_2_mtbi = final_sample[(final_sample['group'] == 2) & (final_sample['eventname'] == '2_year_follow_up_y_arm_1')]['interview_age']
final_2_both = final_sample[(final_sample['group'] == 3) & (final_sample['eventname'] == '2_year_follow_up_y_arm_1')]['interview_age']


# Filtrer les NaN
final_2_ctrl = final_2_ctrl.dropna()
final_2_dm = final_2_dm.dropna()
final_2_mtbi = final_2_mtbi.dropna()
final_2_both = final_2_both.dropna()


# ANOVA
anova_age2 = f_oneway(final_2_ctrl, final_2_dm, final_2_mtbi, final_2_both)

# Créer un résumé des résultats
anova_summary2 = (
    "Résultats de l'ANOVA :\n"
    f"F-statistic : {anova_age2.statistic}\n"
    f"P-value : {anova_age2.pvalue}\n"
)

# Interprétation
if anova_age2.pvalue < 0.05:
    anova_summary2 += "Les moyennes d'âge sont significativement différentes entre les groupes (p < 0.05).\n"
else:
    anova_summary2 += "Les moyennes d'âge ne sont pas significativement différentes entre les groupes (p >= 0.05).\n"

# Exporter les résultats dans un fichier texte
output_path = "/Users/amarais/Documents/abcd/result/stats_age2.txt"
with open(output_path, "w") as f:
    f.write(anova_summary2)

print(anova_summary2)


#------------YEAR 4


# Extraire les âges par groupe
final_4_ctrl = final_sample[(final_sample['group'] == 0) & (final_sample['eventname'] == '4_year_follow_up_y_arm_1')]['interview_age']
final_4_dm = final_sample[(final_sample['group'] == 1) & (final_sample['eventname'] == '4_year_follow_up_y_arm_1')]['interview_age']
final_4_mtbi = final_sample[(final_sample['group'] == 2) & (final_sample['eventname'] == '4_year_follow_up_y_arm_1')]['interview_age']
final_4_both = final_sample[(final_sample['group'] == 3) & (final_sample['eventname'] == '4_year_follow_up_y_arm_1')]['interview_age']


# Filtrer les NaN
final_4_ctrl = final_4_ctrl.dropna()
final_4_dm = final_4_dm.dropna()
final_4_mtbi = final_4_mtbi.dropna()
final_4_both = final_4_both.dropna()


# ANOVA
anova_age4 = f_oneway(final_4_ctrl, final_4_dm, final_4_mtbi, final_4_both)

# Créer un résumé des résultats
anova_summary4 = (
    "Résultats de l'ANOVA :\n"
    f"F-statistic : {anova_age4.statistic}\n"
    f"P-value : {anova_age4.pvalue}\n"
)

# Interprétation
if anova_age4.pvalue < 0.05:
    anova_summary4 += "Les moyennes d'âge sont significativement différentes entre les groupes (p < 0.05).\n"
else:
    anova_summary4 += "Les moyennes d'âge ne sont pas significativement différentes entre les groupes (p >= 0.05).\n"

# Exporter les résultats dans un fichier texte
output_path = "/Users/amarais/Documents/abcd/result/stats_age4.txt"
with open(output_path, "w") as f:
    f.write(anova_summary4)

print(anova_summary4)



#------------------END STATS AGEv



# Compter le nombre de sujets par groupe et sexe
sex_counts = all_sub_r_demo_dm_otbi.groupby(['tbi_group', 'dm_group'])['demo_sex_v2'].value_counts().unstack(fill_value=0)

# Renommer les colonnes pour plus de clarté
sex_counts.columns = ['NMale', 'NFemale']

# Calculer la proportion de chaque sexe par groupe
sex_counts['PMale'] = sex_counts['NMale'] / (sex_counts['NMale'] + sex_counts['NFemale'])
sex_counts['PFemale'] = sex_counts['NFemale'] / (sex_counts['NMale'] + sex_counts['NFemale'])


# Afficher les résultats
print("Nombre et proportion de 'sexe' par groupe :")
print(sex_counts)
sex_counts.to_excel('/Users/amarais/Documents/abcd/result/sexe_by_group.xlsx', index_label='Group')


#------------------STATS SEX

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, norm
from statsmodels.stats.multitest import multipletests

# Test du Chi²
chi2, p, dof, expected = chi2_contingency(sex_counts[['NMale', 'NFemale']])

# Afficher les résultats du test du Chi²
chi_square_results = f"""
Test du Chi-carré:
Chi2: {chi2:.4f}, p-value: {p:.4f}, degrés de liberté: {dof}
Valeurs attendues:\n{expected}
"""

if p < 0.05:
    chi_square_results += "Il y a une association significative entre le sexe et le groupe (p < 0.05).\n\n"
else:
    chi_square_results += "Aucune association significative entre le sexe et le groupe (p >= 0.05).\n\n"

# === Post-hoc : Comparaisons multiples avec correction de Bonferroni ===
# Calcul des résidus ajustés de Pearson
observed = sex_counts[['NMale', 'NFemale']].values  # Valeurs observées
residuals = (observed - expected) / np.sqrt(expected)  # Résidus ajustés de Pearson

# Calcul des p-values à partir des résidus ajustés (test bilatéral basé sur la loi normale)
p_values = 2 * (1 - norm.cdf(np.abs(residuals)))

# Correction de Bonferroni
flat_pvals = p_values.flatten()
_, pvals_corrected, _, _ = multipletests(flat_pvals, method='bonferroni')

# Restructurer les p-values corrigées
pvals_corrected = pvals_corrected.reshape(p_values.shape)

# Ajouter les résultats au fichier
chi_square_results += "Résidus ajustés de Pearson :\n" + str(residuals) + "\n\n"
chi_square_results += "P-values corrigées (Bonferroni) :\n" + str(pvals_corrected) + "\n"

# Sauvegarde dans un fichier texte
output_path = "/Users/amarais/Documents/abcd/result/stats_sexe.txt"
with open(output_path, "w") as f:
    f.write(chi_square_results)

# Affichage des résultats
print(chi_square_results)

#------------------END STATS SEX

#education des parents à la baseline
#education des parents à la baseline
# Calculer la moyenne et la standard deviation de 'demo_brthdat_v2' pour chaque groupe
educ_by_event = all_sub_r_demo_dm_otbi['prnt_educ'].agg(['mean', 'std', 'min', 'max'])



print("Education parentale (moyenne et écart-type) pour chaque groupe à la baseline :")
print(educ_by_event)

# Sauvegarder les résultats dans un fichier Excel
educ_by_event.to_excel('/Users/amarais/Documents/abcd/result/educ_by_event.xlsx', index_label='Group')


# Calculer la moyenne et la standard deviation de 'demo_brthdat_v2' pour chaque groupe
educ_by_group = all_sub_r_demo_dm_otbi.groupby(['group'])['prnt_educ'].agg(['mean', 'std'])



print("Education parentale (moyenne et écart-type) pour chaque groupe à la baseline :")
print(educ_by_group)

# Sauvegarder les résultats dans un fichier Excel
educ_by_group.to_excel('/Users/amarais/Documents/abcd/result/educ_by_group.xlsx', index_label='Group')


#------------------STATS EDUC-PARENT

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Définir les groupes pour l'ANOVA
ctrl_educ = all_sub_r_demo_dm_otbi['group']==0 ['prnt_educ']
dm_educ = all_sub_r_demo_dm_otbi['group']==1 ['prnt_educ']
mtbi_educ = all_sub_r_demo_dm_otbi['group']==2 ['prnt_educ']
both_educ = all_sub_r_demo_dm_otbi['group']==3 ['prnt_educ']


# ANOVA
anova_educ = f_oneway(ctrl_educ, dm_educ, mtbi_educ, both_educ)

# Résumé des résultats de l'ANOVA
anova_summary = (
    "Résultats de l'ANOVA :\n"
    f"F-statistic : {anova_educ.statistic:.4f}\n"
    f"P-value : {anova_educ.pvalue:.4f}\n"
)

# Interprétation de l'ANOVA
if anova_educ.pvalue < 0.05:
    anova_summary += "Les moyennes d'éducation parentale sont significativement différentes entre les groupes (p < 0.05).\n\n"
else:
    anova_summary += "Les moyennes d'éducation parentale ne sont pas significativement différentes entre les groupes (p >= 0.05).\n\n"

# POST-HOC : Tukey HSD
# Création des données pour Tukey HSD
all_data = pd.concat([ctrl_educ, dm_educ, mtbi_educ, both_educ])
group_labels = (['ctrl'] * len(ctrl_educ) + 
                ['dm'] * len(dm_educ) + 
                ['mtbi'] * len(mtbi_educ) + 
                ['dm_mtbi'] * len(both_educ))

# Application du test post-hoc
tukey_result = pairwise_tukeyhsd(endog=all_data, groups=group_labels, alpha=0.05)

# Ajout des résultats de Tukey au fichier
anova_summary += "Résultats du test post-hoc Tukey HSD :\n"
anova_summary += str(tukey_result) + "\n"

# Exporter les résultats dans un fichier texte
output_path = "/Users/amarais/Documents/abcd/result/stats_educ.txt"
with open(output_path, "w") as f:
    f.write(anova_summary)

# Afficher les résultats
print(anova_summary)

#------------------END STATS EDUC-PARENT


#------------------ STATS CORRELATION DEMOGRAPHICS


demographics_stats = all_sub_r_demo_dm_otbi[['demo_brthdat_v2','demo_sex_v2','prnt_educ']]



#------------------ END STATS CORRELATION DEMOGRAPHICS
