## Property of Anne-Lise Marais, 2025 {maraisannelise98@gmail.com}

##This code extracts subjects from the ABCD study data 5.1 release. 
#Children are chosen upon three criteria
#1- Has all demographic data
#2- Developmental delay (DD). Determine if a child as a DD using the 90th percentile of our control sample.
#3- mild traumatic brain ijury (mTBI). Determine if a child had a mTBI before enrolment, never after enrolment, and is a unique trauma


##scripted on ipython


import pandas as pd
import scipy
from scipy.stats import chi2_contingency




data = pd.read_excel('/Users/amarais/Documents/abcd/data/final_sample.xlsx', header=0)

data_base = data[data['eventname']== 'baseline_year_1_arm_1']



#N final
N_final = data_base.groupby(['group']).size()
print('N final')
print(N_final)
N_final.to_excel('/Users/amarais/Documents/abcd/result/N_final.xlsx')


##Calculate severities

#N_severity
N_severity = data_base.groupby(['tbi_severity', 'dd_severity']).size()
print('N severity across DD and mTBI condition')
print(N_severity)
N_severity.to_excel('/Users/amarais/Documents/abcd/result/N_severity.xlsx')


#N_severity_bygroup_dd
N_sev_dd = data_base.groupby(['group', 'dd_severity']).size()
total_population = N_sev_dd.sum()
prop_group = N_sev_dd.div(N_sev_dd.groupby(level=0).sum(), level=0)
prop_total = N_sev_dd / total_population
severity_bygroup_dd = (
    pd.concat(
        [N_sev_dd, prop_group, prop_total],
        axis=1
    )
    .set_axis(['Count', 'prop_group', 'prop_total'], axis=1)
)

print('Severity of DD by group')
print(severity_bygroup_dd)
severity_bygroup_dd.to_excel('/Users/amarais/Documents/abcd/result/severity_bygroup_dd.xlsx')


#N_severity_bygroup_mtbi
N_sev_mtbi = data_base.groupby(['group', 'tbi_severity']).size()
total_population = N_sev_mtbi.sum()
prop_group = N_sev_mtbi.div(N_sev_mtbi.groupby(level=0).sum(), level=0)
prop_total = N_sev_mtbi / total_population
severity_bygroup_mtbi = (
    pd.concat(
        [N_sev_mtbi, prop_group, prop_total],
        axis=1
    )
    .set_axis(['Count', 'prop_group', 'prop_total'], axis=1)
)

print('Severity of mTBI by group')
print(severity_bygroup_mtbi)
severity_bygroup_mtbi.to_excel('/Users/amarais/Documents/abcd/result/severity_bygroup_mtbi.xlsx')


#Severity_mTBI
severity_mtbi = data_base.groupby('tbi_group')['tbi_severity'].value_counts().unstack(fill_value=0)
severity_mtbi.columns = [f'N{level}' for level in severity_mtbi.columns]
total_subjects = severity_mtbi.sum(axis=1)
for level in severity_mtbi.columns:
    severity_mtbi[f'P{level}'] = severity_mtbi[level] / total_subjects
column_names = {'N0': 'N Control','N1': 'N mTBI LOC', 'N2': 'N mTBI mem','N3': 'N mTBI LOC+mem', 'PN0': 'P Control', 'PN1': 'P mTBI LOC', 'PN2': 'P mTBI mem','PN3': 'P mTBI LOC+mem',}
severity_mtbi.rename(columns=column_names, inplace=True)

print("Severity of mTBI across group :")
print(severity_mtbi)

severity_mtbi.to_excel('/Users/amarais/Documents/abcd/result/severity_tbi.xlsx', index_label='Group')


##Calculate age




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
ctrl_ages = data['group']==0 ['demo_brthdat_v2']
dd_ages = data['group']==1 ['demo_brthdat_v2']
mtbi_ages = data['group']==2 ['demo_brthdat_v2']
both_ages = data['group']==3 ['demo_brthdat_v2']

# ANOVA
anova_age = f_oneway(ctrl_ages, dd_ages, mtbi_ages, both_ages)

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
final_2_dd = final_sample[(final_sample['group'] == 1) & (final_sample['eventname'] == '2_year_follow_up_y_arm_1')]['interview_age']
final_2_mtbi = final_sample[(final_sample['group'] == 2) & (final_sample['eventname'] == '2_year_follow_up_y_arm_1')]['interview_age']
final_2_both = final_sample[(final_sample['group'] == 3) & (final_sample['eventname'] == '2_year_follow_up_y_arm_1')]['interview_age']


# Filtrer les NaN
final_2_ctrl = final_2_ctrl.dropna()
final_2_dd = final_2_dd.dropna()
final_2_mtbi = final_2_mtbi.dropna()
final_2_both = final_2_both.dropna()


# ANOVA
anova_age2 = f_oneway(final_2_ctrl, final_2_dd, final_2_mtbi, final_2_both)

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
final_4_dd = final_sample[(final_sample['group'] == 1) & (final_sample['eventname'] == '4_year_follow_up_y_arm_1')]['interview_age']
final_4_mtbi = final_sample[(final_sample['group'] == 2) & (final_sample['eventname'] == '4_year_follow_up_y_arm_1')]['interview_age']
final_4_both = final_sample[(final_sample['group'] == 3) & (final_sample['eventname'] == '4_year_follow_up_y_arm_1')]['interview_age']


# Filtrer les NaN
final_4_ctrl = final_4_ctrl.dropna()
final_4_dd = final_4_dd.dropna()
final_4_mtbi = final_4_mtbi.dropna()
final_4_both = final_4_both.dropna()


# ANOVA
anova_age4 = f_oneway(final_4_ctrl, final_4_dd, final_4_mtbi, final_4_both)

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
sex_counts = data.groupby(['tbi_group', 'dd_group'])['demo_sex_v2'].value_counts().unstack(fill_value=0)

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
educ_by_event = data['prnt_educ'].agg(['mean', 'std', 'min', 'max'])



print("Education parentale (moyenne et écart-type) pour chaque groupe à la baseline :")
print(educ_by_event)

# Sauvegarder les résultats dans un fichier Excel
educ_by_event.to_excel('/Users/amarais/Documents/abcd/result/educ_by_event.xlsx', index_label='Group')


# Calculer la moyenne et la standard deviation de 'demo_brthdat_v2' pour chaque groupe
educ_by_group = data.groupby(['group'])['prnt_educ'].agg(['mean', 'std'])



print("Education parentale (moyenne et écart-type) pour chaque groupe à la baseline :")
print(educ_by_group)

# Sauvegarder les résultats dans un fichier Excel
educ_by_group.to_excel('/Users/amarais/Documents/abcd/result/educ_by_group.xlsx', index_label='Group')


#------------------STATS EDUC-PARENT

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Définir les groupes pour l'ANOVA
ctrl_educ = data['group']==0 ['prnt_educ']
dd_educ = data['group']==1 ['prnt_educ']
mtbi_educ = data['group']==2 ['prnt_educ']
both_educ = data['group']==3 ['prnt_educ']


# ANOVA
anova_educ = f_oneway(ctrl_educ, dd_educ, mtbi_educ, both_educ)

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
all_data = pd.concat([ctrl_educ, dd_educ, mtbi_educ, both_educ])
group_labels = (['ctrl'] * len(ctrl_educ) + 
                ['dd'] * len(dd_educ) + 
                ['mtbi'] * len(mtbi_educ) + 
                ['dd_mtbi'] * len(both_educ))

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


demographics_stats = data[['demo_brthdat_v2','demo_sex_v2','prnt_educ']]



#------------------ END STATS CORRELATION DEMOGRAPHICS
