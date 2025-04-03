#Copyright (C) 2025 Anne-Lise Marais

#This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; version 3.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#If you have any question, please contact Fanny Dégeilh at fanny.degeilh@inserm.fr





## Property of Anne-Lise Marais, 2025 {maraisannelise98@gmail.com}

##This code is for descriptive, corraltion and exploratory analysis of the sample created from the first file

##scripted on ipython


import pandas as pd
import scipy
from scipy.stats import chi2_contingency
import scipy.stats as stats
from scipy.stats import ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols



###Load data
data = pd.read_excel('my-path/data/final_sample.xlsx', header=0)

###Definition of arguments used in the analysis

data_base = data[data['eventname']== 'baseline_year_1_arm_1']
data_base["dual_group"] = data_base["group"].apply(lambda x: "Group 3" if x == 3 else "Group 0-1-2")

cut_offs = {'devhx_19a_p': 5, 'devhx_19b_p': 9, 'devhx_19c_p': 18, 'devhx_19d_p': 13}

for var, cut_off in cut_offs.items():
    data_base[f'diff_{var}'] = (data_base[var] - cut_off).clip(lower=0) 

data_base["diff_tot_dd"] = data_base[['diff_devhx_19a_p', 'diff_devhx_19b_p', 'diff_devhx_19c_p', 'diff_devhx_19d_p']].sum(axis=1)

dd_data = data_base[data_base['group'].isin([1, 3])]
tbi_data = data_base[data_base['group'].isin([2, 3])]

cbcl_var = [
    "cbcl_scr_dsm5_adhd_t", "cbcl_scr_dsm5_anxdisord_t",
    "cbcl_scr_dsm5_conduct_t", "cbcl_scr_dsm5_depress_t",
    "cbcl_scr_dsm5_opposit_t", "cbcl_scr_syn_external_t",
    "cbcl_scr_syn_internal_t"
]




#### DESCRIPTIVE RESULTS


N_final = data_base.groupby(['group']).size()
print('N final')
print(N_final)
N_final.to_excel('my-path/result/N_final.xlsx')

descr = data_base.groupby("group")[cbcl_var].agg(['mean', 'std', 'min', 'max'])
descr.to_excel("my-path/result/stats_cbcl_bygroup.xlsx")
print(descr)


##Proportion of severities/symptoms across groups

#Proportion of DD severity
month_over_bygroup = dd_data.groupby('group')[[f'diff_{var}' for var in cut_offs]].agg(['mean', 'std', 'min', 'max'])
print(month_over_bygroup)
month_over_bygroup.to_excel('my-path/result/month_over_bygroup.xlsx', index_label='Group')

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
severity_bygroup_dd.to_excel('my-path/result/prop_severity_dd.xlsx')


#Proportion of mTBI severity
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
severity_bygroup_mtbi.to_excel('my-path/result/prop_severity_mtbi.xlsx')



#### CORRELATION ANALYSIS

demographics_stats = data_base[['demo_brthdat_v2','demo_sex_v2','prnt_educ', 'dd_group', 'tbi_group']]
variables = demographics_stats.columns  

corr_results = []

for i in range(len(variables)):
    for j in range(i + 1, len(variables)):  
        var1, var2 = variables[i], variables[j]
        
        corr, p_value = stats.pearsonr(demographics_stats[var1], demographics_stats[var2])
        
        corr_results.append([var1, var2, corr, p_value, p_value < 0.05])

corr_df = pd.DataFrame(corr_results, columns=['Variable 1', 'Variable 2', 'Corrélation', 'p-value', 'Significatif ?'])

print(corr_df)

corr_df.to_excel('my-path/result/correlation.xlsx')







#### EXPLORATORY ANALYSIS FOR DD SEVERITY

###Choose severity, compare severity distribution between groups

##Severity #1 cumulative delay

dd_data["dd_cumul_bi"] = (dd_data["dd_severity"] >= 2).astype(int)

contingency_table = pd.crosstab(dd_data["group"], dd_data["dd_cumul_bi"])
print(contingency_table)


chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Chi² : {chi2_stat:.3f}")
print(f"P-value : {p_val:.3f}")

#No difference between children with DD and children with DD and mTBI



##Severity #2 delay severity (total number of months over the cut-offs)

group_1 = dd_data[dd_data['group'] == 1]['diff_tot_dd']
group_2 = dd_data[dd_data['group'] == 3]['diff_tot_dd']

statistic, p_value = ttest_ind(group_1, group_2, equal_var=False)  

print("T-test, Welch:")
print(f"t-value: {statistic}")
print(f"P-value: {p_value}")

#No difference between children with DD and children with DD and mTBI

##children with DD and children with DD and mTBI are homogeneous in terms of DD severity




### Regression analysis
## x1 = severity 2 (months), x2 = group, y = cbcl

reg_results_dd = {}

for var in cbcl_var:
    data_clean = dd_data[[var, "diff_tot_dd", "group"]].dropna()
    
    X = sm.add_constant(data_clean[["diff_tot_dd", "group"]])
    y = data_clean[var]
    
    model = sm.OLS(y, X).fit()
    
    reg_results_dd[var] = {
        "Intercept": model.params["const"],
        "Slope (severity)": model.params["diff_tot_dd"],
        "p-value (sevrity)": model.pvalues["diff_tot_dd"],
        "Slope (group)": model.params["group"],
        "p-value (group)": model.pvalues["group"],
        "R²": model.rsquared
    }

results_df = pd.DataFrame(reg_results_dd).T
print(results_df)

results_df.to_excel('my-path/result/reg_dd_severity_bygroup.xlsx')





#### EXPLORATORY ANALYSIS FOR mTBI SYPTOMS

###Compare symptoms distribution between groups

contingency_table = pd.crosstab(tbi_data["group"], tbi_data["tbi_severity"])
print(contingency_table)

chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
print(f"Statistique du chi² : {chi2_stat:.3f}")
print(f"P-value : {p_val:.3f}")

#No difference between children with DD and children with DD and mTBI


###Regression analysis
## x1 = symptoms, x2 = group, y = cbcl


reg_results_tbi = {}

for var in cbcl_var:
    data_clean = tbi_data[[var, "tbi_severity", "group"]].dropna()
    
    X = sm.add_constant(data_clean[["tbi_severity", "group"]])
    y = data_clean[var]
    
    model = sm.OLS(y, X).fit()
    
    reg_results_tbi[var] = {
        "Intercept": model.params["const"],
        "Slope (tbi_severity)": model.params["tbi_severity"],
        "p-value (tbi_severity)": model.pvalues["tbi_severity"],
        "Slope (group)": model.params["group"],
        "p-value (group)": model.pvalues["group"],
        "R²": model.rsquared
    }

results_df = pd.DataFrame(reg_results_tbi).T
print(results_df)

results_df.to_excel('my-path/result/reg_tbi_severity_bygroup.xlsx')

