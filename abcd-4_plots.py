#Copyright (C) 2025 Anne-Lise Marais

#This program is free software; you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation; version 3.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

#If you have any question, please contact Fanny Dégeilh at fanny.degeilh@inserm.fr


#PLOT

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


data = pd.read_excel('/Users/amarais/Documents/abcd/data/final_data.xlsx', header=0)

event_order = ["baseline_year_1_arm_1", "2_year_follow_up_y_arm_1", "4_year_follow_up_y_arm_1"]



##Graphs mean scores in y, age and group in x
#First graph for DD (0,1)
#Second graph for mTBI (0,1)
#First set of graphs for dsm-v variables (y-lim 51-56.5)

variables = ['cbcl_scr_dsm5_adhd_t', 'cbcl_scr_dsm5_anxdisord_t',
       'cbcl_scr_dsm5_conduct_t', 'cbcl_scr_dsm5_depress_t',
       'cbcl_scr_dsm5_opposit_t']
variables_fitted = ['adhd_fitted', 'anxdisord_fitted',
       'conduct_fitted', 'depress_fitted',
       'opposit_fitted']

for var_fitted in variables_fitted:
    plt.figure(figsize = (6,4))
    
    sns.pointplot(x='eventname', y=var_fitted, hue='dd_group', data=data, 
                  dodge=True, order=event_order,
                  linestyle='solid', errorbar=("ci",95))
    
    plt.xlabel("eventname")
    plt.ylabel("Estimated score")
    plt.ylim(51, 56.2)

    plt.title(f"Mean score for {var_fitted} by age and DD")
    plt.legend().remove()
    sns.despine(top=True, right=True)

    plt.tight_layout()
    
    output_path = f"/Users/amarais/Documents/abcd/result/figures/dd_{var_fitted}_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()


for var_fitted in variables_fitted:
    plt.figure(figsize=(6, 4))

    sns.pointplot(
        x='eventname', y=var_fitted, hue='tbi_group', data=data,
        dodge=True, order=event_order,
        linestyle='solid', errorbar=("ci",95),
        palette=["purple", "red"]  # Définition des couleurs
    )

    plt.xlabel("eventname")
    plt.ylabel("Estimated score")
    plt.ylim(51, 56.2)

    plt.title(f"Mean score for {var_fitted} by age and mTBI")
    plt.legend().remove()
    sns.despine(top=True, right=True)

    plt.tight_layout()

    output_path = f"/Users/amarais/Documents/abcd/result/figures/tbi_{var_fitted}_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    plt.show()


##Same for externalizing and internalizing scores (with modified y lim)


variables = ['cbcl_scr_syn_external_t',
       'cbcl_scr_syn_internal_t']
variables_fitted = ['external_fitted',
       'internal_fitted']


for var_fitted in (variables_fitted):
    plt.figure(figsize = (6,4))
    
    sns.pointplot(x='eventname', y=var_fitted, hue='dd_group', data=data, 
                  dodge=True, order=event_order,
                  linestyle='solid', errorbar=("ci",95))
    
    plt.xlabel("eventname")
    plt.ylabel("Estimated score")
    plt.ylim(41.8, 53.5)

    plt.title(f"Mean score for {var_fitted} by age and DD")
    plt.legend().remove()
    sns.despine(top=True, right=True)

    plt.tight_layout()
    
    output_path = f"/Users/amarais/Documents/abcd/result/figures/dd_{var_fitted}_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()




for var_fitted in (variables_fitted):
    plt.figure(figsize = (6,4))
    
    sns.pointplot(x='eventname', y=var_fitted, hue='tbi_group', data=data, 
                  dodge=True, order=event_order, palette=["purple", "red"],
                  linestyle='solid', errorbar=("ci",95))
    
    plt.xlabel("eventname")
    plt.ylabel("Estimated score")
    plt.ylim(41.8, 53.5)

    plt.title(f"Mean score for {var_fitted} by age and mTBI")
    plt.legend().remove()
    sns.despine(top=True, right=True)

    plt.tight_layout()
    
    output_path = f"/Users/amarais/Documents/abcd/result/figures/tbi_{var_fitted}_plot.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    plt.show()
