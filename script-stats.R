# Comparaison des niveaux de group_nom dans m_1
#posthoc_group <- emmeans(m_1, pairwise ~ group_nom, adjust = "bonferroni")
#print(posthoc_group)


# Chargement des bibliothèques
library(lme4)
library(lmerTest)  
library(readxl)
library(dplyr)
library(emmeans)
library(ggplot2)
library(openxlsx)



# Chargement des données
data <- read_excel("/Users/amarais/Documents/abcd/data/final_sample.xlsx")

data$demo_sex_v2 <- as.factor(data$demo_sex_v2)



#cbcl_scr_dsm5_adhd_t

m0_adhd <- lmer(cbcl_scr_dsm5_adhd_t ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_adhd <- lmer(cbcl_scr_dsm5_adhd_t ~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_adhd <- lmer(cbcl_scr_dsm5_adhd_t ~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m3_adhd <- lmer(cbcl_scr_dsm5_adhd_t ~ age0 * (dd_group + tbi_group) + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m4_adhd <- lmer(cbcl_scr_dsm5_adhd_t ~ age0 * dd_group * tbi_group + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)


anova_models <- anova(m0_adhd, m1_adhd, m2_adhd, m3_adhd, m4_adhd)
print(anova_models)
# best model sumary
best_adhd <- m2_adhd
summary_best_adhd <- summary(best_adhd)

# Extraire les p-values des coefficients
p_values <- summary_best_adhd$coefficients[, "Pr(>|t|)"]

# Correction FDR
p_values_fdr <- p.adjust(p_values, method = "fdr")

# Ajouter la correction à la table des résultats
summary_best_adhd$coefficients <- cbind(summary_best_adhd$coefficients, FDR_p = p_values_fdr)

# Afficher les résultats avec les p-values corrigées
print(summary_best_adhd)


data$adhd_fitted <- NA  
adhd_fitted <- fitted(best_adhd)
data[names(adhd_fitted), "adhd_fitted"] <- adhd_fitted




#cbcl_scr_dsm5_anxdisord_t

m0_anxdisord <- lmer(cbcl_scr_dsm5_anxdisord_t ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_anxdisord <- lmer(cbcl_scr_dsm5_anxdisord_t ~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_anxdisord <- lmer(cbcl_scr_dsm5_anxdisord_t ~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m3_anxdisord <- lmer(cbcl_scr_dsm5_anxdisord_t ~ age0 * (dd_group + tbi_group) + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m4_anxdisord <- lmer(cbcl_scr_dsm5_anxdisord_t ~ age0 * dd_group * tbi_group + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)


anova(m0_anxdisord, m1_anxdisord, m2_anxdisord, m3_anxdisord, m4_anxdisord)
# best model sumary
# Extraire les résultats du modèle
best_anxdisord <- m2_anxdisord
summary_best_anxdisord <- summary(best_anxdisord)

# Extraire les p-values des coefficients
p_values <- summary_best_anxdisord$coefficients[, "Pr(>|t|)"]

# Correction FDR
p_values_fdr <- p.adjust(p_values, method = "fdr")

# Ajouter la correction à la table des résultats
summary_best_anxdisord$coefficients <- cbind(summary_best_anxdisord$coefficients, FDR_p = p_values_fdr)

# Afficher les résultats avec les p-values corrigées
print(summary_best_anxdisord)


data$anxdisord_fitted <- NA  
anxdisord_fitted <- fitted(best_anxdisord)
data[names(anxdisord_fitted), "anxdisord_fitted"] <- anxdisord_fitted



#cbcl_scr_dsm5_conduct_t

m0_conduct <- lmer(cbcl_scr_dsm5_conduct_t ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_conduct <- lmer(cbcl_scr_dsm5_conduct_t ~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_conduct <- lmer(cbcl_scr_dsm5_conduct_t ~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m3_conduct <- lmer(cbcl_scr_dsm5_conduct_t ~ age0 * (dd_group + tbi_group) + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m4_conduct <- lmer(cbcl_scr_dsm5_conduct_t ~ age0 * dd_group * tbi_group + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_conduct, m1_conduct, m2_conduct, m3_conduct, m4_conduct)
# best model sumary
best_conduct <- m2_conduct
summary_best_conduct <- summary(best_conduct)

# Extraire les p-values des coefficients
p_values <- summary_best_conduct$coefficients[, "Pr(>|t|)"]

# Correction FDR
p_values_fdr <- p.adjust(p_values, method = "fdr")

# Ajouter la correction à la table des résultats
summary_best_conduct$coefficients <- cbind(summary_best_conduct$coefficients, FDR_p = p_values_fdr)

# Afficher les résultats avec les p-values corrigées
print(summary_best_conduct)

data$conduct_fitted <- NA  
conduct_fitted <- fitted(best_conduct)
data[names(conduct_fitted), "conduct_fitted"] <- conduct_fitted





#cbcl_scr_dsm5_depress_t

m0_depress <- lmer(cbcl_scr_dsm5_depress_t ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_depress <- lmer(cbcl_scr_dsm5_depress_t ~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_depress <- lmer(cbcl_scr_dsm5_depress_t ~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m3_depress <- lmer(cbcl_scr_dsm5_depress_t ~ age0 * (dd_group + tbi_group) + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m4_depress <- lmer(cbcl_scr_dsm5_depress_t ~ age0 * dd_group * tbi_group + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_depress, m1_depress, m2_depress, m3_depress, m4_depress)
# best model sumary
best_depress <- m2_depress
summary_best_depress <- summary(best_depress)

# Extraire les p-values des coefficients
p_values <- summary_best_depress$coefficients[, "Pr(>|t|)"]

# Correction FDR
p_values_fdr <- p.adjust(p_values, method = "fdr")

# Ajouter la correction à la table des résultats
summary_best_depress$coefficients <- cbind(summary_best_depress$coefficients, FDR_p = p_values_fdr)

# Afficher les résultats avec les p-values corrigées
print(summary_best_depress)


data$depress_fitted <- NA  
depress_fitted <- fitted(best_depress)
data[names(depress_fitted), "depress_fitted"] <- depress_fitted






#cbcl_scr_dsm5_opposit_t

m0_opposit <- lmer(cbcl_scr_dsm5_opposit_t ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_opposit <- lmer(cbcl_scr_dsm5_opposit_t ~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_opposit <- lmer(cbcl_scr_dsm5_opposit_t ~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m3_opposit <- lmer(cbcl_scr_dsm5_opposit_t ~ age0 * (dd_group + tbi_group) + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m4_opposit <- lmer(cbcl_scr_dsm5_opposit_t ~ age0 * dd_group * tbi_group + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_opposit, m1_opposit, m2_opposit, m3_opposit, m4_opposit)
# best model sumary
best_opposit <- m2_opposit
summary_best_opposit <- summary(best_opposit)

# Extraire les p-values des coefficients
p_values <- summary_best_opposit$coefficients[, "Pr(>|t|)"]

# Correction FDR
p_values_fdr <- p.adjust(p_values, method = "fdr")

# Ajouter la correction à la table des résultats
summary_best_opposit$coefficients <- cbind(summary_best_opposit$coefficients, FDR_p = p_values_fdr)

# Afficher les résultats avec les p-values corrigées
print(summary_best_opposit)


data$opposit_fitted <- NA  
opposit_fitted <- fitted(best_opposit)
data[names(opposit_fitted), "opposit_fitted"] <- opposit_fitted




#cbcl_scr_syn_external_t

m0_external <- lmer(cbcl_scr_syn_external_t ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_external <- lmer(cbcl_scr_syn_external_t ~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_external <- lmer(cbcl_scr_syn_external_t ~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m3_external <- lmer(cbcl_scr_syn_external_t ~ age0 * (dd_group + tbi_group) + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m4_external <- lmer(cbcl_scr_syn_external_t ~ age0 * dd_group * tbi_group + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_external, m1_external, m2_external, m3_external, m4_external)
# best model sumary
best_external <- m2_external
summary_best_external <- summary(best_external)

# Extraire les p-values des coefficients
p_values <- summary_best_external$coefficients[, "Pr(>|t|)"]

# Correction FDR
p_values_fdr <- p.adjust(p_values, method = "fdr")

# Ajouter la correction à la table des résultats
summary_best_external$coefficients <- cbind(summary_best_external$coefficients, FDR_p = p_values_fdr)

# Afficher les résultats avec les p-values corrigées
print(summary_best_external)


data$external_fitted <- NA  
external_fitted <- fitted(best_external)
data[names(external_fitted), "external_fitted"] <- external_fitted




#cbcl_scr_syn_internal_t

m0_internal <- lmer(cbcl_scr_syn_internal_t ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_internal <- lmer(cbcl_scr_syn_internal_t ~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_internal <- lmer(cbcl_scr_syn_internal_t ~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m3_internal <- lmer(cbcl_scr_syn_internal_t ~ age0 * (dd_group + tbi_group) + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m4_internal <- lmer(cbcl_scr_syn_internal_t ~ age0 * dd_group * tbi_group + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_internal, m1_internal, m2_internal, m3_internal, m4_internal)
# best model sumary
best_internal <- m2_internal
summary_best_internal <- summary(best_internal)

# Extraire les p-values des coefficients
p_values <- summary_best_internal$coefficients[, "Pr(>|t|)"]

# Correction FDR
p_values_fdr <- p.adjust(p_values, method = "fdr")

# Ajouter la correction à la table des résultats
summary_best_internal$coefficients <- cbind(summary_best_internal$coefficients, FDR_p = p_values_fdr)

# Afficher les résultats avec les p-values corrigées
print(summary_best_internal)


data$internal_fitted <- NA  
internal_fitted <- fitted(best_internal)
data[names(internal_fitted), "internal_fitted"] <- internal_fitted



library(writexl)

write_xlsx(data, "/Users/amarais/Documents/abcd/data/final_data.xlsx")


















# Transformer les coefficients (effets fixes) en tableau
fixed_effects <- as.data.frame(summary_best_adhd$coefficients)

# Transformer l'ANOVA en tableau
anova_df <- as.data.frame(anova_models)

# Extraire les effets aléatoires
random_effects <- as.data.frame(VarCorr(m2_adhd))

# Résumé des critères du modèle (AIC, BIC, LogLik, etc.)
model_info <- data.frame(
  AIC = AIC(m2_adhd),
  BIC = BIC(m2_adhd),
  logLik = logLik(m2_adhd),
  REML_Criterion = summary_m2_adhd$AICtab[1]
)

# Créer un fichier Excel
wb <- createWorkbook()

# Ajouter les différentes feuilles avec leurs données
addWorksheet(wb, "Fixed Effects")
writeData(wb, "Fixed Effects", fixed_effects)

addWorksheet(wb, "Model Comparison")
writeData(wb, "Model Comparison", anova_df)

addWorksheet(wb, "Random Effects")
writeData(wb, "Random Effects", random_effects)

addWorksheet(wb, "Model Info")
writeData(wb, "Model Info", model_info)

# Sauvegarde du fichier
saveWorkbook(wb, "summary_best_adhd.xlsx", overwrite = TRUE)

























#nihtbx_pattern_fc

m0_pattern <- lmer(nihtbx_pattern_fc ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_pattern <- lmer(nihtbx_pattern_fc ~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_pattern <- lmer(nihtbx_pattern_fc ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m3_pattern <- lmer(nihtbx_pattern_fc ~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m4_pattern <- lmer(nihtbx_pattern_fc ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)


anova(m0_pattern, m1_pattern, m2_pattern, m3_pattern, m4_pattern)
# best model sumary
summary(m1_pattern)


data$pattern_fitted <- NA  
m1_pattern_fitted <- fitted(m1_pattern)
data[names(m1_pattern_fitted), "pattern_fitted"] <- m1_pattern_fitted

#sans ribbon
ggplot(data, aes(event0, nihtbx_pattern_fc, color = interaction(dd_group, tbi_group))) +
  stat_summary(aes(y = nihtbx_pattern_fc), fun = mean, geom = "line", alpha = 0.2) +  # Correction ici
  stat_summary(aes(y = pattern_fitted, color = interaction(dd_group, tbi_group)), fun = mean, geom = "line", linetype = "dotted") +  
  theme_bw(base_size = 12) + 
  labs(title = "pattern",
       y = "score", 
       x = "Year")

#avec ribbon

ggplot(data, aes(event0, nihtbx_pattern_fc, color = interaction(dd_group, tbi_group))) +
  stat_summary(fun.data = mean_se, geom = "ribbon", alpha = 0.2) +  
  stat_summary(aes(y = pattern_fitted, color = interaction(dd_group, tbi_group)), fun = mean, geom = "line", linetype = "dotted") +  
  theme_bw(base_size = 12) + 
  labs(title = "pattern",
       y = "score", 
       x = "Year")




#nihtbx_picvocab_fc

m0_picvocab <- lmer(nihtbx_picvocab_fc ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_picvocab <- lmer(nihtbx_picvocab_fc ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_picvocab <- lmer(nihtbx_picvocab_fc ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_picvocab, m1_picvocab, m2_picvocab)
# best model sumary
summary(m1_picvocab)


data$picvocab_fitted <- NA  
m1_picvocab_fitted <- fitted(m1_picvocab)
data[names(m1_picvocab_fitted), "picvocab_fitted"] <- m1_picvocab_fitted


ggplot(data, aes(event0, nihtbx_picvocab_fc, color = interaction(dd_group, tbi_group))) +
  stat_summary(aes(y = nihtbx_picvocab_fc), fun = mean, geom = "line", alpha = 0.2) +  # Correction ici
  stat_summary(aes(y = picvocab_fitted, color = interaction(dd_group, tbi_group)), fun = mean, geom = "line", linetype = "dotted") +  
  theme_bw(base_size = 12) + 
  labs(title = "picvocab",
       y = "score", 
       x = "Year")

#nihtbx_reading_fc

m0_reading <- lmer(nihtbx_reading_fc ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_reading <- lmer(nihtbx_reading_fc ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_reading <- lmer(nihtbx_reading_fc ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_reading, m1_reading, m2_reading)
# best model sumary
summary(m1_reading)


data$reading_fitted <- NA  
m1_reading_fitted <- fitted(m1_reading)
data[names(m1_reading_fitted), "reading_fitted"] <- m1_reading_fitted


ggplot(data, aes(event0, nihtbx_reading_fc, color = interaction(dd_group, tbi_group))) +
  stat_summary(aes(y = nihtbx_reading_fc), fun = mean, geom = "line", alpha = 0.2) +  # Correction ici
  stat_summary(aes(y = reading_fitted, color = interaction(dd_group, tbi_group)), fun = mean, geom = "line", linetype = "dotted") +  
  theme_bw(base_size = 12) + 
  labs(title = "reading",
       y = "score", 
       x = "Year")



#nihtbx_picture_fc

m0_picture <- lmer(nihtbx_picture_fc ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_picture <- lmer(nihtbx_picture_fc ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_picture <- lmer(nihtbx_picture_fc ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_picture, m1_picture, m2_picture)
# best model sumary
summary(m1_picture)


data$picture_fitted <- NA  
m1_picture_fitted <- fitted(m1_picture)
data[names(m1_picture_fitted), "picture_fitted"] <- m1_picture_fitted


ggplot(data, aes(event0, nihtbx_picture_fc, color = interaction(dd_group, tbi_group))) +
  stat_summary(aes(y = nihtbx_picture_fc), fun = mean, geom = "line", alpha = 0.2) +  # Correction ici
  stat_summary(aes(y = picture_fitted, color = interaction(dd_group, tbi_group)), fun = mean, geom = "line", linetype = "dotted") +  
  theme_bw(base_size = 12) + 
  labs(title = "picture",
       y = "score", 
       x = "Year")





#nihtbx_flanker_fc

m_0 <- lmer(nihtbx_flanker_fc ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m_1 <- lmer(nihtbx_flanker_fc ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m_2 <- lmer(nihtbx_flanker_fc ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m_0, m_1, m_2)
# best model sumary
##Non significatif





#upps_y_ss_lack_of_perseverance

m0_perseverance <- lmer(upps_y_ss_lack_of_perseverance ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_perseverance <- lmer(upps_y_ss_lack_of_perseverance ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_perseverance <- lmer(upps_y_ss_lack_of_perseverance ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_perseverance, m1_perseverance, m2_perseverance)
# best model sumary
summary(m1_perseverance)


data$perseverance_fitted <- NA  
m1_perseverance_fitted <- fitted(m1_perseverance)
data[names(m1_perseverance_fitted), "perseverance_fitted"] <- m1_perseverance_fitted


ggplot(data, aes(event0, upps_y_ss_lack_of_perseverance, color = interaction(dd_group, tbi_group))) +
  stat_summary(aes(y = upps_y_ss_lack_of_perseverance), fun = mean, geom = "line", alpha = 0.2) +  # Correction ici
  stat_summary(aes(y = perseverance_fitted, color = interaction(dd_group, tbi_group)), fun = mean, geom = "line", linetype = "dotted") +  
  theme_bw(base_size = 12) + 
  labs(title = "perseverance",
       y = "score", 
       x = "Year")



#upps_y_ss_lack_of_planning

m_0 <- lmer(upps_y_ss_lack_of_planning ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m_1 <- lmer(upps_y_ss_lack_of_planning ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m_2 <- lmer(upps_y_ss_lack_of_planning ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m_0, m_1, m_2)
# best model sumary
#Non significatif




#upps_y_ss_negative_urgency

m0_negurgency <- lmer(upps_y_ss_negative_urgency ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_negurgency <- lmer(upps_y_ss_negative_urgency ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_negurgency <- lmer(upps_y_ss_negative_urgency ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_negurgency, m1_negurgency, m2_negurgency)
# best model sumary
summary(m2_negurgency)


data$negurgency_fitted <- NA  
m2_negurgency_fitted <- fitted(m2_negurgency)
data[names(m2_negurgency_fitted), "negurgency_fitted"] <- m2_negurgency_fitted


ggplot(data, aes(event0, upps_y_ss_negative_urgency, color = interaction(dd_group, tbi_group))) +
  stat_summary(aes(y = upps_y_ss_negative_urgency), fun = mean, geom = "line", alpha = 0.2) +  # Correction ici
  stat_summary(aes(y = negurgency_fitted, color = interaction(dd_group, tbi_group)), fun = mean, geom = "line", linetype = "dotted") +  
  theme_bw(base_size = 12) + 
  labs(title = "negurgency",
       y = "score", 
       x = "Year")



#upps_y_ss_positive_urgency

m_0 <- lmer(upps_y_ss_positive_urgency ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m_1 <- lmer(upps_y_ss_positive_urgency ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m_2 <- lmer(upps_y_ss_positive_urgency ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m_0, m_1, m_2)
# best model sumary
#Non significatif




#upps_y_ss_sensation_seeking

m0_sseeking <- lmer(upps_y_ss_sensation_seeking ~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m1_sseeking <- lmer(upps_y_ss_sensation_seeking ~ age0 + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)
m2_sseeking <- lmer(upps_y_ss_sensation_seeking ~ age0 * (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id), data = data)

anova(m0_sseeking, m1_sseeking, m2_sseeking)
# best model sumary
summary(m1_sseeking)


data$sseeking_fitted <- NA  
m1_sseeking_fitted <- fitted(m1_sseeking)
data[names(m1_sseeking_fitted), "sseeking_fitted"] <- m1_sseeking_fitted


ggplot(data, aes(event0, upps_y_ss_sensation_seeking, color = interaction(dd_group, tbi_group))) +
  stat_summary(aes(y = upps_y_ss_sensation_seeking), fun = mean, geom = "line", alpha = 0.2) +  # Correction ici
  stat_summary(aes(y = sseeking_fitted, color = interaction(dd_group, tbi_group)), fun = mean, geom = "line", linetype = "dotted") +  
  theme_bw(base_size = 12) + 
  labs(title = "sseeking",
       y = "score", 
       x = "Year")



