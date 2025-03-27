## Property of Anne-Lise Marais, 2025 {maraisannelise98@gmail.com}

##This code executes the main analysis
##Hierarchical linear model
##One set of models by CBCL variable

##scripted on R

library(lme4)
library(lmerTest)  
library(readxl)
library(openxlsx)



data <- read_excel("/Users/amarais/Documents/abcd/data/final_sample.xlsx")

data$demo_sex_v2 <- as.factor(data$demo_sex_v2)


###-------------MAIN ANALYSIS-----------

analyze_variable <- function(var_name, data) {

  formula_0 <- as.formula(paste(var_name, "~ age0 + demo_sex_v2 + prnt_educ + (1 | src_subject_id)"))
  formula_1 <- as.formula(paste(var_name, "~ age0 + (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id)"))
  formula_2 <- as.formula(paste(var_name, "~ age0 * (dd_group + tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id)"))
  formula_3 <- as.formula(paste(var_name, "~ age0 * (dd_group + tbi_group) + (dd_group * tbi_group) + demo_sex_v2 + prnt_educ + (1 | src_subject_id)"))
  formula_4 <- as.formula(paste(var_name, "~ age0 * dd_group * tbi_group + demo_sex_v2 + prnt_educ + (1 | src_subject_id)"))


  m0 <- lmer(formula_0, data = data)
  m1 <- lmer(formula_1, data = data)
  m2 <- lmer(formula_2, data = data)
  m3 <- lmer(formula_3, data = data)
  m4 <- lmer(formula_4, data = data)

  model_comparison <- anova(m0, m1, m2, m3, m4)
  
  significant_models <- which(model_comparison$`Pr(>Chisq)` < 0.05)

  if (length(significant_models) > 0) {
    best_model_index <- significant_models[which.min(model_comparison$AIC[significant_models])]
  } else {
    best_model_index <- 1
  }

  best_model <- list(m0, m1, m2, m3, m4)[[best_model_index]]
  best_model_name <- paste0("m_", best_model_index - 1)

  summary_best <- summary(best_model)

  p_values <- summary_best$coefficients[, "Pr(>|t|)"]

  p_values_fdr <- p.adjust(p_values, method = "fdr")

  summary_best$coefficients <- cbind(summary_best$coefficients, FDR_p = p_values_fdr)

  print(summary_best)

  fixed_effects <- as.data.frame(summary_best$coefficients)
  fixed_effects <- cbind(Variable = rownames(fixed_effects), fixed_effects)
  rownames(fixed_effects) <- NULL  

  random_effects <- as.data.frame(VarCorr(best_model))

  model_info <- data.frame(
    Model = best_model_name,
    AIC = AIC(best_model),
    BIC = BIC(best_model),
    logLik = logLik(best_model),
    REML_Criterion = summary_best$AICtab[1]
  )

  wb <- createWorkbook()

  addWorksheet(wb, "Fixed Effects")
  writeData(wb, "Fixed Effects", fixed_effects)

  addWorksheet(wb, "Model Comparison")
  writeData(wb, "Model Comparison", model_comparison)

  addWorksheet(wb, "Random Effects")
  writeData(wb, "Random Effects", random_effects)

  addWorksheet(wb, "Model Info")
  writeData(wb, "Model Info", model_info)

  output_file <- paste0("/Users/amarais/Documents/abcd/result/stats/", var_name, "_stats.xlsx")

  saveWorkbook(wb, output_file, overwrite = TRUE)

  data[[paste0(var_name, "_fitted")]] <- NA
  fitted_values <- fitted(best_model)
  data[names(fitted_values), paste0(var_name, "_fitted")] <- fitted_values


  return(list(best_model = best_model, data = data))
}

variables <- c("cbcl_scr_dsm5_adhd_t", "cbcl_scr_dsm5_anxdisord_t", 
               "cbcl_scr_dsm5_conduct_t", "cbcl_scr_dsm5_depress_t", 
               "cbcl_scr_dsm5_opposit_t", "cbcl_scr_syn_external_t", 
               "cbcl_scr_syn_internal_t")


for (var in variables) {
  analyze_variable(var, data)
}



###--------COMPLEMENTARY ANALYSIS--------------


