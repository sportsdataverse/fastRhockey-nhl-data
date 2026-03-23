## test_rda_roundtrip.R - verify xgboost 3.x models survive R save/load
.libPaths("C:/Users/1143552/AppData/Local/R/win-library/4.5")
library(xgboost)

set.seed(1)
mat <- matrix(rnorm(500), nrow = 100, ncol = 5)
colnames(mat) <- paste0("feature_", 1:5)
dtrain <- xgb.DMatrix(data = mat, label = rbinom(100, 1, 0.3))
m <- xgb.train(
    params = list(objective = "binary:logistic"),
    data = dtrain,
    nrounds = 10,
    verbose = 0
)
cat("Before save - feature_names:", xgboost:::xgb.feature_names(m), "\n")

# Save via R's save() (same as sysdata.rda)
save(m, file = "test_model.rda")

# Load in fresh env
env2 <- new.env()
load("test_model.rda", envir = env2)
m2 <- env2$m
cat("After load(rda) - class:", class(m2), "\n")
cat("After load(rda) - feature_names:", xgboost:::xgb.feature_names(m2), "\n")

# Test prediction
preds <- predict(m2, xgboost::xgb.DMatrix(mat))
cat("Prediction range:", round(range(preds), 4), "\n")

cat("\nSUCCESS: xgboost 3.x models survive R save/load cycle\n")
file.remove("test_model.rda")
