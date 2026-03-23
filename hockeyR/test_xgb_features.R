## test_xgb_features.R - test xgboost 3.x feature names after save/load
.libPaths("C:/Users/1143552/AppData/Local/R/win-library/4.5")
library(xgboost)
cat("xgboost:", as.character(packageVersion("xgboost")), "\n")

# Train a simple model
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
cat("Trained model feature_names (:::):", xgboost:::xgb.feature_names(m), "\n")
cat("Trained model num_features:", xgboost:::xgb.num_feature(m), "\n")

# Save and reload
xgb.save(m, "test_model.json")
m2 <- xgb.load("test_model.json")
cat("Reloaded model class:", class(m2), "\n")
tryCatch(
    cat("Reloaded feature_names:", xgboost:::xgb.feature_names(m2), "\n"),
    error = function(e) {
        cat("xgb.feature_names error:", conditionMessage(e), "\n")
    }
)

# Clean up
file.remove("test_model.json")
cat("Done\n")
