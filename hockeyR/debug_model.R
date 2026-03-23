## debug_model.R - check xgboost 3.x model feature names API
.libPaths("C:/Users/1143552/AppData/Local/R/win-library/4.5")
library(xgboost)
cat("xgboost:", as.character(packageVersion("xgboost")), "\n")

# Load the saved JSON model
m <- xgb.load("inst/extdata/xg_model_5v5.json")
cat("class:", class(m), "\n")
cat("length feature_names:", length(m$feature_names), "\n")
cat("class feature_names:", class(m$feature_names), "\n")
cat("feature_names:", head(m$feature_names, 10), "\n")
cat("available names/methods:", paste(names(m), collapse = ", "), "\n")
# Check via xgb.model.dt.tree
cat("nfeatures:", m$nfeatures, "\n")

# Make a tiny test matrix and model to check feature name behavior
set.seed(1)
mat <- matrix(rnorm(100), nrow = 20, ncol = 5)
colnames(mat) <- paste0("feat_", 1:5)
dtrain <- xgb.DMatrix(data = mat, label = rbinom(20, 1, 0.3))
cat("DMatrix colnames:", colnames(dtrain), "\n")
cat("DMatrix nrow:", nrow(dtrain), "\n")
tiny_m <- xgb.train(
    params = list(objective = "binary:logistic"),
    data = dtrain,
    nrounds = 5,
    verbose = 0
)
cat("tiny model feature_names:", tiny_m$feature_names, "\n")
cat("tiny model nfeatures:", tiny_m$nfeatures, "\n")
