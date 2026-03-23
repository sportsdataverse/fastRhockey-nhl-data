## debug_model2.R - explore all available xgb.Booster APIs in xgboost 3.x
.libPaths("C:/Users/1143552/AppData/Local/R/win-library/4.5")
library(xgboost)

# Train a tiny model with named features
set.seed(1)
mat <- matrix(rnorm(100), nrow = 20, ncol = 5)
colnames(mat) <- paste0("feat_", 1:5)
dtrain <- xgb.DMatrix(data = mat, label = rbinom(20, 1, 0.3))
m <- xgb.train(
    params = list(objective = "binary:logistic"),
    data = dtrain,
    nrounds = 5,
    verbose = 0
)

# Try all ways to get feature names from the model
cat("== Exploring xgb.Booster in xgboost 3.x ==\n")
cat("class(m):", class(m), "\n")
cat("typeof(m):", typeof(m), "\n")
cat("names(m):", paste(names(m), collapse = ", "), "\n")
cat("isS4(m):", isS4(m), "\n")

# Try dump_model
dump <- xgb.dump(m, with_stats = FALSE)
cat("\nxgb.dump first line:", dump[1], "\n")

# Try xgb.attributes
attrs <- xgb.attributes(m)
cat("xgb.attributes:", paste(names(attrs), collapse = ", "), "\n")

# Try getting feature names from attributes
cat("attr feature_names:", xgb.attributes(m)$feature_names, "\n")

# Try xgb.model.dt.tree to get feature info
tryCatch(
    {
        tree_dt <- xgb.model.dt.tree(model = m)
        cat(
            "Tree features:",
            paste(unique(tree_dt$Feature), collapse = ", "),
            "\n"
        )
    },
    error = function(e) {
        cat("xgb.model.dt.tree error:", conditionMessage(e), "\n")
    }
)

# Try xgboost:::xgb.get.fnames or similar internal
cat("\nSearching for feature_names functions:\n")
fns <- ls(getNamespace("xgboost"), pattern = "feature|fname")
cat(paste(fns, collapse = "\n"), "\n")

# Try xgb.Booster field access methods
cat("\nm$ptr class:", class(m$ptr), "\n")
