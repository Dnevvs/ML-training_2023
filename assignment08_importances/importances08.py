def get_coef_from_shap_values(shap_values, X_train_scaled):
    # YOUR CODE HERE
    return (shap_values.values *
            (X_train_scaled - X_train_scaled.mean(0))).mean(0)
