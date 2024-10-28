# Misc imports
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, List
# Models and evaluation
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
# Import additional regression models
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor

# Local imports
from .analyze_inf import load_cal_inference_stats
from .analysis_utils.parse_sweep import get_global_optimal_parameter 


def gather_baseline_dfs(
    data_df,
    baseline_dict,
    add_naive_baselines: bool = True,
    add_tuned_baselines: bool = True
):
    # First, we need to get the dataset that the df corresponds to
    unique_datasets = data_df['data'].unique()
    # Assert that we only have one dataset
    assert len(unique_datasets) == 1
    dataset = unique_datasets[0].split('.')[-1]

    # Next, we use the baseline_dict to get the baselines
    root_baseline_dir = baseline_dict[dataset]
    results_cfgs = {
        "log":{
            "root": root_baseline_dir,
            "inference_group": []
        },
        "options": {
            "verify_graceful_exit": True,
            "equal_rows_per_cfg_assert": False 
        }
    }

    # Make a dict to store the baselines
    baselines = {}

    # We can add to the plot the naive baselines which are
    # - just the default sum up probs 
    # - threshold at 0.5.
    if add_naive_baselines:
        naive_result_cfg = results_cfgs.copy()
        naive_result_cfg['log']['inference_group'] = [
            "Base_CrossEntropy",
            "Base_SoftDice"
        ]
        # Load the naive baselines
        naive_inference_df = load_cal_inference_stats(
            results_cfg=naive_result_cfg,
            load_cached=True
        )
        baselines.update({"base": naive_inference_df})

    # Then we have another set of baselines that are when we tune the
    # on the calibratio set for the 
    # - hard threshold
    # - soft temperature
    if add_tuned_baselines:
        # Load the threshold baselines.
        tuned_threshold_cfg = results_cfgs.copy()
        tuned_threshold_cfg['log']['inference_group'] = [
            "Optimal_Temperature_CrossEntropy",
            "Optimal_Temperature_SoftDice",
            "Optimal_Threshold_CrossEntropy",
            "Optimal_Threshold_SoftDice"
        ]
        # Load the tuned baselines
        tuned_threshold_df = load_cal_inference_stats(
            results_cfg=tuned_threshold_cfg,
            load_cached=True
        )

        # Load the temperatures baselines.
        tuned_temp_cfg = results_cfgs.copy()
        tuned_temp_cfg['log']['inference_group'] = [
            "Optimal_Temperature_CrossEntropy",
            "Optimal_Temperature_SoftDice",
            "Optimal_Threshold_CrossEntropy",
            "Optimal_Threshold_SoftDice"
        ]
        # Load the tuned baselines
        tuned_temperature_df = load_cal_inference_stats(
            results_cfg=tuned_temp_cfg,
            load_cached=True
        )
        baselines.update(
            {
                "threshold": tuned_threshold_df,
                "temperature": tuned_temperature_df
            }
        )
    
    return baselines


def get_baseline_values(y_key, baselines_dict):

    method_dfs = []

    if 'threshold' in baselines_dict:
        thresh_opt_vals = get_global_optimal_parameter(
            baselines_dict['threshold'], 
            sweep_key='threshold', 
            y_key="hard_"+y_key,
            group_keys=['split', 'loss_func_class']
        )
        # Rename the y_key to be the same as the input y_key
        thresh_opt_vals.rename(columns={f"hard_{y_key}": y_key}, inplace=True)
        thresh_opt_vals['method'] = "threshold_tuning"
        method_dfs.append(thresh_opt_vals)

    if 'temperature' in baselines_dict:
        temp_opt_vals = get_global_optimal_parameter(
            baselines_dict['temperature'], 
            sweep_key='temperature', 
            y_key="soft_"+y_key,
            group_keys=['split', 'loss_func_class']
        )
        # Rename the y_key to be the same as the input y_key
        temp_opt_vals.rename(columns={f"soft_{y_key}": y_key}, inplace=True)
        # Move it to be the last column
        temp_opt_vals['method'] = "temperature_tuning"
        method_dfs.append(temp_opt_vals)
    
    # Concatenate the dataframes and return
    return pd.concat(method_dfs)
    

def fit_posthoc_calibrators(
    data_df,
    train_split: str,
    val_split: str,
    x_feats: List[str],
    global_opt_temp: Optional[float] = None,
):
    # Assert that the number of rows in the dataframe is equal to the number of uniuqe
    # subjects, otherwise we have a problem.
    assert data_df['data_id'].nunique() == data_df.shape[0], "Dataframe has duplicate data ids"

    # Split the data into training and validation sets based on 'split' column
    df_train = data_df[data_df['split'] == train_split]
    df_val = data_df[data_df['split'] == val_split]

    X_train = df_train[x_feats]
    X_val = df_val[x_feats]

    # Prepare the features (X) and target (y)
    y_train = df_train['temperature']
    y_val = df_val['temperature']

    # Define the models to test, including additional regressors
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(),
        'Lasso Regression': Lasso(),
        'ElasticNet Regression': ElasticNet(),
        'Kernel Ridge Regression': KernelRidge(),
        'Huber Regressor': HuberRegressor(),
        'Polynomial Regression (degree=2)': make_pipeline(PolynomialFeatures(degree=2), LinearRegression()),
        'Polynomial Regression (degree=3)': make_pipeline(PolynomialFeatures(degree=3), LinearRegression()),
        'K-Nearest Neighbors': KNeighborsRegressor(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(random_state=42),
        'Extra Trees Regressor': ExtraTreesRegressor(random_state=42),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
        'AdaBoost Regressor': AdaBoostRegressor(random_state=42),
        'Support Vector Regression': SVR(),
        'MLP Regressor': MLPRegressor(random_state=42, max_iter=1000),
    }

    # Dictionary to store the results
    results = {}

    # Train and evaluate each model
    for name, model in models.items():
        # Fit the model
        model.fit(X_train, y_train)
        # Predict on validation set
        y_pred = model.predict(X_val)
        # Evaluate
        mse = mean_squared_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        # Store the results
        results[name] = {'MSE': mse, 'R2 Score': r2}

    # If we have a global optimal temperature, we can evaluate that as well
    if global_opt_temp is not None:
        # Make the y_pred by repeating the global temperature for each validation sample
        y_pred_global_temp = np.repeat(global_opt_temp, len(y_val))
        # Evaluate
        mse = mean_squared_error(y_val, y_pred_global_temp)
        r2 = r2_score(y_val, y_pred_global_temp)
        # Store the results
        results['Global Optimal Temperature'] = {'MSE': mse, 'R2 Score': r2}
    
    return results


def viz_posthoc_calibrators(models_dict, X_val, y_val, global_temp = None):
    # Determine the layout of the subplot grid
    num_models = len(models_dict)
    cols = 2  # Reduce the number of columns for larger subplots
    rows = math.ceil(num_models / cols)  # Calculate the number of rows needed

    # Increase the figure size for larger subplots
    fig, axs = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
    axs = axs.flatten()  # Flatten the array of axes for easy iteration

    # Plot each model's predictions in a subplot
    for idx, (name, model) in enumerate(models_dict.items()):
        y_pred = model.predict(X_val)
        axs[idx].scatter(X_val, y_val, color='black', label='Actual Validation Data')
        axs[idx].scatter(X_val, y_pred, color='blue', label='Predicted Data')
        # Place a horizontal line at the global optimal temperature
        if global_temp is not None:
            axs[idx].axhline(global_temp, color='red', linestyle='--', label='Global Optimal Temperature')
        axs[idx].set_title(f"{name}")
        axs[idx].set_xlabel('Hard Volume')
        axs[idx].set_ylabel('Temperature')
        axs[idx].legend()

    # Hide any unused subplots if the grid is larger than the number of models
    for idx in range(num_models, len(axs)):
        fig.delaxes(axs[idx])

    plt.tight_layout()
    plt.show()