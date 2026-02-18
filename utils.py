import json

# Scikit-learn - models
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression

# XGBoost and CatBoost
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from lightgbm import LGBMClassifier, LGBMRegressor

def load_search_space(model_name, trial, path="configs/optuna_search"):
    with open(f"{path}/{model_name}.json", "r") as f:
        full_config = json.load(f)
    
    config = full_config[model_name]["model"]  # Access the model key
    
    params = {}
    for key, value in config.items():
        param_type = value[0].lstrip("?")
        if param_type == "int":
            # if value[3] doesnt exist
            if len(value) == 3:
                params[key] = trial.suggest_int(key, value[1], value[2])
            else:
                params[key] = trial.suggest_int(key, value[1], value[2], step=value[3])
        elif param_type == "float":
            params[key] = trial.suggest_float(key, value[1], value[2])
        elif param_type == "loguniform":
            params[key] = trial.suggest_float(key, value[1], value[2], log=True)
        elif param_type == "categorical":
            params[key] = trial.suggest_categorical(key, value[1])
        elif param_type == "uniform":
            params[key] = trial.suggest_float(key, value[1], value[2])
        else:
            raise ValueError(f"Unknown type: {param_type}")
    return params

def get_model(name, params, task_type, num_classes=None, SEED=42, device="cuda", class_weight=None):
    name = name.lower()
    task_type = task_type.lower()

    if name == "logisticregression":
        if task_type not in ["binary", "multiclass"]:
            raise ValueError("LogisticRegression only supports binary and multiclass tasks.")
        
        if class_weight is not None:
            params["class_weight"] = class_weight
        
        return LogisticRegression(**params, random_state=SEED)
    
    elif name == "linearregression":
        if task_type != "regression":
            raise ValueError("LinearRegression only supports regression tasks.")
        return LinearRegression(**params)

    elif name == "xgboost":
        if class_weight is not None and task_type == "binary":
        # Compute scale_pos_weight from class_weight
            if isinstance(class_weight, dict) and 0 in class_weight and 1 in class_weight:
                pos_weight = class_weight[1] / class_weight[0]
                params["scale_pos_weight"] = pos_weight

        if task_type == "regression":
            return XGBRegressor(**params, random_state=SEED)
        elif task_type == "multiclass":
            return XGBClassifier(**params, objective='multi:softprob',
                                 num_class=num_classes, eval_metric='mlogloss',
                                 n_jobs=-1, random_state=SEED)
        else:  # binary classification
            return XGBClassifier(**params, objective='binary:logistic',
                                 eval_metric='logloss', n_jobs=-1, random_state=SEED)
        
    elif name == "lightgbm":
        if class_weight is not None and task_type == "binary":
            # Compute scale_pos_weight from class_weight
            if isinstance(class_weight, dict) and 0 in class_weight and 1 in class_weight:
                pos_weight = class_weight[1] / class_weight[0]
                params["scale_pos_weight"] = pos_weight

        if task_type == "regression":
            return LGBMRegressor(**params, random_state=SEED, verbosity=-1)
        elif task_type == "multiclass":
            return LGBMClassifier(**params, objective="multiclass",
                                  num_class=num_classes,
                                  n_jobs=-1, random_state=SEED, verbosity=-1)
        else:  # binary classification
            return LGBMClassifier(**params, objective="binary",
                                  n_jobs=-1, random_state=SEED, verbosity=-1)

    elif name == "catboost":
        if class_weight is not None:
            params["class_weights"] = [class_weight[k] for k in sorted(class_weight.keys())]

        if task_type == "regression":
            return CatBoostRegressor(**params, verbose=0, random_seed=SEED)
        elif task_type == "multiclass":
            return CatBoostClassifier(**params, loss_function='MultiClass',
                                      classes_count=num_classes, verbose=0, random_seed=SEED)
        else:  # binary classification
            return CatBoostClassifier(**params, loss_function='Logloss', verbose=0, random_seed=SEED)

    else:
        raise ValueError(f"Unknown model name: {name}")

