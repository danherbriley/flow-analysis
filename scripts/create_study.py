import optuna
# workaround to be able to create study with more refined params

study_name = "/project/project_465001585/viry/flow-analysis/data"
storage_name = f"sqlite:///{study_name}.db"
optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner(n_warmup_steps=5), storage=storage_name, study_name=study_name)