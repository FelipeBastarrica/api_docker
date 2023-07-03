import wandb
run = wandb.login(key="<WANDB_KEY>")
artifact = run.use_artifact('bgm-team/ml-en-produccion/run_qmo2l35u_model:v14', type='model')
artifact_dir = artifact.download()
