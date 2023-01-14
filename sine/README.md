# Adaptive Image Editing with Text-to-Image Diffusion Models 

The scripts in this repo are mostly derived from [SINE](https://github.com/zhang-zx/SINE) original github repo. To applied their model on our tasks, we use our own scripts to fine tune the models and use jupyter-notebook (`notebooks/`) to visualized the inference results.

## How to reproduce?

The models can be downloaded via `download_finetune_models.sh`, however,  if you want to reproduce our models yourselves, please refer to`fine_tune.sh` (need to download pretrained model and data first, `download_pretrained_model.sh` and `download_data.sh`). Tt contains several steps to reproduce all models, one can manually run the commands contains in the scripts.

## Run environment

Please refer to `environment.yaml` to rebuild the conda environment where our code run on.

## Inference of differest task

Please refer to notebooks listed in `notebooks/`

