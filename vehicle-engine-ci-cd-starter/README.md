# Vehicle Engine Failure Prediction — GitHub Actions + Streamlit

This repo contains a 4-job machine learning workflow: register-dataset → data-prep → model-training → publish-model.
The Streamlit application is deployed *once* through Streamlit Community Cloud using entrypoint `predictive_maintenance/hosting/app.py`, branch `main`, and will automatically pick up subsequent model commits.

## Required first step
Place the original CSV at `predictive_maintenance/data/engine_data.csv`. Do not replace it with made-up data. Check permission to publish; public GitHub repositories expose uploaded data.

## GitHub Actions
Run `.github/workflows/pipeline.yml` using the Actions tab or push changes to the tracked paths.
The train/test splits are transferred with GitHub Actions artifacts; the best fitted model and reports are committed to `main` automatically.
The MLflow experiment database is retained in the model-results workflow artifact (not committed to Git).

## Local testing
```
python -m pip install -r predictive_maintenance/requirements.txt
python predictive_maintenance/model_building/data_register.py
python predictive_maintenance/model_building/prep.py
python predictive_maintenance/model_building/train.py --iterations 12
python -m pip install -r predictive_maintenance/hosting/requirements.txt
streamlit run predictive_maintenance/hosting/app.py
```

## Security
Never commit Google Drive credentials, Hugging Face tokens, or GitHub personal-access tokens.
Use only shareable datasets. This app is a demonstration; model accuracy/recall must be reported from the real training run.
