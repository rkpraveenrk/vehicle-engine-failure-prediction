"""Streamlit prediction UI loads the model committed by GitHub Actions."""
import json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'predictive_maintenance'/'model_building'))
from features import RAW_FEATURES,add_features

MODEL=ROOT/'predictive_maintenance'/'models'/'best_vehicle_engine_failure_prediction_model_v1.joblib'
CONFIG=ROOT/'predictive_maintenance'/'models'/'config.json'
st.set_page_config(page_title='Vehicle Engine Failure Prediction',page_icon='🔧')
st.title('🔧 Vehicle Engine Failure Prediction - Capstone Project')
st.caption('Capstone demonstration only. Not approved for real-world maintenance decisions.')
if not MODEL.exists() or not CONFIG.exists():
    st.error('Model not available yet. Run the GitHub Actions pipeline successfully first.')
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL),json.loads(CONFIG.read_text())

model,config=load_model()
st.write('Enter six sensor measurements. Values should be expressed in the same units as the training data.')
with st.form('predict'):
    values={key:st.number_input(key,value=0.0,format='%.5f') for key in RAW_FEATURES}
    submit=st.form_submit_button('Predict engine condition')
if submit:
    raw=pd.DataFrame([values])
    engineered=add_features(raw)
    expected=config['feature_names']
    if any(name not in engineered.columns for name in expected):
        st.error('Training features do not match application inputs.')
        st.stop()
    X=engineered[expected]
    if X.isna().any().any():
        st.warning('Some derived features are undefined; training-fitted imputation will be used.')
    probability=float(model.predict_proba(X)[0,1])
    threshold=float(config['threshold'])
    st.metric('Estimated probability of Engine Condition = 1',f'{probability:.1%}')
    st.write('Decision threshold:',threshold)
    st.dataframe(raw,use_container_width=True)
    if probability>=threshold:
        st.warning('Flagged for inspection (predicted class 1)')
    else:
        st.success('Not flagged (predicted class 0)')
    st.caption('Model has limited test performance. Use alongside professional engineering assessment.')
