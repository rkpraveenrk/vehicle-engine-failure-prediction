"""XGBoost + SMOTE CV + SQLite MLflow + export deployable artifacts."""
import argparse
import json
import os
from pathlib import Path
import joblib
import mlflow
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from scipy.stats import randint, uniform
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

THRESHOLD = 0.45  # fixed threshold; do not tune using test data

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='predictive_maintenance/data/processed')
    ap.add_argument('--models', default='predictive_maintenance/models')
    ap.add_argument('--reports', default='predictive_maintenance/reports')
    ap.add_argument('--tracking', default='run_output')
    ap.add_argument('--iterations', type=int, default=12)
    args = ap.parse_args()
    folder = Path(args.input)
    data = {name: pd.read_csv(folder / (name+'.csv')) for name in ('Xtrain','Xtest','ytrain','ytest')}
    Xtrain,Xtest = data['Xtrain'],data['Xtest']
    ytrain,ytest = data['ytrain'].iloc[:,0].astype(int),data['ytest'].iloc[:,0].astype(int)
    if Xtrain.columns.tolist() != Xtest.columns.tolist():
        raise ValueError('Feature order differs across training and test files')
    if not set(ytrain.unique()).issubset({0,1}) or not set(ytest.unique()).issubset({0,1}):
        raise ValueError('Target is not binary')
    if ytrain.value_counts().min() < 6:
        raise ValueError('Not enough minority class records for 3-fold SMOTE CV')
    models,reports,tracking = Path(args.models),Path(args.reports),Path(args.tracking)
    for d in [models,reports,tracking]: d.mkdir(parents=True,exist_ok=True)
    db = (tracking/'mlflow.db').resolve()
    mlflow.set_tracking_uri('sqlite:///'+db.as_posix())
    mlflow.set_experiment('vehicle_engine_failure_prediction_experiment')
    estimator = Pipeline([
       ('imputer',SimpleImputer(strategy='median')),
       ('scaler',RobustScaler()),
       ('smote',SMOTE(random_state=42,k_neighbors=5)),
       ('xgb',XGBClassifier(objective='binary:logistic',eval_metric='logloss',tree_method='hist',
                            random_state=42,n_jobs=2))
    ])
    distributions = {
       'xgb__n_estimators':randint(100,400),
       'xgb__max_depth':randint(2,7),
       'xgb__learning_rate':uniform(0.01,0.15),
       'xgb__min_child_weight':randint(2,9),
       'xgb__subsample':uniform(0.65,0.35),
       'xgb__colsample_bytree':uniform(0.65,0.35),
       'xgb__gamma':uniform(0,2),
       'xgb__reg_alpha':uniform(0,1),
       'xgb__reg_lambda':uniform(1,4),
    }
    print('Starting RandomizedSearchCV:',args.iterations,'candidates x 3 folds')
    with mlflow.start_run(run_name='xgboost_smote_cv'):
        search=RandomizedSearchCV(estimator,distributions,n_iter=args.iterations,cv=3,
                                  scoring='recall',n_jobs=2,random_state=42,verbose=1,
                                  error_score='raise')
        search.fit(Xtrain,ytrain)
        model=search.best_estimator_
        mlflow.log_params({k:float(v) if isinstance(v,np.floating) else int(v) if isinstance(v,np.integer) else v
                           for k,v in search.best_params_.items()})
        mlflow.log_metric('cv_best_recall',float(search.best_score_))
        mlflow.log_param('threshold',THRESHOLD)
        results={}
        for split,X,y in [('train',Xtrain,ytrain),('test',Xtest,ytest)]:
            yp=(model.predict_proba(X)[:,1]>=THRESHOLD).astype(int)
            results[split]={
              'accuracy':accuracy_score(y,yp), 'precision':precision_score(y,yp,zero_division=0),
              'recall':recall_score(y,yp,zero_division=0), 'f1':f1_score(y,yp,zero_division=0)
            }
            mlflow.log_metrics({f'{split}_{k}':float(v) for k,v in results[split].items()})
            (reports/f'{split}_classification_report.txt').write_text(classification_report(y,yp,zero_division=0))
            if split=='test':
                cm=confusion_matrix(y,yp,labels=[0,1])
                pd.DataFrame(cm,index=['actual_0','actual_1'],columns=['pred_0','pred_1']).to_csv(reports/'confusion_matrix.csv')
                fig,ax=plt.subplots(figsize=(5,4))
                ConfusionMatrixDisplay(cm,display_labels=[0,1]).plot(ax=ax,colorbar=False)
                fig.tight_layout();fig.savefig(reports/'confusion_matrix.png',dpi=150);plt.close(fig)
        path=models/'best_vehicle_engine_failure_prediction_model_v1.joblib'
        joblib.dump(model,path)
        config={'threshold':THRESHOLD,'feature_names':Xtrain.columns.tolist(),'positive_class':1}
        (models/'config.json').write_text(json.dumps(config,indent=2))
        (reports/'best_params.json').write_text(json.dumps(search.best_params_,indent=2,default=float))
        pd.DataFrame([{'split':split,**metrics} for split,metrics in results.items()]).to_csv(reports/'model_metrics.csv',index=False)
        for artifact in [path,models/'config.json',reports/'model_metrics.csv',reports/'confusion_matrix.png',reports/'best_params.json']:
            mlflow.log_artifact(str(artifact),artifact_path='capstone')
        print('Best parameters:',search.best_params_)
        print(pd.read_csv(reports/'model_metrics.csv').to_string(index=False))
        print('Model:',path.resolve())
        print('MLflow SQLite:',db)

if __name__=='__main__':
    main()
