import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from pathsnap.optimisation.bayes_search import BayesianSearchConfig, optimise_model

def test_optuna_runs_grouped_cv_synthetically(tmp_path):
    rng=np.random.default_rng(3)
    groups=np.repeat([f'{k:08d}' for k in range(1,7)],30)
    observation=np.tile(np.repeat(np.arange(10),3),6)
    c=np.tile(np.arange(3),60)
    y=(c==0).astype(int)
    x=pd.DataFrame({'dist_m':np.where(y==1,2.0,30.0)+rng.normal(size=len(y)),
                    'heading_diff':np.where(y==1,5.0,50.0)+rng.normal(size=len(y))})
    def factory(p):
        return make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression(max_iter=300,random_state=42,**p))
    cfg=BayesianSearchConfig(model_name='logistic_regression',n_trials=2,n_splits=3,seed=42,
      objective='candidate_table_top1',cv_strategy='group_kfold',search_space={'C':{'type':'float','low':0.1,'high':5.0}})
    r=optimise_model(factory,x,y,groups,observation,config=cfg,output_dir=tmp_path)
    assert len(r.trials)==2
    assert np.isfinite(r.best_value)
    assert (tmp_path/'tuned_model.joblib').exists()
    assert (tmp_path/'fold_results.csv').exists()
    assert (tmp_path/'tuned_vs_baseline_cv.csv').exists()
