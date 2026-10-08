"""Train final selected models on all training data."""
import pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Lasso, Ridge

ROLL='BT2024108'
CONFIGS={
 'var1':(['x1','x2','x3','x4','x5','x6'],'Lasso',5,0.01),
 'var2':(['x1','x2','x3'],'Ridge',11,1.0),
}
for var,(features,method,degree,alpha) in CONFIGS.items():
    df=pd.read_csv(f'data/{ROLL}_train_{var}.csv')
    model=Lasso(alpha=alpha,max_iter=50000,tol=1e-4) if method=='Lasso' else Ridge(alpha=alpha)
    pipe=Pipeline([('poly',PolynomialFeatures(degree,include_bias=False)),('scaler',StandardScaler()),('model',model)])
    pipe.fit(df[features],df.y)
    joblib.dump(pipe,f'models/model_{var}.joblib')
    print(f'{var}: {method}, degree={degree}, alpha={alpha}')
