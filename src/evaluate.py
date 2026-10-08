"""Exhaustive model comparison for BT2024108.
Compares Polynomial + Linear, Polynomial + Ridge and Polynomial + Lasso.
Uses 5-fold shuffled CV (random_state=42), with scaling inside the pipeline.
"""
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, cross_val_score

ROLL='BT2024108'
ALPHAS=[0.01,0.1,1,10]
CV=KFold(n_splits=5,shuffle=True,random_state=42)

def run(var,max_degree,features):
    train=pd.read_csv(f'data/{ROLL}_train_{var}.csv')
    X,y=train[features],train['y']
    rows=[]
    for degree in range(1,max_degree+1):
        for method in ['Linear','Ridge','Lasso']:
            alphas=[None] if method=='Linear' else ALPHAS
            for alpha in alphas:
                if method=='Linear': model=LinearRegression()
                elif method=='Ridge': model=Ridge(alpha=alpha)
                else: model=Lasso(alpha=alpha,max_iter=50000,tol=1e-4)
                pipe=Pipeline([('poly',PolynomialFeatures(degree,include_bias=False)),('scaler',StandardScaler()),('model',model)])
                mse=-cross_val_score(pipe,X,y,cv=CV,scoring='neg_mean_squared_error').mean()
                r2=cross_val_score(pipe,X,y,cv=CV,scoring='r2').mean()
                rows.append([var,method,degree,alpha,mse,r2])
    return pd.DataFrame(rows,columns=['dataset','method','degree','alpha','cv_mse','cv_r2'])

r1=run('var1',10,[f'x{i}' for i in range(1,7)])
r2=run('var2',20,['x1','x2','x3'])
res=pd.concat([r1,r2],ignore_index=True)
res.to_csv('results/all_model_comparison.csv',index=False)
best=res.loc[res.groupby('dataset').cv_mse.idxmin()]
best.to_csv('results/best_overall.csv',index=False)
print(best.to_string(index=False))
