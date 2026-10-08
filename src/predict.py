"""Generate the required prediction CSVs."""
import pandas as pd, joblib
ROLL='BT2024108'
for var in ['var1','var2']:
    test=pd.read_csv(f'data/{ROLL}_test_{var}.csv')
    model=joblib.load(f'models/model_{var}.joblib')
    features=[c for c in test.columns if c.startswith('x')]
    pred=model.predict(test[features])
    pd.DataFrame({'y':pred}).to_csv(f'{ROLL}_pred_{var}.csv',index=False)
    pd.DataFrame({'y':pred}).to_csv(f'results/{ROLL}_pred_{var}.csv',index=False)
    print(f'Saved {ROLL}_pred_{var}.csv')
