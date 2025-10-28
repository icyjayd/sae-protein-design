import os, pandas as pd

def load_cache(path:str)->pd.DataFrame:
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()

def update_cache(cache_path:str, new_df:pd.DataFrame):
    if os.path.exists(cache_path):
        df=pd.read_csv(cache_path)
        full=pd.concat([df,new_df],ignore_index=True)
    else:
        full=new_df
    full.to_csv(cache_path,index=False)
    return len(new_df)
