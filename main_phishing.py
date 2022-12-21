import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


def make_dataset(path: str, n_cols: int=10) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, index_col='id')

    X = df[top_correleted_columns(df.corr(), "CLASS_LABEL", n_cols)].drop('CLASS_LABEL', axis=1)
    y = df.CLASS_LABEL
    
    return X, y
    

def top_correleted_columns(corr: pd.DataFrame, target: str, top_n:int = 5) -> np.array:
    '''Restituisce un array di nomi di colonne che sono più correlate con la colonna di `target`

    :param pd.DataFrame corr: correlazione del dataframe, consiglio di passare data.corr()
    :param str target: nome della colonna di `target`, la y 
    :param int top_n: numero di nomi colonne darestituire, defaults to 5
    '''    
    sorted_corr = corr.loc[target].abs().sort_values(ascending=False)
    return sorted_corr[:top_n+1].index.values


if __name__ == "__main__":
    X, y = make_dataset('.\data\Phishing_Legitimate_full.csv', n_cols=10)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.33)
    
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    
    pred = clf.predict(X_test)
    
    print(classification_report(y_test, pred))