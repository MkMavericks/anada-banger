import numpy as np, pandas as pd
from hmmlearn.hmm import GaussianHMM

def detect_regime(prices: pd.DataFrame, n_states=3):
    r = np.log(prices["Close"]).diff().dropna()
    X = np.column_stack([r, r.rolling(10).std().bfill()])
    model = GaussianHMM(n_states, covariance_type="full", n_iter=200, random_state=42).fit(X)
    states = pd.Series(model.predict(X), index=r.index)
    order = np.argsort(model.means_[:, 0])          # sort by mean return
    labels = dict(zip(order, ["bear", "chop", "bull"]))
    return states.map(labels)