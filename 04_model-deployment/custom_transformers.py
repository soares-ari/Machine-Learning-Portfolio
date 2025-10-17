from sklearn.base import BaseEstimator, TransformerMixin

class CleanBrandTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        X_ = X.copy()
        X_['brand'] = (
            X_['brand']
            .astype(str)
            .str.strip()
            .str.replace('.', '', regex=False)
            .str.lower()
        )
        return X_
