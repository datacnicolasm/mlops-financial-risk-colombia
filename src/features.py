import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

FEATURE_NAMES = [
    "patrimonio_negativo",
    "margen_operacional",
    "rotacion_activos",
    "endeudamiento",
    "liquidez_estructural"
]

class FinancialRatioTransformer(BaseEstimator, TransformerMixin):
    """
    Clase que toma las partidas contables y calcula los 5 indicadores
    """
    def __init__(self):
        self.feature_names = FEATURE_NAMES
    
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, dict):
            df = pd.DataFrame([X])
        elif isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise ValueError("La entrada debe ser un DataFrame o un diccionario.")

        out = pd.DataFrame(index=df.index)

        # Patrimonio negativo
        out["patrimonio_negativo"] = (df["esf_Equity"] <= 0).astype(int)

        # Margen operacional [-1.0, 1.0]
        out["margen_operacional"] = np.where(
            df["eri_Revenue"] > 1000,
            df["eri_GananciaPerdidaPorActividadesDeOperacion"] / df["eri_Revenue"],
            0.0
        ).clip(-1.0, 1.0)

        # Rotación de activos [0.0, 10.0]
        out["rotacion_activos"] = np.where(
            df["esf_Assets"] > 0,
            df["eri_Revenue"] / df["esf_Assets"],
            0.0
        ).clip(0.0, 10.0)

        # Endeudamiento [0.0, 5.0]
        out["endeudamiento"] = np.where(
            df["esf_Assets"] > 0,
            df["esf_Liabilities"] / df["esf_Assets"],
            0.0
        ).clip(0.0, 5.0)

        # Liquidez estructural [-2.0, 1.0]
        out["liquidez_estructural"] = np.where(
            df["esf_Assets"] > 0,
            (df["esf_CurrentAssets"] - df["esf_CurrentLiabilities"]) / df["esf_Assets"],
            0.0
        ).clip(-2.0, 1.0)

        return out[self.feature_names]