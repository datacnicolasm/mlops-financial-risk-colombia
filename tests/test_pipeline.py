import pytest
import pandas as pd
from fastapi.testclient import TestClient
from src.features import FinancialRatioTransformer
from app.main import app

client = TestClient(app)

def test_financial_ratio_transformer():
    raw_data = pd.DataFrame([{
        "esf_Assets": 1000000.0,
        "esf_Liabilities": 400000.0,
        "esf_Equity": 600000.0,
        "esf_CurrentAssets": 500000.0,
        "esf_CurrentLiabilities": 200000.0,
        "eri_Revenue": 1200000.0,
        "eri_GananciaPerdidaPorActividadesDeOperacion": 150000.0
    }])

    transformer = FinancialRatioTransformer()
    features = transformer.transform(raw_data)

    assert features.shape == (1, 5)
    assert features["patrimonio_negativo"].iloc[0] == 0
    assert pytest.approx(features["endeudameinto"].iloc[0], 0.01) == 0.4
    assert pytest.approx(features["rotacion_activos"].iloc[0], 0.01) == 1.2

def test_health_endpoint():
    response = client.get("helath")
    assert response.status_code == 200
    assert "status" in response.json()