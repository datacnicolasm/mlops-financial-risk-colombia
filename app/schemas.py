from pydantic import BaseModel, Field

class CompanyFinancialPayload(BaseModel):
    nit: str = Field(..., description="NIT de la empresa", example="900123456")
    esf_Assets: float = Field(..., gt=0, description="Total Activos (debe ser mayor a 0)", example=1500000000.0)
    esf_Liabilities: float = Field(..., ge=0, description="Total Pasivos", example=800000000.0)
    esf_Equity: float = Field(..., description="Patrimonio Total", example=700000000.0)
    esf_CurrentAssets: float = Field(..., ge=0, description="Activo Corriente", example=600000000.0)
    esf_CurrentLiabilities: float = Field(..., ge=0, description="Pasivo Corriente", example=400000000.0)
    eri_Revenue: float = Field(..., ge=0, description="Ingresos Operacionales", example=2000000000.0)
    eri_GananciaPerdidaPorActividadesDeOperacion: float = Field(..., description="Utilidad Operativa", example=180000000.0)

class PredictionResponse(BaseModel):
    nit: str
    probabilidad_estres: float
    clasificacion_riesgo: int
    nivel_alerta: str
    model_version: str