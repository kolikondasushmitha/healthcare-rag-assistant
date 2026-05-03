from pydantic import BaseModel, Field

class DiabetesInput(BaseModel):
    Pregnancies: float              = Field(..., ge=0,  le=20,  example=2)
    Glucose: float                  = Field(..., ge=0,  le=300, example=120)
    BloodPressure: float            = Field(..., ge=0,  le=200, example=72)
    SkinThickness: float            = Field(..., ge=0,  le=100, example=23)
    Insulin: float                  = Field(..., ge=0,  le=900, example=85)
    BMI: float                      = Field(..., ge=0,  le=80,  example=28.5)
    DiabetesPedigreeFunction: float = Field(..., ge=0,  le=3,   example=0.45)
    Age: int                        = Field(..., ge=21, le=120, example=33)

class PredictionResponse(BaseModel):
    prediction:  int
    probability: float
    risk_level:  str
    label:       str
    top_factors: list[str]
    explanation: str
    disclaimer:  str