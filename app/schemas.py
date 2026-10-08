"""Input- und Output-Vertrag der Inferenz-API.

Die Felder sind ein PLACEHOLDER und spiegeln den Iris-Datensatz.
Gültige Anfragen erfüllen dieses Schema; alles andere lehnt FastAPI
mit HTTP 422 ab, bevor ein Modell aufgerufen wird. Wenn der echte
Datensatz feststeht, wird der Vertrag hier ersetzt – zusammen mit
``FEATURE_COLUMNS`` in ``src/config.py``.
"""

from pydantic import BaseModel, ConfigDict, Field


class PredictRequest(BaseModel):
    """Eine Zeile Merkmale, für die eine Klasse vorhergesagt werden soll."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {
                    "sepal_length_cm": 5.1,
                    "sepal_width_cm": 3.5,
                    "petal_length_cm": 1.4,
                    "petal_width_cm": 0.2,
                }
            ]
        },
    )

    sepal_length_cm: float = Field(..., description="PLACEHOLDER: Kelchblattlänge in cm.")
    sepal_width_cm: float = Field(..., description="PLACEHOLDER: Kelchblattbreite in cm.")
    petal_length_cm: float = Field(..., description="PLACEHOLDER: Kronblattlänge in cm.")
    petal_width_cm: float = Field(..., description="PLACEHOLDER: Kronblattbreite in cm.")


class PredictResponse(BaseModel):
    """Antwort einer erfolgreichen Vorhersage."""

    prediction: str = Field(..., description="Vorhergesagte Klasse (Platzhalter: Iris-Art).")
    model_name: str = Field(..., description="Name des geladenen Modells.")


class HealthResponse(BaseModel):
    """Zustand des Prozesses, unabhängig davon, ob schon trainiert wurde."""

    status: str
    model_loaded: bool
