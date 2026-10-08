"""Input- und Output-Vertrag der Inferenz-API.

Genau eines von ``text`` oder ``url``. Die Antwort ist immer das
Drei-Klassen-Sentiment plus die Verteilung und die Modellversion.
"""

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class PredictRequest(BaseModel):
    """Ein Artikel als Rohtext oder als einzelne URL."""

    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "examples": [
                {"text": "The clinic reported fewer infections after the new treatment."},
                {"url": "https://example.com/news/library"},
            ]
        },
    )

    text: str | None = Field(
        default=None,
        max_length=20_000,
        description="Artikeltext. Genau eines von text oder url.",
    )
    url: AnyHttpUrl | None = Field(
        default=None,
        description="URL genau eines Artikels. Genau eines von text oder url.",
    )

    @model_validator(mode="after")
    def exactly_one_source(self):
        has_text = self.text is not None
        has_url = self.url is not None
        if has_text == has_url:
            raise ValueError("Genau eines von text oder url angeben.")
        if has_text and not self.text.strip():
            raise ValueError("text darf nicht leer sein.")
        return self


class PredictResponse(BaseModel):
    """Vorhersage im gemeinsamen Drei-Klassen-Raum."""

    label: str = Field(..., description="negative, neutral oder positive.")
    scores: dict[str, float] = Field(..., description="Verteilung über die drei Klassen.")
    model_version: str = Field(..., description="Version oder Checkpoint des geladenen Modells.")


class HealthResponse(BaseModel):
    """Prozesszustand. model_loaded ist falsch, solange das Artefakt fehlt."""

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
