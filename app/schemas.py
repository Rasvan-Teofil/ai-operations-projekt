"""Input and output contract of the inference API.

Exactly one of ``text`` or ``url``. The response is always the
three-class sentiment plus the distribution and the model version.
"""

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, model_validator


class PredictRequest(BaseModel):
    """One article as raw text or as a single URL."""

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
        description="Article text. Exactly one of text or url.",
    )
    url: AnyHttpUrl | None = Field(
        default=None,
        description="URL of exactly one article. Exactly one of text or url.",
    )

    @model_validator(mode="after")
    def exactly_one_source(self):
        has_text = self.text is not None
        has_url = self.url is not None
        if has_text == has_url:
            raise ValueError("Provide exactly one of text or url.")
        if has_text and not self.text.strip():
            raise ValueError("text must not be empty.")
        return self


class PredictResponse(BaseModel):
    """Prediction in the shared three-class space."""

    label: str = Field(..., description="negative, neutral or positive.")
    scores: dict[str, float] = Field(..., description="Distribution over the three classes.")
    model_version: str = Field(..., description="Version or checkpoint of the loaded model.")


class HealthResponse(BaseModel):
    """Process state. model_loaded is false while the artefact is missing."""

    status: str
    model_loaded: bool
    model_name: str | None = None
    model_version: str | None = None
