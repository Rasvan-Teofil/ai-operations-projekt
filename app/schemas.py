"""HTTP input and output contracts; no inference or training imports."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, StrictStr, model_validator


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    text: StrictStr | None = None
    url: StrictStr | None = None

    @model_validator(mode='after')
    def exactly_one(self):
        if len(self.model_fields_set) != 1:
            raise ValueError('Provide exactly one of text or url; omit the other field')
        value = self.text if self.text is not None else self.url
        if not value or not value.strip() or len(value) > 20000:
            raise ValueError('Input must contain 1–20000 non-blank characters')
        if self.url is not None:
            from urllib.parse import urlsplit
            try:
                u = urlsplit(self.url)
                if (u.scheme not in ('http','https') or not u.hostname or u.username or u.password
                    or any(c.isspace() for c in self.url) or u.port not in (None,80,443)):
                    raise ValueError('Exactly one http(s) URL is required')
            except (ValueError, UnicodeError) as e:
                raise ValueError('Exactly one http(s) URL is required') from e
        return self


class Scores(BaseModel):
    negative: float
    neutral: float
    positive: float


class Prediction(BaseModel):
    label: Literal['negative','neutral','positive']
    scores: Scores
    model_version: str
    scraped_excerpt: str | None = None
    input_policy: str | None = None


class Highlight(BaseModel):
    start: int
    end: int
    text: str
    label: Literal['negative', 'neutral', 'positive']
    scores: Scores


class Analysis(Prediction):
    text: str
    sentences: list[Highlight]
    total_sentences: int
    highlighting_limited: bool
