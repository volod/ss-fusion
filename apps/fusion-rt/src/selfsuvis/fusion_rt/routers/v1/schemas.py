"""Pydantic models for fusion-rt v1 API endpoints."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

VALID_MODALITIES = {"camera", "audio", "rf", "thermal", "vibration", "custom"}


class EventEnvelope(BaseModel):
    ts: datetime = Field(..., description="ISO 8601 event timestamp")
    zone_id: str = Field(..., min_length=1)
    sensor_id: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)
    payload: dict[str, Any] = Field(default_factory=dict)
    artifact_uri: str | None = None


class SiteEventResponse(BaseModel):
    event_id: str
    ts: str
    zone_id: str
    sensor_id: str
    modality: str
    confidence: float
    payload: dict[str, Any]
    artifact_uri: str | None
    created_at: str


class ZoneCreate(BaseModel):
    zone_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    description: str | None = None
    map_x: int | None = None
    map_y: int | None = None
    map_w: int | None = None
    map_h: int | None = None


class ZoneUpdate(BaseModel):
    label: str | None = None
    description: str | None = None
    map_x: int | None = None
    map_y: int | None = None
    map_w: int | None = None
    map_h: int | None = None


class ZoneResponse(BaseModel):
    zone_id: str
    label: str
    description: str | None
    map_x: int | None
    map_y: int | None
    map_w: int | None
    map_h: int | None
    created_at: str


class IncidentResponse(BaseModel):
    incident_id: str
    ts: str
    zone_id: str
    modalities: list[str]
    confidence: float
    risk_level: str
    summary_text: str | None
    evidence_refs: list[dict[str, Any]]
    rule_id: str | None
    acknowledged_at: str | None
    dismissed_at: str | None
    dismissal_reason: str | None
    created_at: str


class DismissBody(BaseModel):
    reason: str | None = None


class SiteStateZone(BaseModel):
    zone_id: str
    label: str
    risk_level: str | None
    active_incidents: list[IncidentResponse]


class SiteStateSnapshot(BaseModel):
    ts: str
    zones: list[SiteStateZone]


class NoteCreate(BaseModel):
    body: str = Field(..., min_length=1, max_length=4000)
    operator_id: str | None = None


class NoteResponse(BaseModel):
    note_id: str
    incident_id: str
    body: str
    operator_id: str | None
    created_at: str


class RuleCreate(BaseModel):
    rule_id: str = Field(..., min_length=1)
    label: str = Field(..., min_length=1)
    modalities: list[str] = Field(..., min_length=1)
    zone_id: str | None = None
    window_s: int = Field(default=30, ge=1, le=3600)
    min_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    enabled: bool = True

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("modalities must be non-empty")
        invalid = set(v) - VALID_MODALITIES
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}. Valid: {VALID_MODALITIES}")
        return v


class RuleUpdate(BaseModel):
    label: str | None = None
    modalities: list[str] | None = None
    zone_id: str | None = None
    window_s: int | None = Field(default=None, ge=1, le=3600)
    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    enabled: bool | None = None

    @field_validator("modalities")
    @classmethod
    def validate_modalities(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return v
        if not v:
            raise ValueError("modalities must be non-empty")
        invalid = set(v) - VALID_MODALITIES
        if invalid:
            raise ValueError(f"Invalid modalities: {invalid}")
        return v


class RuleResponse(BaseModel):
    rule_id: str
    label: str
    modalities: list[str]
    zone_id: str | None
    window_s: int
    min_confidence: float
    enabled: bool
    created_at: str
    updated_at: str
