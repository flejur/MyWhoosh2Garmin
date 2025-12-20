import json
from datetime import datetime
from pathlib import Path
from typing import List

from fit_tool.fit_file_builder import FitFileBuilder
from fit_tool.profile.messages.activity_message import ActivityMessage
from fit_tool.profile.messages.device_info_message import DeviceInfoMessage  # ✅ NEU
from fit_tool.profile.messages.event_message import EventMessage
from fit_tool.profile.messages.file_creator_message import FileCreatorMessage
from fit_tool.profile.messages.file_id_message import FileIdMessage
from fit_tool.profile.messages.lap_message import LapMessage
from fit_tool.profile.messages.record_message import RecordMessage
from fit_tool.profile.messages.session_message import SessionMessage
from fit_tool.profile.profile_type import (
    Event,
    EventType,
    FileType,
    GarminProduct,
    Intensity,
    Manufacturer,
    Sport,
    SubSport,
)
from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator


class ActivityData(BaseModel):
    """Model for MyWhoosh activity JSON data."""

    model_config = ConfigDict(
        extra="forbid",  # Forbid extra fields
        validate_assignment=True,  # Validate on attribute assignment
        str_to_lower=True,  # Convert strings to lowercase
        strict=True,  # Enforce strict type checking
    )

    # From activity metadata api
    name: str = Field(default_factory=str, alias="strava_activity_name")
    id: int = Field(..., alias="strava_activity_id")
    activity_distance: float
    moving_time: int
    elapsed_time: int
    total_elevation_gain: float
    type: str
    start_date: datetime | int
    start_date_local: datetime | int
    timezone: str
    utc_offset: float
    average_speed: float
    max_speed: float
    average_cadence: float
    average_watts: float
    max_watts: int
    weighted_average_watts: int
    kilojoules: float
    average_heartrate: float
    max_heartrate: float
    calories: float

    # From streams
    lat: List[float]
    long: List[float]
    watts: List[int]
    cadence: List[int]
    velocity_smooth: List[float]
    heartrate: List[int]
    time: List[int]
    heartrates: List[int]
    distance: List[float]
    grade_smooth: List[float] | None
    altitude: List[float] | None

    @model_validator(mode="after")
    def validate_streams(self) -> "ActivityData":
        """Validate that all stream lists have the same length and records exist."""
        # Collect all stream attributes that are lists
        stream_attrs = [
            "lat",
            "long",
            "watts",
            "cadence",
            "velocity_smooth",
            "heartrate",
            "time",
            "heartrates",
            "distance",
        ]
        # Optionally include nullable streams if present
        if self.grade_smooth is not None:
            stream_attrs.append("grade_smooth")
        if self.altitude is not None:
            stream_attrs.append("altitude")

        lengths = [
            len(getattr(self, attr))
            for attr in stream_attrs
            if getattr(self, attr) is not None
        ]
        if lengths and any(length != lengths[0] for length in lengths):
            raise ValueError("All stream lists must have the same length.")

        return self

    @property
    def stream_length(self) -> int:
        return len(self.time)

    @property
    def elapsed_time(self) -> int:
        """Get elapsed time in milliseconds."""
        return self.elapsed_time * 1000

    @classmethod
    def from_json_file(cls, json_file_path: str) -> "ActivityData":
        """Load and parse the JSON activity file into the model."""
        with open(json_file_path, "r") as f:
            raw_data = json.load(f)

        # Extract metadata and streams
        metadata = raw_data.get("metadata", {})
        streams = raw_data.get("streams", {})

        # Combine metadata fields with stream data
        combined_data = {
            # Metadata fields (activity summary)
            "strava_activity_name": metadata.get("name", ""),
            "strava_activity_id": metadata.get("id"),
            "activity_distance": metadata.get("distance"),
            "moving_time": metadata.get("moving_time"),
            "elapsed_time": metadata.get("elapsed_time"),
            "total_elevation_gain": metadata.get("total_elevation_gain"),
            "type": metadata.get("type"),
            "start_date": datetime.fromisoformat(metadata.get("start_date"))
            if metadata.get("start_date")
            else None,
            "start_date_local": datetime.fromisoformat(metadata.get("start_date_local"))
            if metadata.get("start_date_local")
            else None,
            "timezone": metadata.get("timezone"),
            "utc_offset": metadata.get("utc_offset"),
            "average_speed": metadata.get("average_speed"),
            "max_speed": metadata.get("max_speed"),
            "average_cadence": metadata.get("average_cadence"),
            "average_watts": metadata.get("average_watts"),
            "max_watts": metadata.get("max_watts"),
            "weighted_average_watts": metadata.get("weighted_average_watts"),
            "kilojoules": metadata.get("kilojoules"),
            "average_heartrate": metadata.get("average_heartrate"),
            "max_heartrate": metadata.get("max_heartrate"),
            "calories": metadata.get("calories"),
            # Stream data (time series)
            # Extract and separate lat/long from latlng pairs
            "lat": [],
            "long": [],
        }

        # Stream data (time series)
        # Extract and separate lat/long from latlng pairs
        latlng_data = streams.get("latlng", {}).get("data", [])
        if latlng_data:
            lat_values, long_values = zip(*latlng_data)
            combined_data["lat"] = list(lat_values)
            combined_data["long"] = list(long_values)
        else:
            combined_data
