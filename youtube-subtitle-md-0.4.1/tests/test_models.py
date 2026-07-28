import pytest
from pydantic import ValidationError

from submd.models import CloudOcrConfig, Roi


def test_roi_parse() -> None:
    roi = Roi.parse("0.05, 0.60, 0.90, 0.30")
    assert roi.model_dump() == {"x": 0.05, "y": 0.6, "width": 0.9, "height": 0.3}


@pytest.mark.parametrize("value", ["0,0,1", "a,0,1,1", "0.5,0,0.6,1"])
def test_roi_rejects_invalid_value(value: str) -> None:
    with pytest.raises((ValueError, ValidationError)):
        Roi.parse(value)


def test_cloud_config_normalizes_base_url() -> None:
    config = CloudOcrConfig(base_url="https://vendor.example/v1/", model="vision-ocr")
    assert config.base_url == "https://vendor.example/v1"


def test_cloud_config_rejects_non_http_url() -> None:
    with pytest.raises(ValidationError):
        CloudOcrConfig(base_url="vendor.example/v1", model="vision-ocr")
