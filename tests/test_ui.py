import pytest
from utils import get_risk_color

def test_risk_color_thresholds():
    """Test that risk colors match expected probability zones."""
    assert get_risk_color(0.2) == "green"
    assert get_risk_color(0.5) == "orange"
    assert get_risk_color(0.8) == "red"

def test_risk_color_boundary():
    """Test boundary conditions for risk color."""
    assert get_risk_color(0.4) == "orange"
    assert get_risk_color(0.7) == "red"
