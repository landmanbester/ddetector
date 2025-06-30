"""
DDetector: Automated Direction-Dependent Calibration Region Detection

A tool for detecting regions requiring direction-dependent calibration
in radio interferometric residual images.
"""

__version__ = "0.1.0"
__author__ = "Landman Bester"
__email__ = "lbester@sarao.ac.za"

from .core import DetectionParameters, RegionInfo
from .detection import ResidualImageAnalyzer
from .io_utils import write_ds9_region_file, write_casa_region_file
from .visualization import DetectionVisualizer, AdvancedAnalyzer

__all__ = [
    "DetectionParameters",
    "RegionInfo",
    "ResidualImageAnalyzer",
    "write_ds9_region_file",
    "write_casa_region_file",
    "DetectionVisualizer",
    "AdvancedAnalyzer"
]
