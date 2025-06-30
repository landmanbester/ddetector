"""
Core data structures and classes for DDetector.

This module contains the fundamental data structures used throughout
the DDetector package for direction-dependent calibration detection.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


@dataclass
class DetectionParameters:
    """Parameters for direction-dependent calibration region detection.

    This class encapsulates all the configurable parameters used by the
    detection algorithms to identify regions requiring DDCal.

    Attributes:
        negative_threshold_sigma: Threshold for negative outliers in sigma units.
            Higher values = less sensitive to negative spikes.
        spike_threshold_percentile: Percentile threshold for spike detection.
            Higher values = only detect very prominent spikes.
        ripple_frequency_range: Tuple of (min_freq, max_freq) for spatial
            frequency analysis of ripple patterns.
        min_region_size_pixels: Minimum region size in pixels to be considered
            a valid detection.
        clustering_eps: DBSCAN clustering epsilon parameter in pixels.
            Larger values merge more distant detections.
        clustering_min_samples: DBSCAN minimum samples parameter.
        morphology_disk_size: Size of morphological operations disk.
        edge_buffer_pixels: Buffer distance from image edges in pixels.
    """

    # Detection sensitivity parameters
    negative_threshold_sigma: float = 3.0
    spike_threshold_percentile: float = 95.0
    ripple_frequency_range: Tuple[float, float] = (0.1, 2.0)

    # Region filtering parameters
    min_region_size_pixels: int = 50
    edge_buffer_pixels: int = 20

    # Clustering parameters
    clustering_eps: float = 10.0
    clustering_min_samples: int = 5

    # Morphological processing parameters
    morphology_disk_size: int = 3

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.negative_threshold_sigma <= 0:
            raise ValueError("negative_threshold_sigma must be positive")

        if not (0 < self.spike_threshold_percentile < 100):
            raise ValueError("spike_threshold_percentile must be between 0 and 100")

        if self.min_region_size_pixels <= 0:
            raise ValueError("min_region_size_pixels must be positive")

        if len(self.ripple_frequency_range) != 2:
            raise ValueError("ripple_frequency_range must be a tuple of (min, max)")

        min_freq, max_freq = self.ripple_frequency_range
        if min_freq >= max_freq or min_freq <= 0:
            raise ValueError("Invalid ripple_frequency_range: must have 0 < min < max")

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary for serialization."""
        return {
            'negative_threshold_sigma': self.negative_threshold_sigma,
            'spike_threshold_percentile': self.spike_threshold_percentile,
            'ripple_frequency_range': self.ripple_frequency_range,
            'min_region_size_pixels': self.min_region_size_pixels,
            'clustering_eps': self.clustering_eps,
            'clustering_min_samples': self.clustering_min_samples,
            'morphology_disk_size': self.morphology_disk_size,
            'edge_buffer_pixels': self.edge_buffer_pixels,
        }

    @classmethod
    def from_dict(cls, params_dict: Dict[str, Any]) -> 'DetectionParameters':
        """Create DetectionParameters from dictionary."""
        return cls(**params_dict)

    def get_frequency_band_preset(self, band: str) -> 'DetectionParameters':
        """Get parameter preset optimized for specific frequency bands.

        Args:
            band: Frequency band name ('lband', 'cband', 'xband', etc.)

        Returns:
            New DetectionParameters instance with band-specific settings.
        """
        presets = {
            'lband': {  # 1-2 GHz
                'negative_threshold_sigma': 3.0,
                'spike_threshold_percentile': 95.0,
                'ripple_frequency_range': (0.1, 1.5),
                'min_region_size_pixels': 50,
            },
            'cband': {  # 4-8 GHz
                'negative_threshold_sigma': 3.5,
                'spike_threshold_percentile': 96.0,
                'ripple_frequency_range': (0.15, 2.0),
                'min_region_size_pixels': 30,
            },
            'xband': {  # 8-12 GHz
                'negative_threshold_sigma': 4.0,
                'spike_threshold_percentile': 97.0,
                'ripple_frequency_range': (0.2, 2.5),
                'min_region_size_pixels': 25,
            },
            'wide_field': {  # Wide field observations
                'negative_threshold_sigma': 2.5,
                'ripple_frequency_range': (0.05, 1.0),
                'min_region_size_pixels': 100,
                'clustering_eps': 25.0,
                'edge_buffer_pixels': 50,
            },
            'conservative': {  # Conservative detection
                'negative_threshold_sigma': 4.0,
                'spike_threshold_percentile': 98.0,
                'min_region_size_pixels': 80,
                'clustering_eps': 20.0,
            },
            'aggressive': {  # Aggressive detection
                'negative_threshold_sigma': 2.0,
                'spike_threshold_percentile': 90.0,
                'min_region_size_pixels': 20,
                'clustering_eps': 6.0,
            }
        }

        if band not in presets:
            raise ValueError(f"Unknown band '{band}'. Available: {list(presets.keys())}")

        # Start with current parameters and override with preset values
        params_dict = self.to_dict()
        params_dict.update(presets[band])

        return self.__class__.from_dict(params_dict)


@dataclass
class RegionInfo:
    """Information about a detected calibration region.

    This class encapsulates all information about a region that has been
    identified as requiring direction-dependent calibration.

    Attributes:
        center_ra: Right ascension of region center in degrees.
        center_dec: Declination of region center in degrees.
        radius_arcsec: Region radius in arcseconds.
        confidence: Detection confidence score (0-1).
        detection_type: Type of artifact detected ('negative', 'spike', 'ripple', etc.).
        pixel_coords: Center coordinates in pixel space (x, y).
        stats: Dictionary containing region statistics and properties.
    """

    # Position and size
    center_ra: float
    center_dec: float
    radius_arcsec: float

    # Detection metadata
    confidence: float
    detection_type: str
    pixel_coords: Tuple[int, int]

    # Additional properties
    stats: Dict[str, float]

    def __post_init__(self):
        """Validate region information after initialization."""
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0 and 1")

        if self.radius_arcsec <= 0:
            raise ValueError("radius_arcsec must be positive")

        if not (-360 <= self.center_ra <= 360):
            raise ValueError("center_ra must be valid RA coordinate")

        if not (-90 <= self.center_dec <= 90):
            raise ValueError("center_dec must be valid Dec coordinate")

    @property
    def area_arcsec2(self) -> float:
        """Calculate region area in square arcseconds."""
        return np.pi * self.radius_arcsec**2

    @property
    def area_sq_degrees(self) -> float:
        """Calculate region area in square degrees."""
        return self.area_arcsec2 / (3600.0**2)

    def get_detection_types(self) -> list:
        """Get list of individual detection types."""
        return self.detection_type.split('+')

    def has_detection_type(self, detection_type: str) -> bool:
        """Check if region contains a specific detection type."""
        return detection_type in self.get_detection_types()

    def to_dict(self) -> Dict[str, Any]:
        """Convert region to dictionary for serialization."""
        return {
            'center_ra': self.center_ra,
            'center_dec': self.center_dec,
            'radius_arcsec': self.radius_arcsec,
            'confidence': self.confidence,
            'detection_type': self.detection_type,
            'pixel_coords': self.pixel_coords,
            'stats': self.stats,
            'area_arcsec2': self.area_arcsec2,
            'area_sq_degrees': self.area_sq_degrees,
        }

    @classmethod
    def from_dict(cls, region_dict: Dict[str, Any]) -> 'RegionInfo':
        """Create RegionInfo from dictionary."""
        # Remove computed properties if present
        clean_dict = {k: v for k, v in region_dict.items()
                     if k not in ['area_arcsec2', 'area_sq_degrees']}
        return cls(**clean_dict)


@dataclass
class ImageStatistics:
    """Statistical properties of a residual image.

    This class contains robust statistical measures computed from
    the residual image for use in detection algorithms.

    Attributes:
        mean: Mean pixel value.
        std: Standard deviation of pixel values.
        median: Median pixel value.
        mad: Median absolute deviation.
        robust_sigma: Robust sigma estimate (1.4826 * MAD).
        rms: Root mean square value.
        dynamic_range: Ratio of peak to noise.
        percentiles: Dictionary of percentile values.
    """

    mean: float
    std: float
    median: float
    mad: float
    robust_sigma: float
    rms: float
    min_value: float
    max_value: float

    # Additional percentiles for robust analysis
    p1: float = 0.0
    p5: float = 0.0
    p95: float = 0.0
    p99: float = 0.0

    @property
    def dynamic_range(self) -> float:
        """Calculate dynamic range as peak/noise ratio."""
        if self.robust_sigma > 0:
            return abs(self.max_value) / self.robust_sigma
        return 0.0

    @property
    def noise_level(self) -> float:
        """Get the noise level (robust sigma estimate)."""
        return self.robust_sigma

    def get_percentiles(self) -> Dict[str, float]:
        """Get dictionary of percentile values."""
        return {
            'p1': self.p1,
            'p5': self.p5,
            'p95': self.p95,
            'p99': self.p99,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'mean': self.mean,
            'std': self.std,
            'median': self.median,
            'mad': self.mad,
            'robust_sigma': self.robust_sigma,
            'rms': self.rms,
            'min_value': self.min_value,
            'max_value': self.max_value,
            'dynamic_range': self.dynamic_range,
            'percentiles': self.get_percentiles(),
        }


class DetectionResult:
    """Container for detection analysis results.

    This class holds the complete results of a detection analysis,
    including the detected regions, image statistics, and metadata.
    """

    def __init__(self,
                 regions: list[RegionInfo],
                 image_stats: ImageStatistics,
                 detection_params: DetectionParameters,
                 input_file: Optional[str] = None):
        """Initialize detection result.

        Args:
            regions: List of detected regions.
            image_stats: Image statistics.
            detection_params: Parameters used for detection.
            input_file: Path to input FITS file.
        """
        self.regions = regions
        self.image_stats = image_stats
        self.detection_params = detection_params
        self.input_file = input_file
        self._analysis_metadata = {}

    @property
    def num_regions(self) -> int:
        """Number of detected regions."""
        return len(self.regions)

    @property
    def total_affected_area(self) -> float:
        """Total area of all detected regions in square arcseconds."""
        return sum(region.area_arcsec2 for region in self.regions)

    def get_regions_by_type(self, detection_type: str) -> list[RegionInfo]:
        """Get regions containing a specific detection type."""
        return [r for r in self.regions if r.has_detection_type(detection_type)]

    def get_high_confidence_regions(self, threshold: float = 0.7) -> list[RegionInfo]:
        """Get regions with confidence above threshold."""
        return [r for r in self.regions if r.confidence >= threshold]

    def set_metadata(self, key: str, value: Any):
        """Set analysis metadata."""
        self._analysis_metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get analysis metadata."""
        return self._analysis_metadata.get(key, default)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Create summary dictionary for reporting."""
        return {
            'input_file': self.input_file,
            'num_regions': self.num_regions,
            'total_affected_area_arcsec2': self.total_affected_area,
            'high_confidence_regions': len(self.get_high_confidence_regions()),
            'detection_types': {
                'negative': len(self.get_regions_by_type('negative')),
                'spike': len(self.get_regions_by_type('spike')),
                'ripple': len(self.get_regions_by_type('ripple')),
            },
            'image_stats': self.image_stats,
            'detection_params': self.detection_params.to_dict(),
            'metadata': self._analysis_metadata,
        }


# Version information
__version__ = "0.1.0"
__author__ = "Landman Bester"
__email__ = "lbester@sarao.ac.za"

# Export main classes
__all__ = [
    'DetectionParameters',
    'RegionInfo',
    'ImageStatistics',
    'DetectionResult',
]
