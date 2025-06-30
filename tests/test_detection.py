"""
Test suite for the DDCal region detection tool.
Includes unit tests, integration tests, and synthetic data generation.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from astropy.io import fits
from astropy.wcs import WCS
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the main module (assuming it's saved as ddcal_detector.py)
from ddcal_detector import (
    DetectionParameters, RegionInfo, ResidualImageAnalyzer,
    write_ds9_region_file, write_casa_region_file
)

class TestSyntheticDataGenerator:
    """Generate synthetic residual images for testing."""

    @staticmethod
    def create_synthetic_residual_image(
        shape: tuple = (512, 512),
        noise_level: float = 1e-5,
        add_artifacts: bool = True
    ) -> tuple:
        """Create a synthetic residual image with known artifacts."""

        # Base noise
        np.random.seed(42)  # For reproducible tests
        image = np.random.normal(0, noise_level, shape)

        if add_artifacts:
            # Add negative spike (point source calibration error)
            y, x = shape[0] // 3, shape[1] // 3
            image[y-2:y+3, x-2:x+3] = -10 * noise_level

            # Add positive spike
            y, x = 2 * shape[0] // 3, 2 * shape[1] // 3
            image[y-1:y+2, x-1:x+2] = 8 * noise_level

            # Add ripple pattern (beam error)
            y_grid, x_grid = np.ogrid[:shape[0], :shape[1]]
            ripple = 2 * noise_level * np.sin(2 * np.pi * x_grid / 50) * \
                    np.exp(-((x_grid - shape[1]//2)**2 + (y_grid - shape[0]//2)**2) / (100**2))
            image += ripple

            # Add extended negative region (ionospheric error)
            y, x = shape[0] // 4, 3 * shape[1] // 4
            for dy in range(-15, 16):
                for dx in range(-15, 16):
                    if (dy**2 + dx**2) < 15**2:
                        if 0 <= y+dy < shape[0] and 0 <= x+dx < shape[1]:
                            image[y+dy, x+dx] = -3 * noise_level

        # Create simple WCS
        header = fits.Header()
        header['NAXIS'] = 2
        header['NAXIS1'] = shape[1]
        header['NAXIS2'] = shape[0]
        header['CTYPE1'] = 'RA---SIN'
        header['CTYPE2'] = 'DEC--SIN'
        header['CRVAL1'] = 0.0
        header['CRVAL2'] = -30.0
        header['CRPIX1'] = shape[1] // 2
        header['CRPIX2'] = shape[0] // 2
        header['CDELT1'] = -1.0 / 3600.0  # 1 arcsec/pixel
        header['CDELT2'] = 1.0 / 3600.0

        wcs = WCS(header)

        return image, wcs, header

class TestDetectionParameters:
    """Test the DetectionParameters dataclass."""

    def test_default_parameters(self):
        """Test default parameter values."""
        params = DetectionParameters()
        assert params.negative_threshold_sigma == 3.0
        assert params.spike_threshold_percentile == 95.0
        assert params.min_region_size_pixels == 50
        assert params.clustering_eps == 10.0

    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = DetectionParameters(
            negative_threshold_sigma=5.0,
            min_region_size_pixels=100
        )
        assert params.negative_threshold_sigma == 5.0
        assert params.min_region_size_pixels == 100

class TestRegionInfo:
    """Test the RegionInfo dataclass."""

    def test_region_creation(self):
        """Test creating a RegionInfo object."""
        stats = {'area_pixels': 100, 'eccentricity': 0.5}
        region = RegionInfo(
            center_ra=10.0,
            center_dec=-30.0,
            radius_arcsec=30.0,
            confidence=0.8,
            detection_type='negative',
            pixel_coords=(100, 200),
            stats=stats
        )

        assert region.center_ra == 10.0
        assert region.center_dec == -30.0
        assert region.detection_type == 'negative'
        assert region.stats['area_pixels'] == 100

class TestResidualImageAnalyzer:
    """Test the main analyzer class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = DetectionParameters(min_region_size_pixels=10)
        self.analyzer = ResidualImageAnalyzer(self.params)

        # Create synthetic data
        self.test_data, self.test_wcs, self.test_header = \
            TestSyntheticDataGenerator.create_synthetic_residual_image()

    def test_compute_image_statistics(self):
        """Test image statistics computation."""
        stats = self.analyzer.compute_image_statistics(self.test_data)

        assert 'mean' in stats
        assert 'std' in stats
        assert 'robust_sigma' in stats
        assert 'rms' in stats

        # Check that robust sigma is reasonable
        assert stats['robust_sigma'] > 0
        assert stats['rms'] > 0

    def test_detect_negative_outliers(self):
        """Test negative outlier detection."""
        data_jax = jnp.array(self.test_data)
        threshold_sigma = 3.0
        robust_sigma = 1e-5

        mask = self.analyzer.detect_negative_outliers(
            data_jax, threshold_sigma, robust_sigma
        )

        # Should detect some negative outliers in synthetic data
        assert np.any(mask)

        # Check that it's detecting the most negative values
        min_val = np.min(self.test_data)
        min_idx = np.unravel_index(np.argmin(self.test_data), self.test_data.shape)
        assert mask[min_idx]

    def test_detect_spikes_and_artifacts(self):
        """Test spike detection."""
        spike_mask = self.analyzer.detect_spikes_and_artifacts(self.test_data)

        # Should detect some spikes in synthetic data
        assert np.any(spike_mask)

        # Check mask is boolean
        assert spike_mask.dtype == bool

    def test_detect_ripple_patterns(self):
        """Test ripple pattern detection."""
        ripple_mask = self.analyzer.detect_ripple_patterns(self.test_data)

        # Should detect some ripples in synthetic data
        assert np.any(ripple_mask)

        # Check mask is boolean
        assert ripple_mask.dtype == bool

    def test_combine_detection_masks(self):
        """Test mask combination."""
        # Create simple test masks
        negative_mask = np.zeros((100, 100), dtype=bool)
        negative_mask[20:25, 20:25] = True

        spike_mask = np.zeros((100, 100), dtype=bool)
        spike_mask[30:35, 30:35] = True

        ripple_mask = np.zeros((100, 100), dtype=bool)
        ripple_mask[40:45, 40:45] = True

        combined = self.analyzer.combine_detection_masks(
            negative_mask, spike_mask, ripple_mask
        )

        # Should have detected regions
        assert np.any(combined)

        # Should be boolean
