"""
Input/Output utilities for DDetector.

This module handles reading FITS files, writing region files in various formats,
and managing configuration files for the detection pipeline.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Union, Optional, Dict, Any
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from .core import RegionInfo, DetectionParameters, DetectionResult, ImageStatistics

logger = logging.getLogger(__name__)


class FitsImageLoader:
    """Utility class for loading and processing FITS images."""

    @staticmethod
    def load_fits_image(fits_path: Union[str, Path]) -> Tuple[np.ndarray, WCS, fits.Header]:
        """Load FITS image and extract data, WCS, and header information.

        Args:
            fits_path: Path to FITS file.

        Returns:
            Tuple of (image_data, wcs, header).

        Raises:
            FileNotFoundError: If FITS file doesn't exist.
            ValueError: If FITS file format is unsupported.
        """
        fits_path = Path(fits_path)
        if not fits_path.exists():
            raise FileNotFoundError(f"FITS file not found: {fits_path}")

        logger.info(f"Loading FITS image: {fits_path}")

        try:
            with fits.open(fits_path) as hdul:
                # Get primary HDU data and header
                primary_hdu = hdul[0]
                data = primary_hdu.data
                header = primary_hdu.header.copy()

                if data is None:
                    raise ValueError("No data found in primary HDU")

                # Handle different FITS formats (2D, 3D, 4D)
                original_shape = data.shape
                data = FitsImageLoader._extract_2d_image(data)

                logger.info(f"Original shape: {original_shape}, extracted shape: {data.shape}")

                # Clean data (remove NaN/inf values)
                data = FitsImageLoader._clean_image_data(data)

                # Extract WCS information
                wcs = FitsImageLoader._extract_wcs(header)

                return data, wcs, header

        except Exception as e:
            logger.error(f"Error loading FITS file {fits_path}: {e}")
            raise

    @staticmethod
    def _extract_2d_image(data: np.ndarray) -> np.ndarray:
        """Extract 2D image from potentially higher-dimensional FITS data.

        Args:
            data: Input data array of arbitrary dimensions.

        Returns:
            2D image array.
        """
        if data.ndim == 2:
            return data
        elif data.ndim == 3:
            # Take first plane (usually frequency or Stokes)
            logger.info("3D FITS detected, taking first plane")
            return data[0]
        elif data.ndim == 4:
            # Take first frequency and first Stokes parameter
            logger.info("4D FITS detected, taking [0,0] plane")
            return data[0, 0]
        else:
            raise ValueError(f"Unsupported FITS dimensions: {data.ndim}D")

    @staticmethod
    def _clean_image_data(data: np.ndarray) -> np.ndarray:
        """Clean image data by handling NaN and infinite values.

        Args:
            data: Input image data.

        Returns:
            Cleaned image data.
        """
        # Count problematic values
        n_nan = np.sum(np.isnan(data))
        n_inf = np.sum(np.isinf(data))

        if n_nan > 0:
            logger.warning(f"Found {n_nan} NaN values, replacing with 0")
        if n_inf > 0:
            logger.warning(f"Found {n_inf} infinite values, replacing with 0")

        # Replace NaN and infinite values with 0
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        return data

    @staticmethod
    def _extract_wcs(header: fits.Header) -> WCS:
        """Extract WCS information from FITS header.

        Args:
            header: FITS header.

        Returns:
            WCS object.
        """
        try:
            wcs = WCS(header)

            # For 3D/4D headers, extract celestial WCS
            if wcs.naxis > 2:
                wcs = wcs.celestial

            # Validate WCS
            if not wcs.has_celestial:
                logger.warning("WCS does not contain celestial coordinates")

            return wcs

        except Exception as e:
            logger.warning(f"Error extracting WCS: {e}")
            # Return a dummy WCS for pixel coordinates
            return WCS(naxis=2)


class RegionFileWriter:
    """Utility class for writing region files in various formats."""

    @staticmethod
    def write_ds9_region_file(regions: List[RegionInfo],
                            output_path: Union[str, Path],
                            coordinate_system: str = 'fk5',
                            color: str = 'green') -> None:
        """Write regions to DS9 region file format.

        Args:
            regions: List of detected regions.
            output_path: Output file path.
            coordinate_system: Coordinate system ('fk5', 'icrs', 'galactic').
            color: Region color for display.
        """
        output_path = Path(output_path)
        logger.info(f"Writing DS9 region file: {output_path}")

        with open(output_path, 'w') as f:
            # Write header
            f.write("# Region file format: DS9 version 4.1\n")
            f.write(f"global color={color} dashlist=8 3 width=2 font=\"helvetica 12 normal roman\" ")
            f.write(f"select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 ")
            f.write(f"include=1 source=1\n")
            f.write(f"{coordinate_system}\n")

            # Write regions
            for i, region in enumerate(regions, 1):
                # Format: circle(ra, dec, radius)
                f.write(f"circle({region.center_ra:.6f},{region.center_dec:.6f},")
                f.write(f"{region.radius_arcsec}\")")

                # Add metadata as comments
                f.write(f" # text={{DDetector-{i:03d}}} ")
                f.write(f"tag={{type={region.detection_type}}} ")
                f.write(f"tag={{confidence={region.confidence:.3f}}}")

                if region.stats.get('area_pixels'):
                    f.write(f" tag={{area_pix={region.stats['area_pixels']:.0f}}}")

                f.write("\n")

        logger.info(f"Wrote {len(regions)} regions to {output_path}")

    @staticmethod
    def write_casa_region_file(regions: List[RegionInfo],
                             output_path: Union[str, Path]) -> None:
        """Write regions to CASA region file format (CRTF).

        Args:
            regions: List of detected regions.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        logger.info(f"Writing CASA region file: {output_path}")

        with open(output_path, 'w') as f:
            # Write header
            f.write("#CRTFv0 CASA Region Text Format version 0\n")
            f.write("# DDetector generated regions for direction-dependent calibration\n")

            # Write regions
            for i, region in enumerate(regions, 1):
                # Format: circle[[ra, dec], radius]
                f.write(f"circle[[{region.center_ra:.6f}deg, {region.center_dec:.6f}deg], ")
                f.write(f"{region.radius_arcsec}arcsec] ")
                f.write(f"coord=J2000, corr=[I], ")
                f.write(f"color=red, linewidth=2, ")
                f.write(f"label='DDetector-{i:03d} ({region.detection_type}, ")
                f.write(f"conf={region.confidence:.2f})'\n")

        logger.info(f"Wrote {len(regions)} regions to {output_path}")

    @staticmethod
    def write_kvis_region_file(regions: List[RegionInfo],
                             output_path: Union[str, Path]) -> None:
        """Write regions to KVIS annotation file format.

        Args:
            regions: List of detected regions.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        logger.info(f"Writing KVIS region file: {output_path}")

        with open(output_path, 'w') as f:
            f.write("# KVIS annotation file\n")
            f.write("# Generated by DDetector\n")
            f.write("COORD W\n")  # World coordinates

            for i, region in enumerate(regions, 1):
                # KVIS circle format: CIRCLE ra dec radius
                f.write(f"CIRCLE {region.center_ra:.6f} {region.center_dec:.6f} ")
                f.write(f"{region.radius_arcsec/3600.0:.6f}\n")  # Convert to degrees

        logger.info(f"Wrote {len(regions)} regions to {output_path}")


class ConfigurationManager:
    """Utility class for managing configuration files."""

    @staticmethod
    def load_config(config_path: Union[str, Path]) -> DetectionParameters:
        """Load detection parameters from JSON configuration file.

        Args:
            config_path: Path to JSON configuration file.

        Returns:
            DetectionParameters object.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        logger.info(f"Loading configuration from: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Remove description field if present
        config_dict.pop('description', None)

        return DetectionParameters.from_dict(config_dict)

    @staticmethod
    def save_config(params: DetectionParameters,
                   output_path: Union[str, Path],
                   description: Optional[str] = None) -> None:
        """Save detection parameters to JSON configuration file.

        Args:
            params: DetectionParameters to save.
            output_path: Output file path.
            description: Optional description of the configuration.
        """
        output_path = Path(output_path)
        logger.info(f"Saving configuration to: {output_path}")

        config_dict = params.to_dict()
        if description:
            config_dict['description'] = description

        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

    @staticmethod
    def get_builtin_config_path(config_name: str) -> Path:
        """Get path to built-in configuration file.

        Args:
            config_name: Name of built-in configuration.

        Returns:
            Path to configuration file.
        """
        # This would point to package data
        package_dir = Path(__file__).parent
        config_path = package_dir / "configs" / f"{config_name}.json"

        if not config_path.exists():
            available_configs = [f.stem for f in (package_dir / "configs").glob("*.json")]
            raise ValueError(f"Configuration '{config_name}' not found. "
                           f"Available: {available_configs}")

        return config_path


class ResultsExporter:
    """Utility class for exporting analysis results in various formats."""

    @staticmethod
    def export_summary_json(result: DetectionResult,
                          output_path: Union[str, Path]) -> None:
        """Export detection results summary to JSON.

        Args:
            result: DetectionResult object.
            output_path: Output file path.
        """
        output_path = Path(output_path)
        logger.info(f"Exporting summary to: {output_path}")

        summary = result.to_summary_dict()

        # Add individual region details
        summary['regions'] = [region.to_dict() for region in result.regions]

        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

    @staticmethod
    def export_csv_catalog(regions: List[RegionInfo],
                         output_path: Union[str, Path]) -> None:
        """Export regions as CSV catalog.

        Args:
            regions: List of detected regions.
            output_path: Output file path.
        """
        import csv

        output_path = Path(output_path)
        logger.info(f"Exporting CSV catalog to: {output_path}")

        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            writer.writerow([
                'region_id', 'ra_deg', 'dec_deg', 'radius_arcsec',
                'confidence', 'detection_type', 'area_arcsec2',
                'pixel_x', 'pixel_y', 'area_pixels'
            ])

            # Write regions
            for i, region in enumerate(regions, 1):
                writer.writerow([
                    i, region.center_ra, region.center_dec, region.radius_arcsec,
                    region.confidence, region.detection_type, region.area_arcsec2,
                    region.pixel_coords[0], region.pixel_coords[1],
                    region.stats.get('area_pixels', 0)
                ])

    @staticmethod
    def export_fits_mask(regions: List[RegionInfo],
                        reference_fits: Union[str, Path],
                        output_path: Union[str, Path],
                        wcs: Optional[WCS] = None) -> None:
        """Export regions as FITS mask image.

        Args:
            regions: List of detected regions.
            reference_fits: Reference FITS file for header/WCS.
            output_path: Output FITS file path.
            wcs: Optional WCS object (if not provided, loaded from reference).
        """
        output_path = Path(output_path)
        logger.info(f"Exporting FITS mask to: {output_path}")

        # Load reference image for header/shape
        ref_data, ref_wcs, ref_header = FitsImageLoader.load_fits_image(reference_fits)

        if wcs is None:
            wcs = ref_wcs

        # Create mask array
        mask = np.zeros_like(ref_data, dtype=np.uint8)

        # Fill regions in mask
        for i, region in enumerate(regions, 1):
            try:
                # Convert sky coordinates to pixel coordinates
                sky_coord = SkyCoord(region.center_ra * u.deg, region.center_dec * u.deg)
                x_pix, y_pix = wcs.world_to_pixel(sky_coord)

                # Create circular mask
                y_indices, x_indices = np.ogrid[:mask.shape[0], :mask.shape[1]]
                pixel_scale = abs(wcs.pixel_scale_matrix[0, 0]) * 3600  # arcsec/pixel
                radius_pix = region.radius_arcsec / pixel_scale

                distance = np.sqrt((x_indices - x_pix)**2 + (y_indices - y_pix)**2)
                mask[distance <= radius_pix] = i  # Use region ID as mask value

            except Exception as e:
                logger.warning(f"Could not create mask for region {i}: {e}")

        # Create new FITS file
        hdu = fits.PrimaryHDU(data=mask, header=ref_header)
        hdu.header['COMMENT'] = 'DDetector region mask'
        hdu.header['COMMENT'] = f'Generated from {len(regions)} detected regions'

        hdu.writeto(output_path, overwrite=True)
        logger.info(f"Exported mask with {len(regions)} regions")


def write_ds9_region_file(regions: List[RegionInfo],
                         output_path: Union[str, Path],
                         **kwargs) -> None:
    """Convenience function for writing DS9 region files.

    This is a wrapper around RegionFileWriter.write_ds9_region_file
    for backward compatibility and ease of use.
    """
    RegionFileWriter.write_ds9_region_file(regions, output_path, **kwargs)


def write_casa_region_file(regions: List[RegionInfo],
                          output_path: Union[str, Path]) -> None:
    """Convenience function for writing CASA region files.

    This is a wrapper around RegionFileWriter.write_casa_region_file
    for backward compatibility and ease of use.
    """
    RegionFileWriter.write_casa_region_file(regions, output_path)


def load_detection_config(config_path: Union[str, Path]) -> DetectionParameters:
    """Convenience function for loading detection configuration.

    This is a wrapper around ConfigurationManager.load_config
    for backward compatibility and ease of use.
    """
    return ConfigurationManager.load_config(config_path)


# Export main functions and classes
__all__ = [
    'FitsImageLoader',
    'RegionFileWriter',
    'ConfigurationManager',
    'ResultsExporter',
    'write_ds9_region_file',
    'write_casa_region_file',
    'load_detection_config',
]
