"""
Radio Interferometric Direction-Dependent Calibration Region Detection Tool

This tool analyzes residual FITS images from radio interferometric observations
to automatically identify regions requiring direction-dependent calibration.
"""

import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_log_compiles', False)
import jax.numpy as jnp
from jax import jit, vmap
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import ndimage
from scipy.signal import find_peaks
from sklearn.cluster import DBSCAN
from skimage import measure, morphology, filters
import argparse
import logging
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
import json
from .core import RegionInfo, DetectionParameters

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionParameters:
    """Parameters for calibration region detection."""
    negative_threshold_sigma: float = 3.0  # Threshold for negative outliers in sigma
    spike_threshold_percentile: float = 95.0  # Percentile threshold for spike detection
    ripple_frequency_range: Tuple[float, float] = (0.1, 2.0)  # Spatial frequency range for ripples
    min_region_size_pixels: int = 50  # Minimum region size in pixels
    clustering_eps: float = 10.0  # DBSCAN clustering epsilon in pixels
    clustering_min_samples: int = 5  # DBSCAN minimum samples
    morphology_disk_size: int = 3  # Morphological operations disk size
    edge_buffer_pixels: int = 20  # Buffer from image edges


class ResidualImageAnalyzer:
    """Analyzes residual images for direction-dependent calibration regions."""

    def __init__(self, params: DetectionParameters):
        self.params = params
        self.wcs = None
        self.image_stats = None

    def load_fits_image(self, fits_path: Union[str, Path]) -> Tuple[np.ndarray, WCS]:
        """Load FITS image and extract WCS information."""
        logger.info(f"Loading FITS image: {fits_path}")

        with fits.open(fits_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header

            # Handle different FITS formats (2D, 3D, 4D)
            if data.ndim == 4:
                data = data[0, 0]  # Take first frequency and Stokes
            elif data.ndim == 3:
                data = data[0]  # Take first plane
            elif data.ndim != 2:
                raise ValueError(f"Unsupported data dimensionality: {data.ndim}")

            # Remove NaN values
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

            wcs = WCS(header).celestial if hasattr(WCS(header), 'celestial') else WCS(header)

        logger.info(f"Loaded image shape: {data.shape}")
        return data, wcs

    def compute_image_statistics(self, data: np.ndarray) -> Dict[str, float]:
        """Compute robust statistics for the residual image."""
        # Use JAX for efficient computation
        data_jax = jnp.array(data.flatten())

        # Robust statistics using percentiles
        stats = {
            'mean': float(jnp.mean(data_jax)),
            'std': float(jnp.std(data_jax)),
            'median': float(jnp.percentile(data_jax, 50)),
            'mad': float(jnp.median(jnp.abs(data_jax - jnp.median(data_jax)))),
            'p5': float(jnp.percentile(data_jax, 5)),
            'p95': float(jnp.percentile(data_jax, 95)),
            'min': float(jnp.min(data_jax)),
            'max': float(jnp.max(data_jax)),
            'rms': float(jnp.sqrt(jnp.mean(data_jax**2)))
        }

        # Robust sigma estimate
        stats['robust_sigma'] = 1.4826 * stats['mad']

        logger.info(f"Image statistics: RMS={stats['rms']:.2e}, robust_sigma={stats['robust_sigma']:.2e}")
        return stats

    # @jit
    def detect_negative_outliers(self, data: jnp.ndarray,
                                 threshold_sigma: float,
                                 robust_sigma: float) -> jnp.ndarray:
        """Detect significant negative outliers using JAX."""
        print(data, threshold_sigma, robust_sigma)
        threshold = -threshold_sigma * robust_sigma
        return data < threshold

    def detect_spikes_and_artifacts(self, data: np.ndarray) -> np.ndarray:
        """Detect unnatural spikes and artifacts in the residual image."""
        # Laplacian filter to detect sharp features
        laplacian = ndimage.laplace(data)

        # Threshold based on percentile of Laplacian magnitude
        threshold = np.percentile(np.abs(laplacian), self.params.spike_threshold_percentile)
        spike_mask = np.abs(laplacian) > threshold

        # Morphological closing to connect nearby detections
        kernel = morphology.disk(self.params.morphology_disk_size)
        spike_mask = morphology.binary_closing(spike_mask, kernel)

        return spike_mask

    def detect_ripple_patterns(self, data: np.ndarray) -> np.ndarray:
        """Detect ripple patterns that indicate calibration errors."""
        # Apply Fourier transform to detect periodic patterns
        fft_data = np.fft.fft2(data)
        fft_mag = np.abs(fft_data)

        # Create frequency masks for different spatial frequencies
        ny, nx = data.shape
        freq_y = np.fft.fftfreq(ny)
        freq_x = np.fft.fftfreq(nx)

        freq_grid = np.sqrt(freq_x[None, :]**2 + freq_y[:, None]**2)

        # Focus on mid-range spatial frequencies where ripples typically occur
        freq_min, freq_max = self.params.ripple_frequency_range
        freq_mask = (freq_grid >= freq_min) & (freq_grid <= freq_max)

        # Enhanced FFT magnitude in ripple frequency range
        enhanced_fft = fft_mag * freq_mask

        # Inverse transform to get spatial ripple map
        ripple_map = np.abs(np.fft.ifft2(enhanced_fft))

        # Threshold ripple detections
        threshold = np.percentile(ripple_map, 90)
        ripple_mask = ripple_map > threshold

        # Clean up small isolated regions
        ripple_mask = morphology.remove_small_objects(ripple_mask,
                                                     min_size=self.params.min_region_size_pixels)

        return ripple_mask

    def combine_detection_masks(self, negative_mask: np.ndarray,
                              spike_mask: np.ndarray,
                              ripple_mask: np.ndarray) -> np.ndarray:
        """Combine different detection masks with appropriate weighting."""
        # Create weighted combination
        combined = negative_mask.astype(float) * 0.4 + \
                  spike_mask.astype(float) * 0.3 + \
                  ripple_mask.astype(float) * 0.3

        # Apply threshold to create final binary mask
        final_mask = combined > 0.3

        # Remove regions too close to edges
        buffer = self.params.edge_buffer_pixels
        final_mask[:buffer, :] = False
        final_mask[-buffer:, :] = False
        final_mask[:, :buffer] = False
        final_mask[:, -buffer:] = False

        return final_mask

    def extract_regions_from_mask(self, mask: np.ndarray,
                                 individual_masks: Dict[str, np.ndarray]) -> List[RegionInfo]:
        """Extract individual regions from combined detection mask."""
        # Label connected components
        labeled_mask = measure.label(mask)
        regions = measure.regionprops(labeled_mask)

        region_list = []

        for region in regions:
            if region.area < self.params.min_region_size_pixels:
                continue

            # Get region center in pixel coordinates
            center_y, center_x = region.centroid

            # Convert to world coordinates
            if self.wcs is not None:
                try:
                    sky_coord = self.wcs.pixel_to_world(center_x, center_y)
                    ra = sky_coord.ra.deg
                    dec = sky_coord.dec.deg

                    # Estimate region size in arcseconds
                    # Use equivalent radius for irregular shapes
                    equiv_radius_pixels = np.sqrt(region.area / np.pi)
                    pixel_scale = np.abs(self.wcs.pixel_scale_matrix[0, 0]) * 3600  # arcsec/pixel
                    radius_arcsec = equiv_radius_pixels * pixel_scale

                except Exception as e:
                    logger.warning(f"WCS conversion failed: {e}")
                    ra, dec = center_x, center_y
                    radius_arcsec = equiv_radius_pixels
            else:
                ra, dec = center_x, center_y
                radius_arcsec = np.sqrt(region.area / np.pi)

            # Determine detection type and confidence
            region_slice = (slice(region.bbox[0], region.bbox[2]),
                           slice(region.bbox[1], region.bbox[3]))

            detection_types = []
            if np.any(individual_masks['negative'][region_slice]):
                detection_types.append('negative')
            if np.any(individual_masks['spike'][region_slice]):
                detection_types.append('spike')
            if np.any(individual_masks['ripple'][region_slice]):
                detection_types.append('ripple')

            detection_type = '+'.join(detection_types) if detection_types else 'unknown'

            # Calculate confidence based on region properties
            confidence = min(1.0, region.area / (2 * self.params.min_region_size_pixels))

            # Region statistics
            stats = {
                'area_pixels': region.area,
                'eccentricity': region.eccentricity,
                'solidity': region.solidity,
                'extent': region.extent
            }

            region_info = RegionInfo(
                center_ra=ra,
                center_dec=dec,
                radius_arcsec=radius_arcsec,
                confidence=confidence,
                detection_type=detection_type,
                pixel_coords=(int(center_x), int(center_y)),
                stats=stats
            )

            region_list.append(region_info)

        return region_list

    def cluster_nearby_regions(self, regions: List[RegionInfo]) -> List[RegionInfo]:
        """Cluster nearby regions to avoid over-segmentation."""
        if len(regions) < 2:
            return regions

        # Extract pixel coordinates
        coords = np.array([r.pixel_coords for r in regions])

        # Apply DBSCAN clustering
        clustering = DBSCAN(eps=self.params.clustering_eps,
                           min_samples=self.params.clustering_min_samples)
        cluster_labels = clustering.fit_predict(coords)

        # Merge regions in the same cluster
        clustered_regions = []
        unique_labels = np.unique(cluster_labels)

        for label in unique_labels:
            cluster_mask = cluster_labels == label
            cluster_regions = [regions[i] for i in np.where(cluster_mask)[0]]

            if label == -1:  # Noise points (unclustered)
                clustered_regions.extend(cluster_regions)
            else:
                # Merge clustered regions
                merged_region = self.merge_regions(cluster_regions)
                clustered_regions.append(merged_region)

        return clustered_regions

    def merge_regions(self, regions: List[RegionInfo]) -> RegionInfo:
        """Merge multiple regions into a single region."""
        if len(regions) == 1:
            return regions[0]

        # Compute weighted average position
        total_area = sum(r.stats['area_pixels'] for r in regions)

        avg_ra = sum(r.center_ra * r.stats['area_pixels'] for r in regions) / total_area
        avg_dec = sum(r.center_dec * r.stats['area_pixels'] for r in regions) / total_area

        # Maximum radius to encompass all regions
        max_radius = max(r.radius_arcsec for r in regions) * 1.2

        # Combined detection types
        all_types = set()
        for r in regions:
            all_types.update(r.detection_type.split('+'))
        detection_type = '+'.join(sorted(all_types))

        # Average confidence weighted by area
        avg_confidence = sum(r.confidence * r.stats['area_pixels'] for r in regions) / total_area

        # Merged statistics
        merged_stats = {
            'area_pixels': total_area,
            'eccentricity': np.mean([r.stats['eccentricity'] for r in regions]),
            'solidity': np.mean([r.stats['solidity'] for r in regions]),
            'extent': np.mean([r.stats['extent'] for r in regions])
        }

        return RegionInfo(
            center_ra=avg_ra,
            center_dec=avg_dec,
            radius_arcsec=max_radius,
            confidence=avg_confidence,
            detection_type=detection_type,
            pixel_coords=(int(avg_ra), int(avg_dec)),  # Approximate
            stats=merged_stats
        )

    def analyze_residual_image(self, fits_path: Union[str, Path]) -> List[RegionInfo]:
        """Main analysis pipeline for residual image."""
        logger.info("Starting residual image analysis...")

        # Load image
        data, wcs = self.load_fits_image(fits_path)
        self.wcs = wcs

        # Compute image statistics
        self.image_stats = self.compute_image_statistics(data)

        # Convert to JAX array for efficient computation
        data_jax = jnp.array(data)

        # Detect negative outliers
        negative_mask = np.array(self.detect_negative_outliers(
            data_jax,
            self.params.negative_threshold_sigma,
            self.image_stats['robust_sigma']
        ))

        # Detect spikes and artifacts
        spike_mask = self.detect_spikes_and_artifacts(data)

        # Detect ripple patterns
        ripple_mask = self.detect_ripple_patterns(data)

        logger.info(f"Detection summary - Negative: {np.sum(negative_mask)} pixels, "
                   f"Spikes: {np.sum(spike_mask)} pixels, "
                   f"Ripples: {np.sum(ripple_mask)} pixels")

        # Combine detection masks
        individual_masks = {
            'negative': negative_mask,
            'spike': spike_mask,
            'ripple': ripple_mask
        }

        combined_mask = self.combine_detection_masks(negative_mask, spike_mask, ripple_mask)

        # Extract regions
        regions = self.extract_regions_from_mask(combined_mask, individual_masks)

        # Cluster nearby regions
        clustered_regions = self.cluster_nearby_regions(regions)

        logger.info(f"Found {len(clustered_regions)} calibration regions")

        return clustered_regions

def write_ds9_region_file(regions: List[RegionInfo], output_path: Union[str, Path],
                         coordinate_system: str = 'fk5'):
    """Write regions to DS9 region file format."""
    logger.info(f"Writing DS9 region file: {output_path}")

    with open(output_path, 'w') as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f"global color=green dashlist=8 3 width=1 font=\"helvetica 10 normal roman\" ")
        f.write(f"select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 ")
        f.write(f"include=1 source=1\n")
        f.write(f"{coordinate_system}\n")

        for i, region in enumerate(regions):
            # Write circle region
            f.write(f"circle({region.center_ra:.6f},{region.center_dec:.6f},")
            f.write(f"{region.radius_arcsec}\")")
            f.write(f" # text={{DDCal-{i+1} ({region.detection_type})}}")
            f.write(f" tag={{confidence={region.confidence:.2f}}}\n")

def write_casa_region_file(regions: List[RegionInfo], output_path: Union[str, Path]):
    """Write regions to CASA region file format."""
    logger.info(f"Writing CASA region file: {output_path}")

    with open(output_path, 'w') as f:
        f.write("#CRTFv0 CASA Region Text Format version 0\n")

        for i, region in enumerate(regions):
            f.write(f"circle[[{region.center_ra:.6f}deg, {region.center_dec:.6f}deg], ")
            f.write(f"{region.radius_arcsec}arcsec] coord=J2000, ")
            f.write(f"corr=[I], color=red, ")
            f.write(f"label='DDCal-{i+1} ({region.detection_type})'\n")

def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Detect direction-dependent calibration regions in residual images"
    )
    parser.add_argument("input_fits", help="Input residual FITS image")
    parser.add_argument("-o", "--output", default="ddetector_regions",
                       help="Output region file prefix (default: ddetector_regions)")
    parser.add_argument("--format", choices=['ds9', 'casa', 'both'], default='both',
                       help="Output region file format")
    parser.add_argument("--config", help="Configuration JSON file")
    parser.add_argument("--negative-sigma", type=float, default=3.0,
                       help="Negative outlier threshold in sigma units")
    parser.add_argument("--min-size", type=int, default=50,
                       help="Minimum region size in pixels")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    params = DetectionParameters()
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
        # Update parameters from config
        for key, value in config.items():
            if hasattr(params, key):
                setattr(params, key, value)

    # Override with command line arguments
    params.negative_threshold_sigma = args.negative_sigma
    params.min_region_size_pixels = args.min_size

    # Initialize analyzer
    analyzer = ResidualImageAnalyzer(params)

    # Analyze image
    regions = analyzer.analyze_residual_image(args.input_fits)

    if not regions:
        logger.warning("No calibration regions detected")
        return

    # Write output files
    if args.format in ['ds9', 'both']:
        write_ds9_region_file(regions, f"{args.output}.reg")

    if args.format in ['casa', 'both']:
        write_casa_region_file(regions, f"{args.output}.crtf")

    # Write summary report
    with open(f"{args.output}_summary.json", 'w') as f:
        summary = {
            'total_regions': len(regions),
            'image_stats': analyzer.image_stats,
            'detection_params': {
                'negative_threshold_sigma': params.negative_threshold_sigma,
                'min_region_size_pixels': params.min_region_size_pixels,
                'clustering_eps': params.clustering_eps
            },
            'regions': [
                {
                    'id': i + 1,
                    'ra': r.center_ra,
                    'dec': r.center_dec,
                    'radius_arcsec': r.radius_arcsec,
                    'confidence': r.confidence,
                    'detection_type': r.detection_type,
                    'stats': r.stats
                }
                for i, r in enumerate(regions)
            ]
        }
        json.dump(summary, f, indent=2)

    logger.info(f"Analysis complete. Found {len(regions)} regions requiring DDCal.")

if __name__ == "__main__":
    main()
