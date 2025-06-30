"""
Visualization and advanced utilities for DDCal detector.

This module provides visualization tools, advanced analysis features,
and integration utilities for the direction-dependent calibration detector.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LogNorm, PowerNorm
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from typing import List, Tuple, Optional, Dict, Any
import jax.numpy as jnp
from pathlib import Path
import json

from ddcal_detector import RegionInfo, ResidualImageAnalyzer


class DetectionVisualizer:
    """Advanced visualization tools for DDCal detection results."""

    def __init__(self, figsize: Tuple[float, float] = (12, 10)):
        self.figsize = figsize
        self.colors = {
            'negative': 'red',
            'spike': 'blue',
            'ripple': 'green',
            'combined': 'orange'
        }

    def plot_detection_overview(self,
                              image_data: np.ndarray,
                              regions: List[RegionInfo],
                              wcs: Optional[WCS] = None,
                              output_path: Optional[str] = None,
                              show_masks: bool = True) -> plt.Figure:
        """Create comprehensive detection overview plot."""

        fig = plt.figure(figsize=(16, 12))

        # Main image panel
        ax_main = plt.subplot(2, 3, (1, 4))

        # Display residual image with adaptive scaling
        vmin, vmax = self._get_image_scale(image_data)

        if wcs is not None:
            im = ax_main.imshow(image_data, origin='lower', cmap='RdBu_r',
                              vmin=vmin, vmax=vmax, aspect='equal')
            ax_main.set_xlabel('RA (J2000)')
            ax_main.set_ylabel('Dec (J2000)')
        else:
            im = ax_main.imshow(image_data, origin='lower', cmap='RdBu_r',
                              vmin=vmin, vmax=vmax, aspect='equal')
            ax_main.set_xlabel('X (pixels)')
            ax_main.set_ylabel('Y (pixels)')

        plt.colorbar(im, ax=ax_main, label='Flux Density (Jy/beam)')

        # Overlay detected regions
        for i, region in enumerate(regions):
            if wcs is not None:
                # Convert sky coordinates to pixel coordinates
                sky_coord = SkyCoord(region.center_ra * u.deg,
                                   region.center_dec * u.deg)
                x_pix, y_pix = wcs.world_to_pixel(sky_coord)
                radius_pix = region.radius_arcsec / 3600.0 / abs(wcs.pixel_scale_matrix[0, 0])
            else:
                x_pix, y_pix = region.pixel_coords
                radius_pix = region.radius_arcsec

            # Color by detection type
            color = self._get_region_color(region.detection_type)

            circle = patches.Circle((x_pix, y_pix), radius_pix,
                                  linewidth=2, edgecolor=color,
                                  facecolor='none', alpha=0.8)
            ax_main.add_patch(circle)

            # Add region labels
            ax_main.annotate(f'{i+1}', (x_pix, y_pix),
                           color='white', fontsize=10, fontweight='bold',
                           ha='center', va='center')

        ax_main.set_title('Residual Image with Detected Regions')

        # Statistics panel
        ax_stats = plt.subplot(2, 3, 2)
        self._plot_region_statistics(ax_stats, regions)

        # Detection type distribution
        ax_types = plt.subplot(2, 3, 3)
        self._plot_detection_types(ax_types, regions)

        # Confidence histogram
        ax_conf = plt.subplot(2, 3, 5)
        self._plot_confidence_distribution(ax_conf, regions)

        # Size distribution
        ax_size = plt.subplot(2, 3, 6)
        self._plot_size_distribution(ax_size, regions)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Detection overview saved to {output_path}")

        return fig

    def plot_detection_masks(self,
                           analyzer: ResidualImageAnalyzer,
                           negative_mask: np.ndarray,
                           spike_mask: np.ndarray,
                           ripple_mask: np.ndarray,
                           combined_mask: np.ndarray,
                           output_path: Optional[str] = None) -> plt.Figure:
        """Plot individual detection masks and their combination."""

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))

        # Original image (if available)
        if hasattr(analyzer, '_original_data'):
            ax = axes[0, 0]
            im = ax.imshow(analyzer._original_data, origin='lower',
                          cmap='RdBu_r', aspect='equal')
            ax.set_title('Original Residual Image')
            plt.colorbar(im, ax=ax)
        else:
            axes[0, 0].axis('off')
            axes[0, 0].text(0.5, 0.5, 'Original\nImage\nNot Available',
                          ha='center', va='center', transform=axes[0, 0].transAxes)

        # Individual masks
        mask_data = [
            (negative_mask, 'Negative Outliers', 'Reds'),
            (spike_mask, 'Spikes & Artifacts', 'Blues'),
            (ripple_mask, 'Ripple Patterns', 'Greens')
        ]

        for i, (mask, title, cmap) in enumerate(mask_data):
            ax = axes[0, i+1] if i < 2 else axes[1, 0]
            im = ax.imshow(mask, origin='lower', cmap=cmap, aspect='equal')
            ax.set_title(title)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')

        # Combined mask
        ax = axes[1, 1]
        im = ax.imshow(combined_mask, origin='lower', cmap='hot', aspect='equal')
        ax.set_title('Combined Detection Mask')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=ax)

        # Statistics panel
        ax = axes[1, 2]
        self._plot_mask_statistics(ax, negative_mask, spike_mask,
                                 ripple_mask, combined_mask)

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Detection masks saved to {output_path}")

        return fig

    def plot_fourier_analysis(self,
                            image_data: np.ndarray,
                            ripple_mask: np.ndarray,
                            output_path: Optional[str] = None) -> plt.Figure:
        """Plot Fourier domain analysis for ripple detection."""

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Original image
        ax = axes[0, 0]
        im = ax.imshow(image_data, origin='lower', cmap='RdBu_r', aspect='equal')
        ax.set_title('Residual Image')
        plt.colorbar(im, ax=ax)

        # FFT magnitude
        fft_data = np.fft.fft2(image_data)
        fft_mag = np.abs(np.fft.fftshift(fft_data))

        ax = axes[0, 1]
        im = ax.imshow(np.log10(fft_mag + 1e-10), origin='lower',
                      cmap='viridis', aspect='equal')
        ax.set_title('FFT Magnitude (log scale)')
        plt.colorbar(im, ax=ax)

        # Frequency space filtering
        ny, nx = image_data.shape
        freq_y = np.fft.fftfreq(ny)
        freq_x = np.fft.fftfreq(nx)
        freq_grid = np.sqrt(freq_x[None, :]**2 + freq_y[:, None]**2)

        ax = axes[1, 0]
        im = ax.imshow(np.fft.fftshift(freq_grid), origin='lower',
                      cmap='plasma', aspect='equal')
        ax.set_title('Spatial Frequency Grid')
        plt.colorbar(im, ax=ax)

        # Ripple detection result
        ax = axes[1, 1]
        im = ax.imshow(ripple_mask, origin='lower', cmap='Greens', aspect='equal')
        ax.set_title('Detected Ripple Patterns')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Fourier analysis saved to {output_path}")

        return fig

    def create_summary_report(self,
                            regions: List[RegionInfo],
                            image_stats: Dict[str, float],
                            detection_params: Dict[str, Any],
                            output_path: str = "ddcal_report.html") -> str:
        """Generate comprehensive HTML report."""

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>DDCal Detection Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .region-table {{ border-collapse: collapse; width: 100%; }}
                .region-table th, .region-table td {{ border: 1px solid #ddd; padding: 8px; }}
                .region-table th {{ background-color: #f2f2f2; }}
                .stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
                .stat-box {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Direction-Dependent Calibration Detection Report</h1>
                <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="section">
                <h2>Executive Summary</h2>
                <div class="stats">
                    <div class="stat-box">
                        <h3>Total Regions</h3>
                        <p style="font-size: 24px; margin: 0;">{len(regions)}</p>
                    </div>
                    <div class="stat-box">
                        <h3>High Confidence</h3>
                        <p style="font-size: 24px; margin: 0;">{sum(1 for r in regions if r.confidence > 0.7)}</p>
                    </div>
                    <div class="stat-box">
                        <h3>Recommendation</h3>
                        <p style="font-size: 18px; margin: 0;">
                        {"Apply DDCal" if len(regions) > 0 else "No DDCal needed"}
                        </p>
                    </div>
                </div>
            </div>

            <div class="section">
                <h2>Image Statistics</h2>
                <ul>
                    <li>RMS: {image_stats.get('rms', 0):.2e}</li>
                    <li>Robust Sigma: {image_stats.get('robust_sigma', 0):.2e}</li>
                    <li>Dynamic Range: {abs(image_stats.get('max', 0)/image_stats.get('robust_sigma', 1)):.1f}</li>
                </ul>
            </div>

            <div class="section">
                <h2>Detection Parameters</h2>
                <ul>
        """

        for key, value in detection_params.items():
            html_content += f"                    <li>{key}: {value}</li>\n"

        html_content += """
                </ul>
            </div>

            <div class="section">
                <h2>Detected Regions</h2>
        """

        if regions:
            html_content += """
                <table class="region-table">
                    <tr>
                        <th>ID</th>
                        <th>RA (deg)</th>
                        <th>Dec (deg)</th>
                        <th>Size (arcsec)</th>
                        <th>Type</th>
                        <th>Confidence</th>
                        <th>Area (pixels)</th>
                    </tr>
            """

            for i, region in enumerate(regions, 1):
                html_content += f"""
                    <tr>
                        <td>{i}</td>
                        <td>{region.center_ra:.6f}</td>
                        <td>{region.center_dec:.6f}</td>
                        <td>{region.radius_arcsec:.1f}</td>
                        <td>{region.detection_type}</td>
                        <td>{region.confidence:.2f}</td>
                        <td>{region.stats['area_pixels']}</td>
                    </tr>
                """

            html_content += """
                </table>
            """
        else:
            html_content += "<p>No calibration regions detected.</p>"

        html_content += """
            </div>
        </body>
        </html>
        """

        with open(output_path, 'w') as f:
            f.write(html_content)

        print(f"HTML report saved to {output_path}")
        return output_path

    def _get_image_scale(self, image_data: np.ndarray) -> Tuple[float, float]:
        """Get appropriate image scaling for visualization."""
        # Use robust percentiles for scaling
        p1, p99 = np.percentile(image_data[np.isfinite(image_data)], [1, 99])

        # Symmetric scaling around zero for residual images
        scale = max(abs(p1), abs(p99))
        return -scale, scale

    def _get_region_color(self, detection_type: str) -> str:
        """Get color for region based on detection type."""
        if '+' in detection_type:
            return self.colors['combined']
        return self.colors.get(detection_type, 'yellow')

    def _plot_region_statistics(self, ax: plt.Axes, regions: List[RegionInfo]):
        """Plot region statistics summary."""
        if not regions:
            ax.text(0.5, 0.5, 'No regions\ndetected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Region Statistics')
            return

        stats_text = f"""
        Total Regions: {len(regions)}

        Confidence:
          Mean: {np.mean([r.confidence for r in regions]):.2f}
          Max:  {np.max([r.confidence for r in regions]):.2f}
          Min:  {np.min([r.confidence for r in regions]):.2f}

        Size (arcsec):
          Mean: {np.mean([r.radius_arcsec for r in regions]):.1f}
          Max:  {np.max([r.radius_arcsec for r in regions]):.1f}
          Min:  {np.min([r.radius_arcsec for r in regions]):.1f}

        Area (pixels):
          Total: {sum([r.stats['area_pixels'] for r in regions])}
        """

        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', fontfamily='monospace')
        ax.set_title('Region Statistics')
        ax.axis('off')

    def _plot_detection_types(self, ax: plt.Axes, regions: List[RegionInfo]):
        """Plot detection type distribution."""
        if not regions:
            ax.text(0.5, 0.5, 'No regions\ndetected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Detection Types')
            return

        types = [r.detection_type for r in regions]
        unique_types, counts = np.unique(types, return_counts=True)

        colors = [self._get_region_color(t) for t in unique_types]
        wedges, texts, autotexts = ax.pie(counts, labels=unique_types,
                                         colors=colors, autopct='%1.1f%%')
        ax.set_title('Detection Types')

    def _plot_confidence_distribution(self, ax: plt.Axes, regions: List[RegionInfo]):
        """Plot confidence score distribution."""
        if not regions:
            ax.text(0.5, 0.5, 'No regions\ndetected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Confidence Distribution')
            return

        confidences = [r.confidence for r in regions]
        ax.hist(confidences, bins=10, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(confidences), color='red', linestyle='--',
                  label=f'Mean: {np.mean(confidences):.2f}')
        ax.set_xlabel('Confidence Score')
        ax.set_ylabel('Count')
        ax.set_title('Confidence Distribution')
        ax.legend()

    def _plot_size_distribution(self, ax: plt.Axes, regions: List[RegionInfo]):
        """Plot size distribution."""
        if not regions:
            ax.text(0.5, 0.5, 'No regions\ndetected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
            ax.set_title('Size Distribution')
            return

        sizes = [r.radius_arcsec for r in regions]
        ax.hist(sizes, bins=10, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(sizes), color='red', linestyle='--',
                  label=f'Mean: {np.mean(sizes):.1f}"')
        ax.set_xlabel('Radius (arcsec)')
        ax.set_ylabel('Count')
        ax.set_title('Size Distribution')
        ax.legend()

    def _plot_mask_statistics(self, ax: plt.Axes,
                            negative_mask: np.ndarray,
                            spike_mask: np.ndarray,
                            ripple_mask: np.ndarray,
                            combined_mask: np.ndarray):
        """Plot statistics about detection masks."""

        total_pixels = negative_mask.size
        stats = {
            'Negative': np.sum(negative_mask) / total_pixels * 100,
            'Spikes': np.sum(spike_mask) / total_pixels * 100,
            'Ripples': np.sum(ripple_mask) / total_pixels * 100,
            'Combined': np.sum(combined_mask) / total_pixels * 100
        }

        colors = ['red', 'blue', 'green', 'orange']
        bars = ax.bar(stats.keys(), stats.values(), color=colors, alpha=0.7)

        ax.set_ylabel('Coverage (%)')
        ax.set_title('Detection Coverage')
        ax.tick_params(axis='x', rotation=45)

        # Add value labels on bars
        for bar, value in zip(bars, stats.values()):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.2f}%', ha='center', va='bottom', fontsize=8)


class AdvancedAnalyzer:
    """Advanced analysis tools extending the basic DDCal detector."""

    def __init__(self, analyzer: ResidualImageAnalyzer):
        self.analyzer = analyzer

    def analyze_temporal_consistency(self,
                                   image_sequence: List[str],
                                   output_path: str = "temporal_analysis.json") -> Dict[str, Any]:
        """Analyze consistency of detections across time sequence."""

        sequence_results = []
        all_positions = []

        for i, image_path in enumerate(image_sequence):
            print(f"Processing time step {i+1}/{len(image_sequence)}: {image_path}")

            regions = self.analyzer.analyze_residual_image(image_path)

            result = {
                'time_step': i,
                'image': image_path,
                'num_regions': len(regions),
                'regions': [
                    {
                        'ra': r.center_ra,
                        'dec': r.center_dec,
                        'confidence': r.confidence,
                        'type': r.detection_type
                    }
                    for r in regions
                ]
            }
            sequence_results.append(result)

            # Collect positions for clustering analysis
            for region in regions:
                all_positions.append((region.center_ra, region.center_dec, i))

        # Analyze temporal clustering
        temporal_analysis = self._analyze_position_clustering(all_positions)

        # Summary statistics
        num_detections = [r['num_regions'] for r in sequence_results]
        summary = {
            'total_images': len(image_sequence),
            'mean_detections_per_image': np.mean(num_detections),
            'std_detections_per_image': np.std(num_detections),
            'max_detections': np.max(num_detections),
            'images_with_detections': sum(1 for n in num_detections if n > 0),
            'temporal_clusters': temporal_analysis,
            'sequence_results': sequence_results
        }

        # Save results
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Temporal analysis saved to {output_path}")
        return summary

    def compare_with_source_catalog(self,
                                  regions: List[RegionInfo],
                                  catalog_path: str,
                                  match_radius_arcsec: float = 30.0) -> Dict[str, Any]:
        """Compare detected regions with known source catalog."""

        try:
            # Load catalog (assume simple CSV format)
            import pandas as pd
            catalog = pd.read_csv(catalog_path)

            # Assume catalog has 'RA', 'DEC', 'Flux' columns
            if not all(col in catalog.columns for col in ['RA', 'DEC']):
                raise ValueError("Catalog must have 'RA' and 'DEC' columns")

            catalog_coords = SkyCoord(catalog['RA'].values * u.deg,
                                    catalog['DEC'].values * u.deg)

        except Exception as e:
            print(f"Error loading catalog: {e}")
            return {'error': str(e)}

        matches = []
        unmatched_regions = []

        for i, region in enumerate(regions):
            region_coord = SkyCoord(region.center_ra * u.deg,
                                  region.center_dec * u.deg)

            # Find closest catalog source
            separations = region_coord.separation(catalog_coords)
            min_sep_idx = np.argmin(separations)
            min_separation = separations[min_sep_idx]

            if min_separation.arcsec < match_radius_arcsec:
                match_info = {
                    'region_id': i,
                    'catalog_id': min_sep_idx,
                    'separation_arcsec': min_separation.arcsec,
                    'catalog_source': {
                        'ra': catalog.iloc[min_sep_idx]['RA'],
                        'dec': catalog.iloc[min_sep_idx]['DEC'],
                        'flux': catalog.iloc[min_sep_idx].get('Flux', 'N/A')
                    },
                    'region_type': region.detection_type,
                    'region_confidence': region.confidence
                }
                matches.append(match_info)
            else:
                unmatched_regions.append({
                    'region_id': i,
                    'ra': region.center_ra,
                    'dec': region.center_dec,
                    'closest_source_separation': min_separation.arcsec,
                    'detection_type': region.detection_type
                })

        analysis = {
            'total_regions': len(regions),
            'matched_regions': len(matches),
            'unmatched_regions': len(unmatched_regions),
            'match_rate': len(matches) / len(regions) if regions else 0,
            'matches': matches,
            'unmatched': unmatched_regions,
            'catalog_size': len(catalog),
            'match_radius_arcsec': match_radius_arcsec
        }

        return analysis

    def estimate_calibration_impact(self,
                                  regions: List[RegionInfo],
                                  image_stats: Dict[str, float]) -> Dict[str, Any]:
        """Estimate the potential impact of applying direction-dependent calibration."""

        if not regions:
            return {
                'estimated_improvement': 0.0,
                'priority_regions': [],
                'total_affected_area': 0.0,
                'recommendation': 'No DDCal needed'
            }

        # Calculate total affected area
        total_area = sum(np.pi * (r.radius_arcsec/3600)**2 for r in regions)  # sq degrees

        # Estimate improvement based on region properties
        high_confidence_regions = [r for r in regions if r.confidence > 0.7]
        negative_regions = [r for r in regions if 'negative' in r.detection_type]

        # Heuristic improvement estimate
        base_improvement = min(0.3, len(regions) * 0.05)  # Max 30% improvement
        confidence_boost = np.mean([r.confidence for r in regions]) * 0.2
        negative_penalty = len(negative_regions) / len(regions) * 0.1

        estimated_improvement = base_improvement + confidence_boost + negative_penalty
        estimated_improvement = min(0.5, estimated_improvement)  # Cap at 50%

        # Priority ranking
        priority_regions = sorted(regions,
                                key=lambda r: r.confidence * r.stats['area_pixels'],
                                reverse=True)[:5]

        # Recommendation logic
        if len(regions) >= 5 or any(r.confidence > 0.8 for r in regions):
            recommendation = "Strongly recommend DDCal"
        elif len(regions) >= 2:
            recommendation = "Consider DDCal"
        else:
            recommendation = "DDCal may provide minor improvement"

        return {
            'estimated_improvement_percent': estimated_improvement * 100,
            'total_regions': len(regions),
            'high_confidence_regions': len(high_confidence_regions),
            'total_affected_area_sq_deg': total_area,
            'priority_regions': [
                {
                    'ra': r.center_ra,
                    'dec': r.center_dec,
                    'confidence': r.confidence,
                    'type': r.detection_type,
                    'priority_score': r.confidence * r.stats['area_pixels']
                }
                for r in priority_regions
            ],
            'recommendation': recommendation,
            'current_rms': image_stats.get('rms', 0),
            'estimated_final_rms': image_stats.get('rms', 0) * (1 - estimated_improvement)
        }

    def _analyze_position_clustering(self, positions: List[Tuple[float, float, int]]) -> Dict[str, Any]:
        """Analyze spatial-temporal clustering of detections."""

        if not positions:
            return {'clusters': [], 'persistent_sources': []}

        from sklearn.cluster import DBSCAN

        # Extract just spatial coordinates for clustering
        spatial_coords = [(ra, dec) for ra, dec, t in positions]

        if len(spatial_coords) < 2:
            return {'clusters': [], 'persistent_sources': []}

        # Convert to array and normalize (rough approximation)
        coords_array = np.array(spatial_coords)

        # Cluster in angular space (eps in degrees)
        clustering = DBSCAN(eps=0.01, min_samples=2)  # ~36 arcsec at dec=0
        cluster_labels = clustering.fit_predict(coords_array)

        # Analyze clusters
        clusters = []
        for label in np.unique(cluster_labels):
            if label == -1:  # Noise points
                continue

            cluster_positions = [positions[i] for i, l in enumerate(cluster_labels) if l == label]
            time_steps = [t for _, _, t in cluster_positions]

            cluster_info = {
                'cluster_id': int(label),
                'num_detections': len(cluster_positions),
                'time_steps': time_steps,
                'temporal_span': max(time_steps) - min(time_steps) + 1,
                'mean_ra': np.mean([ra for ra, _, _ in cluster_positions]),
                'mean_dec': np.mean([dec for _, dec, _ in cluster_positions]),
                'persistence': len(set(time_steps))  # Number of unique time steps
            }
            clusters.append(cluster_info)

        # Identify persistent sources (detected in multiple time steps)
        persistent_sources = [c for c in clusters if c['persistence'] >= 3]

        return {
            'total_detections': len(positions),
            'spatial_clusters': len(clusters),
            'persistent_sources': len(persistent_sources),
            'clusters': clusters,
            'persistent_source_details': persistent_sources
        }


def create_comprehensive_analysis_report(fits_image: str,
                                       config_file: Optional[str] = None,
                                       output_dir: str = "ddcal_analysis") -> str:
    """Create a comprehensive analysis report with all visualizations."""

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Load configuration
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
        params = DetectionParameters(**config)
    else:
        params = DetectionParameters()

    # Initialize analyzer and visualizer
    analyzer = ResidualImageAnalyzer(params)
    visualizer = DetectionVisualizer()

    print(f"Analyzing {fits_image}...")

    # Load image data for visualization
    data, wcs = analyzer.load_fits_image(fits_image)

    # Run analysis
    regions = analyzer.analyze_residual_image(fits_image)

    # Create visualizations
    print("Creating visualizations...")

    # Main overview plot
    fig1 = visualizer.plot_detection_overview(
        data, regions, wcs,
        output_path / "detection_overview.png"
    )
    plt.close(fig1)

    # Detection masks (if available)
    try:
        # Re-run parts of analysis to get intermediate masks
        stats = analyzer.compute_image_statistics(data)

        # Get individual masks
        data_jax = jnp.array(data)
        negative_mask = np.array(analyzer.detect_negative_outliers(
            data_jax, params.negative_threshold_sigma, stats['robust_sigma']
        ))
        spike_mask = analyzer.detect_spikes_and_artifacts(data)
        ripple_mask = analyzer.detect_ripple_patterns(data)
        combined_mask = analyzer.combine_detection_masks(
            negative_mask, spike_mask, ripple_mask
        )

        fig2 = visualizer.plot_detection_masks(
            analyzer, negative_mask, spike_mask, ripple_mask, combined_mask,
            output_path / "detection_masks.png"
        )
        plt.close(fig2)

        # Fourier analysis
        fig3 = visualizer.plot_fourier_analysis(
            data, ripple_mask,
            output_path / "fourier_analysis.png"
        )
        plt.close(fig3)

    except Exception as e:
        print(f"Warning: Could not create detailed mask plots: {e}")

    # Advanced analysis
    advanced_analyzer = AdvancedAnalyzer(analyzer)

    # Calibration impact estimate
    impact_analysis = advanced_analyzer.estimate_calibration_impact(
        regions, analyzer.image_stats
    )

    # Save detailed results
    detailed_results = {
        'input_image': str(fits_image),
        'detection_parameters': params.__dict__,
        'image_statistics': analyzer.image_stats,
        'regions': [
            {
                'id': i + 1,
                'ra': r.center_ra,
                'dec': r.center_dec,
                'radius_arcsec': r.radius_arcsec,
                'confidence': r.confidence,
                'detection_type': r.detection_type,
                'pixel_coords': r.pixel_coords,
                'stats': r.stats
            }
            for i, r in enumerate(regions)
        ],
        'impact_analysis': impact_analysis,
        'summary': {
            'total_regions': len(regions),
            'recommendation': impact_analysis['recommendation'],
            'estimated_improvement': impact_analysis['estimated_improvement_percent']
        }
    }

    with open(output_path / "detailed_results.json", 'w') as f:
        json.dump(detailed_results, f, indent=2)

    # Create HTML report
    html_report = visualizer.create_summary_report(
        regions, analyzer.image_stats, params.__dict__,
        output_path / "ddcal_report.html"
    )

    print(f"\nComprehensive analysis complete!")
    print(f"Results saved to: {output_path}")
    print(f"Open {html_report} in your browser to view the report.")

    return str(output_path)


# Example usage and testing
if __name__ == "__main__":
    # This would be used as:
    # python visualization_utils.py residual_image.fits

    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualization_utils.py <residual_image.fits> [config.json]")
        sys.exit(1)

    fits_file = sys.argv[1]
    config_file = sys.argv[2] if len(sys.argv) > 2 else None

    report_dir = create_comprehensive_analysis_report(
        fits_file, config_file, "comprehensive_analysis"
    )

    print(f"\nAnalysis complete. Results in: {report_dir}")
