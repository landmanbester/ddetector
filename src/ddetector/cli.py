"""
Command-line interface for DDetector.

This module provides the main entry point for the ddetect command-line tool
used to detect direction-dependent calibration regions in residual images.
"""

import argparse
import logging
import sys
import json
from pathlib import Path
from typing import Optional, List

import click
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_platform_name', 'cpu')
jax.config.update('jax_log_compiles', False)

from .core import DetectionParameters, DetectionResult
from .detection import ResidualImageAnalyzer
from .io_utils import (
    ConfigurationManager, RegionFileWriter, ResultsExporter,
    write_ds9_region_file, write_casa_region_file
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level based on verbosity flags."""
    if quiet:
        logging.getLogger().setLevel(logging.WARNING)
    elif verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    else:
        logging.getLogger().setLevel(logging.INFO)


@click.group(invoke_without_command=True)
@click.version_option(version="0.1.0", prog_name="ddetect")
@click.pass_context
def main(ctx):
    """DDetector: Automated detection of direction-dependent calibration regions.

    Analyzes residual FITS images from radio interferometric observations
    to identify regions requiring direction-dependent calibration.
    """
    if ctx.invoked_subcommand is None:
        # If no subcommand, show help
        click.echo(ctx.get_help())


@main.command()
@click.argument('input_fits', type=click.Path(exists=True, path_type=Path))
@click.option('-o', '--output', 'output_prefix', default='ddetector_regions',
              help='Output file prefix (default: ddetector_regions)')
@click.option('--format', 'output_format',
              type=click.Choice(['ds9', 'casa', 'kvis', 'all'], case_sensitive=False),
              default='ds9', help='Output region file format(s)')
@click.option('-c', '--config', 'config_file', type=click.Path(path_type=Path),
              help='Configuration JSON file')
@click.option('--preset', type=click.Choice(['lband', 'cband', 'xband', 'wide_field',
                                           'conservative', 'aggressive']),
              help='Use built-in parameter preset')
@click.option('--negative-sigma', type=float, default=None,
              help='Negative outlier threshold in sigma units')
@click.option('--min-size', type=int, default=None,
              help='Minimum region size in pixels')
@click.option('--clustering-eps', type=float, default=None,
              help='DBSCAN clustering epsilon parameter')
@click.option('--save-summary/--no-summary', default=True,
              help='Save analysis summary JSON file')
@click.option('--save-config/--no-config', default=False,
              help='Save used configuration to file')
@click.option('--export-csv/--no-csv', default=False,
              help='Export regions as CSV catalog')
@click.option('--export-mask/--no-mask', default=False,
              help='Export regions as FITS mask')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.option('-q', '--quiet', is_flag=True, help='Suppress non-error output')
@click.option('--dry-run', is_flag=True, help='Show parameters without running analysis')
def detect(input_fits: Path, output_prefix: str, output_format: str,
          config_file: Optional[Path], preset: Optional[str],
          negative_sigma: Optional[float], min_size: Optional[int],
          clustering_eps: Optional[float], save_summary: bool, save_config: bool,
          export_csv: bool, export_mask: bool, verbose: bool, quiet: bool,
          dry_run: bool):
    """Detect direction-dependent calibration regions in a residual image.

    INPUT_FITS: Path to the input residual FITS image.

    Examples:

        # Basic detection with DS9 output
        ddetect residual.fits

        # Use L-band preset with custom output
        ddetect residual.fits --preset lband -o lband_regions

        # Custom parameters with multiple outputs
        ddetect residual.fits --negative-sigma 4.0 --min-size 30 --format all

        # Use configuration file
        ddetect residual.fits -c my_config.json --export-csv
    """
    setup_logging(verbose, quiet)

    try:
        # Load/create detection parameters
        params = _load_detection_parameters(config_file, preset, negative_sigma,
                                          min_size, clustering_eps)

        if dry_run:
            _show_parameters_and_exit(params, input_fits)

        # Run detection
        logger.info(f"Starting DDetector analysis of {input_fits}")
        analyzer = ResidualImageAnalyzer(params)
        regions = analyzer.analyze_residual_image(input_fits)

        # Create detection result
        result = DetectionResult(
            regions=regions,
            image_stats=analyzer.image_stats,
            detection_params=params,
            input_file=str(input_fits)
        )

        # Report results
        _report_detection_results(result)

        if not regions:
            logger.info("No calibration regions detected - no output files created")
            return

        # Write output files
        _write_output_files(result, output_prefix, output_format, save_summary,
                           save_config, export_csv, export_mask, input_fits)

        logger.info("Analysis completed successfully!")

    except Exception as e:
        import ipdb; ipdb.set_trace()
        logger.error(f"Analysis failed: {e}")
        if verbose:
            raise
        sys.exit(1)


@main.command()
@click.argument('config_name')
@click.option('-o', '--output', 'output_file', type=click.Path(path_type=Path),
              help='Output configuration file (default: <config_name>.json)')
def create_config(config_name: str, output_file: Optional[Path]):
    """Create a configuration file from built-in presets.

    CONFIG_NAME: Name of preset configuration (lband, cband, conservative, etc.)

    Examples:

        # Create L-band configuration
        ddetect create-config lband

        # Save to custom file
        ddetect create-config conservative -o my_conservative.json
    """
    try:
        # Create parameters from preset
        base_params = DetectionParameters()
        params = base_params.get_frequency_band_preset(config_name)

        # Determine output file
        if output_file is None:
            output_file = Path(f"{config_name}_config.json")

        # Save configuration
        ConfigurationManager.save_config(
            params, output_file,
            description=f"DDetector preset configuration for {config_name}"
        )

        click.echo(f"Created configuration file: {output_file}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_files', nargs=-1, required=True,
                type=click.Path(exists=True, path_type=Path))
@click.option('-c', '--config', 'config_file', type=click.Path(path_type=Path),
              help='Configuration JSON file')
@click.option('--preset', type=click.Choice(['lband', 'cband', 'xband', 'wide_field',
                                           'conservative', 'aggressive']),
              default='lband', help='Parameter preset to use')
@click.option('-o', '--output-dir', type=click.Path(path_type=Path), default='.',
              help='Output directory for results')
@click.option('--format', 'output_format',
              type=click.Choice(['ds9', 'casa', 'all'], case_sensitive=False),
              default='ds9', help='Output region file format')
@click.option('-v', '--verbose', is_flag=True, help='Enable verbose output')
@click.option('--parallel/--no-parallel', default=True,
              help='Use parallel processing (requires Ray)')
def batch(input_files: List[Path], config_file: Optional[Path], preset: str,
         output_dir: Path, output_format: str, verbose: bool, parallel: bool):
    """Process multiple residual images in batch mode.

    INPUT_FILES: One or more residual FITS images to process.

    Examples:

        # Process all FITS files in current directory
        ddetect batch *.fits

        # Batch process with custom configuration
        ddetect batch img1.fits img2.fits img3.fits -c my_config.json

        # Use parallel processing
        ddetect batch *.fits --parallel --preset cband
    """
    setup_logging(verbose, False)

    try:
        # Load parameters
        if config_file:
            params = ConfigurationManager.load_config(config_file)
        else:
            base_params = DetectionParameters()
            params = base_params.get_frequency_band_preset(preset)

        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)

        # Process files
        if parallel:
            _process_batch_parallel(input_files, params, output_dir, output_format)
        else:
            _process_batch_sequential(input_files, params, output_dir, output_format)

        logger.info(f"Batch processing completed for {len(input_files)} files")

    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        if verbose:
            raise
        sys.exit(1)


@main.command()
@click.argument('config_file', type=click.Path(exists=True, path_type=Path))
def validate_config(config_file: Path):
    """Validate a configuration file.

    CONFIG_FILE: Path to configuration JSON file to validate.
    """
    try:
        params = ConfigurationManager.load_config(config_file)
        click.echo(f"✓ Configuration file '{config_file}' is valid")
        click.echo(f"  Negative threshold: {params.negative_threshold_sigma}σ")
        click.echo(f"  Minimum region size: {params.min_region_size_pixels} pixels")
        click.echo(f"  Clustering epsilon: {params.clustering_eps} pixels")

    except Exception as e:
        click.echo(f"✗ Configuration file '{config_file}' is invalid: {e}", err=True)
        sys.exit(1)


def _load_detection_parameters(config_file: Optional[Path], preset: Optional[str],
                             negative_sigma: Optional[float], min_size: Optional[int],
                             clustering_eps: Optional[float]) -> DetectionParameters:
    """Load detection parameters from various sources with precedence."""

    # Start with defaults
    if config_file:
        params = ConfigurationManager.load_config(config_file)
        logger.info(f"Loaded configuration from: {config_file}")
    elif preset:
        base_params = DetectionParameters()
        params = base_params.get_frequency_band_preset(preset)
        logger.info(f"Using preset configuration: {preset}")
    else:
        params = DetectionParameters()
        logger.info("Using default parameters")

    # Override with command-line arguments
    if negative_sigma is not None:
        params.negative_threshold_sigma = negative_sigma
        logger.info(f"Overriding negative_threshold_sigma: {negative_sigma}")

    if min_size is not None:
        params.min_region_size_pixels = min_size
        logger.info(f"Overriding min_region_size_pixels: {min_size}")

    if clustering_eps is not None:
        params.clustering_eps = clustering_eps
        logger.info(f"Overriding clustering_eps: {clustering_eps}")

    return params


def _show_parameters_and_exit(params: DetectionParameters, input_fits: Path):
    """Show parameters and exit (dry run mode)."""
    click.echo(f"DDetector Analysis Parameters for: {input_fits}")
    click.echo("=" * 50)
    click.echo(f"Negative threshold sigma: {params.negative_threshold_sigma}")
    click.echo(f"Spike threshold percentile: {params.spike_threshold_percentile}")
    click.echo(f"Ripple frequency range: {params.ripple_frequency_range}")
    click.echo(f"Minimum region size: {params.min_region_size_pixels} pixels")
    click.echo(f"Clustering epsilon: {params.clustering_eps} pixels")
    click.echo(f"Edge buffer: {params.edge_buffer_pixels} pixels")
    click.echo("\nUse --no-dry-run to execute analysis")
    sys.exit(0)


def _report_detection_results(result: DetectionResult):
    """Report detection results to user."""
    num_regions = result.num_regions

    if num_regions == 0:
        logger.info("No calibration regions detected")
        return

    logger.info(f"Detection completed: {num_regions} regions found")

    # Count by detection type
    type_counts = {}
    high_confidence = 0

    for region in result.regions:
        for det_type in region.detection_type:
            type_counts[det_type] = type_counts.get(det_type, 0) + 1

        if region.confidence > 0.7:
            high_confidence += 1

    # Report statistics
    logger.info(f"Detection breakdown:")
    for det_type, count in type_counts.items():
        logger.info(f"  {det_type}: {count} regions")

    logger.info(f"High confidence (>0.7): {high_confidence} regions")
    logger.info(f"Total affected area: {result.total_affected_area:.1f} arcsec²")

    # Recommendation
    if high_confidence >= 3 or num_regions >= 5:
        logger.info("💡 Recommendation: Apply direction-dependent calibration")
    elif num_regions >= 2:
        logger.info("💡 Recommendation: Consider direction-dependent calibration")
    else:
        logger.info("💡 Recommendation: DDCal may provide minor improvement")


def _write_output_files(result: DetectionResult, output_prefix: str,
                       output_format: str, save_summary: bool, save_config: bool,
                       export_csv: bool, export_mask: bool, input_fits: Path):
    """Write all requested output files."""
    regions = result.regions

    # Region files
    if output_format in ['ds9', 'all']:
        output_file = f"{output_prefix}.reg"
        write_ds9_region_file(regions, output_file)
        logger.info(f"Wrote DS9 regions: {output_file}")

    if output_format in ['casa', 'all']:
        output_file = f"{output_prefix}.crtf"
        write_casa_region_file(regions, output_file)
        logger.info(f"Wrote CASA regions: {output_file}")

    if output_format in ['kvis', 'all']:
        output_file = f"{output_prefix}.ann"
        RegionFileWriter.write_kvis_region_file(regions, output_file)
        logger.info(f"Wrote KVIS regions: {output_file}")

    # Summary JSON
    if save_summary:
        summary_file = f"{output_prefix}_summary.json"
        ResultsExporter.export_summary_json(result, summary_file)
        logger.info(f"Wrote summary: {summary_file}")

    # Configuration file
    if save_config:
        config_file = f"{output_prefix}_config.json"
        ConfigurationManager.save_config(
            result.detection_params, config_file,
            description=f"Parameters used for {input_fits.name}"
        )
        logger.info(f"Wrote configuration: {config_file}")

    # CSV catalog
    if export_csv:
        csv_file = f"{output_prefix}_catalog.csv"
        ResultsExporter.export_csv_catalog(regions, csv_file)
        logger.info(f"Wrote CSV catalog: {csv_file}")

    # FITS mask
    if export_mask:
        mask_file = f"{output_prefix}_mask.fits"
        ResultsExporter.export_fits_mask(regions, input_fits, mask_file)
        logger.info(f"Wrote FITS mask: {mask_file}")


def _process_batch_sequential(input_files: List[Path], params: DetectionParameters,
                            output_dir: Path, output_format: str):
    """Process files sequentially."""
    results = []

    for i, input_file in enumerate(input_files, 1):
        logger.info(f"Processing {i}/{len(input_files)}: {input_file.name}")

        try:
            analyzer = ResidualImageAnalyzer(params)
            regions = analyzer.analyze_residual_image(input_file)

            result = DetectionResult(
                regions=regions,
                image_stats=analyzer.image_stats,
                detection_params=params,
                input_file=str(input_file)
            )

            # Write outputs
            output_prefix = output_dir / input_file.stem
            _write_output_files(result, str(output_prefix), output_format,
                              True, False, False, False, input_file)

            results.append(result)
            logger.info(f"  → {len(regions)} regions detected")

        except Exception as e:
            logger.error(f"  ✗ Failed to process {input_file.name}: {e}")

    # Write batch summary
    _write_batch_summary(results, output_dir)


def _process_batch_parallel(input_files: List[Path], params: DetectionParameters,
                          output_dir: Path, output_format: str):
    """Process files in parallel using Ray."""
    try:
        import ray
        from .distributed import BatchProcessor

        if not ray.is_initialized():
            ray.init()

        processor = BatchProcessor(params)
        results = processor.process_files(input_files, output_dir, output_format)

        logger.info(f"Parallel processing completed")
        _write_batch_summary(results, output_dir)

    except ImportError:
        logger.warning("Ray not available, falling back to sequential processing")
        _process_batch_sequential(input_files, params, output_dir, output_format)


def _write_batch_summary(results: List[DetectionResult], output_dir: Path):
    """Write summary of batch processing results."""
    summary = {
        'total_files': len(results),
        'total_regions': sum(len(r.regions) for r in results),
        'files_with_detections': sum(1 for r in results if r.regions),
        'results': [
            {
                'file': r.input_file,
                'num_regions': len(r.regions),
                'total_area': r.total_affected_area
            }
            for r in results
        ]
    }

    summary_file = output_dir / "batch_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Wrote batch summary: {summary_file}")


if __name__ == "__main__":
    main()
