from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from datasets import Dataset


# TODO: CHECK catalog what the sentinel values are
def load_fits_dataset(
    data_dir,
    metadata_file="metadata.csv",
    format="torch",
    filter_invalid_mags=True,
    mag_sentinel=999.0,
    mag_cols=None,
    filter_invalid_redshift=True,
    redshift_sentinel=-99.0,
    redshift_col=None,
    nx=None,
    load_noiseless=False,
):
    """
    Load a FITS galaxy dataset produced by generate_fits_dataset.py.

    All galaxies are stored together under ``data_dir/images/`` with
    metadata CSV file(s). The returned dataset is then split in
    make_loaders() (available in the main galgenai package).

    The returned dataset has one column image (nested dict with keys:
    flux, ivar, mask, band, and optionally noiseless) plus all metadata
    columns from the CSV. This layout matches the HSC dataset format so
    that hsc.HSCDataset can be used directly.

    If IVAR is absent from a FITS file, a ones array is used
    (uniform weighting). If MASK is absent, a zeros array is used
    (no masking).

    Parameters:
    -----------
    data_dir : str or Path
        Root directory of the dataset
        (contains ``images/`` and metadata CSV).
    metadata_file : str
        Name of the metadata CSV file. Default "metadata.csv".
    format : str
        Output format for arrays. Options: "torch" (default), "numpy",
        "tensorflow", or None (Python lists). Default "torch".
    filter_invalid_mags : bool
        If True, filter out galaxies where any magnitude column equals
        the sentinel value. Default True.
    mag_sentinel : float
        Sentinel value indicating missing magnitude (default: 999.0).
        Rows with any magnitude exactly equal to this value will be
        filtered out.
    mag_cols : list of str or None
        List of magnitude column names to check (e.g.,
        ['mag_g', 'mag_r', 'mag_i']). If None, auto-detects all columns
        starting with 'mag_'. Default None.
    filter_invalid_redshift : bool
        If True, filter out galaxies where redshift equals the sentinel
        value. Default True.
    redshift_sentinel : float
        Sentinel value indicating missing redshift (default: -99.0).
    redshift_col : str
        Name of the redshift column in metadata. Required if
        filter_invalid_redshift is True.
    nx : int or None
        Optional crop size. If provided, images will be center-cropped
        to nx x nx. If None, images are loaded at their original size.
        Default None.
    load_noiseless : bool
        If True, load noiseless galaxy images from NOISELESS HDU.
        If False, noiseless images are not loaded. Default False.

    Returns:
    --------
    datasets.Dataset
        HuggingFace Dataset with PyTorch tensors
        (default format="torch").
    """
    data_dir = Path(data_dir)
    images_path = data_dir / "images"
    metadata = pd.read_csv(data_dir / metadata_file)

    # Filter out galaxies with invalid magnitudes
    if filter_invalid_mags:
        initial_count = len(metadata)

        # Determine which columns to check
        if mag_cols is None:
            # Auto-detect magnitude columns
            raise ValueError("Mag col names should be provided to apply cuts")

        if mag_cols:
            # Create mask for valid magnitudes
            mask = np.ones(len(metadata), dtype=bool)
            for mag_col in mag_cols:
                if mag_col in metadata.columns:
                    # Filter out rows where magnitude equals
                    # sentinel value
                    col_mask = metadata[mag_col] <= mag_sentinel
                    mask &= col_mask

            metadata = metadata[mask].reset_index(drop=True)
            n_removed = initial_count - len(metadata)
            if n_removed > 0:
                print(
                    f"Filtered {n_removed} galaxies with "
                    f"invalid magnitudes "
                    f"(sentinel={mag_sentinel})"
                )
                print(f"Remaining: {len(metadata)} galaxies")

    if filter_invalid_redshift:
        if redshift_col is None:
            raise ValueError(
                "Redshift col name should be provided to apply cuts"
            )

        if redshift_col in metadata.columns:
            col_mask = metadata[redshift_col] != redshift_sentinel
            metadata = metadata[col_mask].reset_index(drop=True)
            n_removed = len(col_mask) - len(metadata)
            if n_removed > 0:
                print(
                    f"Filtered {n_removed} galaxies with "
                    f"invalid redshift "
                    f"(sentinel={redshift_sentinel})"
                )
                print(f"Remaining: {len(metadata)} galaxies")
        else:
            raise ValueError(
                f"Warning: Redshift column '{redshift_col}' not "
                f"found in metadata. Skipping redshift filtering."
            )

    # Check for Arrow cache (memory-mappable,
    # avoids reopening FITS files)
    from datasets import load_from_disk

    # Cache path includes crop size and noiseless flag
    cache_suffix = ""
    if nx is not None:
        cache_suffix += f"_nx{nx}"
    if load_noiseless:
        cache_suffix += "_noiseless"

    cache_name = (
        f"arrow_cache{cache_suffix}" if cache_suffix else "arrow_cache_raw"
    )
    cache_path = data_dir / cache_name

    if cache_path.exists():
        print(f"Loading from Arrow cache: {cache_path}")
        dataset = load_from_disk(str(cache_path))
    else:
        print(
            f"Arrow cache not found. Loading {len(metadata):,} FITS files..."
        )
        print("This ONE-TIME operation will take a few minutes.")

        if nx is not None:
            print(
                f"Images will be center-cropped to {nx}x{nx} during caching."
            )
        else:
            print("Images will be cached at their original size.")

        from tqdm import tqdm

        n_total = len(metadata)

        # Get original image size and band names from first FITS file
        first_row = metadata.iloc[0]
        with fits.open(images_path / first_row["filename"]) as hdul:
            orig_shape = hdul["IMAGE"].data.shape
            og_h, og_w = orig_shape[1], orig_shape[2]

            # Extract band names from FITS header (same for all images)
            n_bands = orig_shape[0]
            bands = [
                hdul["IMAGE"].header.get(f"BAND{i}", f"band{i}")
                for i in range(n_bands)
            ]

        if nx is not None:
            # Calculate crop indices
            og_nx2, og_ny2 = og_h // 2, og_w // 2
            nx2 = nx // 2
            print(f"  - Original size: {og_h}x{og_w}, cropped to: {nx}x{nx}")

        # Process all samples in ONE pass
        # (no concatenation = no fragmentation)
        print(f"Processing {n_total:,} samples...")
        all_samples = []

        for i in tqdm(range(n_total), desc="Loading FITS files"):
            row = metadata.iloc[i]
            with fits.open(images_path / row["filename"]) as hdul:
                # Load raw data
                flux = hdul["IMAGE"].data.astype("float32")

                if "IVAR" in hdul:
                    ivar = hdul["IVAR"].data.astype("float32")
                else:
                    ivar = np.ones_like(flux)

                if "MASK" in hdul:
                    mask = hdul["MASK"].data.astype(np.int32)
                else:
                    mask = np.zeros(flux.shape, dtype=np.int32)

                if load_noiseless and "NOISELESS" in hdul:
                    noiseless = hdul["NOISELESS"].data.astype("float32")
                else:
                    noiseless = None

            # Crop to target size if nx is provided
            if nx is not None:
                flux = flux[
                    :, og_nx2 - nx2 : og_nx2 + nx2, og_ny2 - nx2 : og_ny2 + nx2
                ]
                ivar = ivar[
                    :, og_nx2 - nx2 : og_nx2 + nx2, og_ny2 - nx2 : og_ny2 + nx2
                ]
                mask = mask[
                    :, og_nx2 - nx2 : og_nx2 + nx2, og_ny2 - nx2 : og_ny2 + nx2
                ]
                if noiseless is not None:
                    noiseless = noiseless[
                        :,
                        og_nx2 - nx2 : og_nx2 + nx2,
                        og_ny2 - nx2 : og_ny2 + nx2,
                    ]

            sample = {
                "image": {
                    "flux": flux,
                    "ivar": ivar,
                    "mask": mask,
                    "band": bands,
                }
            }
            if noiseless is not None:
                sample["image"]["noiseless"] = noiseless

            sample.update(row.to_dict())

            all_samples.append(sample)

        print("Creating Arrow cache...")
        dataset = Dataset.from_list(all_samples)
        del all_samples

        # Save to disk with sharding for optimal memory mapping
        print(f"Saving to: {cache_path}")
        dataset.save_to_disk(str(cache_path))
        cache_size_mb = (
            sum(f.stat().st_size for f in cache_path.rglob("*") if f.is_file())
            / 1e6
        )
        print(
            f"Cached! ({cache_size_mb:.1f} MB) Future loads will be instant."
        )

    # Apply format if specified
    if format is not None:
        dataset = dataset.with_format(format)

    return dataset
