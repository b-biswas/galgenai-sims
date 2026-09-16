"""
Tests using the toy catalog (test_catalog_30objects.fits)
This file tests catalog loading, filtering, and galaxy simulation
with a small test dataset

Test catalog composition (30 objects total):
- 20 good galaxies: high SNR, valid magnitudes, valid redshift
- 4 low SNR galaxies: SNR < 50 (they bad mag = 999.0 sentinel value)
- 1 galaxy with invalid redshift: z <= 0
- 5 non-galaxies: Stars with various properties
"""

import numpy as np
from pathlib import Path

from galgenai_sims.config import load_sim_config
from galgenai_sims import COSMOSWebCatalog, GalaxySim

# Load catalog column mapping from config
CONFIG = load_sim_config()
CATALOG_COLUMNS = CONFIG["cosmos"]["catalog_columns"]
TEST_CATALOG_PATH = Path(CONFIG["cosmos"]["catalog_path"])
OUTPUT_DIR = Path(CONFIG["cosmos"]["hf_dataset_path"])


def test_load_all_objects():
    """Test loading all objects without filters"""
    assert TEST_CATALOG_PATH.exists(), (
        f"Test catalog not found at {TEST_CATALOG_PATH}"
    )

    catalog = COSMOSWebCatalog(
        catalog_path=str(TEST_CATALOG_PATH),
        galaxies_only=False,
        filter_invalid_mags=False,
        warn_flag_cut=None,
    )

    # Should have all 30 objects
    assert len(catalog.data) == 30, (
        f"Expected 30 objects, got {len(catalog.data)}"
    )

    # Check we have different object types
    types = set(catalog.data["type"])
    assert 0 in types, "Should have galaxies (type=0)"
    assert len(types) > 1, "Should have non-galaxies (stars/QSOs)"


def test_galaxies_only_filter():
    """Test galaxies_only filter removes stars and QSOs"""
    # Load only galaxies
    catalog_galaxies = COSMOSWebCatalog(
        catalog_path=str(TEST_CATALOG_PATH),
        galaxies_only=True,
        filter_invalid_mags=False,
        warn_flag_cut=None,
    )

    # Should have 25 galaxies (removed 5 non-galaxies)
    assert len(catalog_galaxies.data) == 25, (
        f"Expected 25 galaxies, got {len(catalog_galaxies.data)}"
    )

    # All should be galaxies
    assert np.all(catalog_galaxies.data["type"] == 0), (
        "All objects should be galaxies (type=0)"
    )


def test_invalid_magnitudes_filter():
    """Test filter_invalid_mags removes objects with sentinel values"""
    # Load with filtering
    catalog_filtered = COSMOSWebCatalog(
        catalog_path=str(TEST_CATALOG_PATH),
        galaxies_only=True,
        filter_invalid_mags=True,
        mag_sentinel=999.0,
    )

    # Should remove 3 galaxies with bad magnitudes (25 - 4)
    assert len(catalog_filtered.data) == 21, (
        f"Expected 22 galaxies after magnitude filtering,"
        f" got {len(catalog_filtered.data)}"
    )

    # No magnitudes should be 999.0
    for col in catalog_filtered.data.colnames:
        if col.startswith("mag_model_hsc-"):
            assert not np.any(catalog_filtered.data[col] == 999.0), (
                f"Column {col} should not have sentinel value 999.0"
            )


def test_snr_filtering():
    """Test SNR-based filtering"""
    catalog = COSMOSWebCatalog(
        catalog_path=str(TEST_CATALOG_PATH),
        galaxies_only=True,
        filter_invalid_mags=False,
    )

    # Filter for high SNR (> 50)
    high_snr = catalog.get_galaxies(filters={"snr_hst-f814w": (50, np.inf)})

    # Should get 20 good galaxies + 1 bad_z galaxy with high SNR = 21
    assert len(high_snr) == 21, (
        f"Expected 21 high-SNR galaxies, got {len(high_snr)}"
    )
    assert np.all(high_snr["snr_hst-f814w"] > 50), "All should have SNR > 50"

    # Filter for low SNR (< 50)
    low_snr = catalog.get_galaxies(filters={"snr_hst-f814w": (0, 50)})

    # Should get 4 low SNR galaxies (some with bad mags)
    assert len(low_snr) == 4, (
        f"Expected 4 low-SNR galaxies, got {len(low_snr)}"
    )
    assert np.all(low_snr["snr_hst-f814w"] < 50), "All should have SNR < 50"


def test_simulate_good_galaxy():
    """Test simulating a good quality galaxy"""
    # Load catalog and get a good galaxy
    catalog = COSMOSWebCatalog(
        catalog_path=str(TEST_CATALOG_PATH),
        galaxies_only=True,
        filter_invalid_mags=True,
    )

    good_galaxies = catalog.get_galaxies(
        filters={"snr_hst-f814w": (50, np.inf)}, n_galaxies=1
    )

    assert len(good_galaxies) > 0, (
        "Need at least one good galaxy for simulation"
    )

    # Initialize simulator
    sim = GalaxySim(
        survey_name="HSC",
        image_size=17,
        random_seed=42,
        max_fft_size=512,
        catalog_columns=CATALOG_COLUMNS,
    )

    # Generate image for one galaxy
    galaxy_row = good_galaxies[0]
    filter_names = ["g", "r", "i", "z", "y"]

    images_dict, pixel_variance_dict, noiseless_dict, galaxy_params = (
        sim.generate_image_from_row(galaxy_row, filter_names=filter_names)
    )

    # Verify outputs
    assert len(images_dict) == 5, "Should have 5 bands"
    assert len(pixel_variance_dict) == 5, "Should have 5 variance maps"
    assert len(noiseless_dict) == 5, "Should have 5 noiseless images"
    assert len(galaxy_params) == 5, "Should have 5 parameter sets"

    for band in filter_names:
        assert images_dict[band].shape == (17, 17), (
            "Image shape should be (17, 17)"
        )
        assert pixel_variance_dict[band].shape == (17, 17), (
            "Variance shape should be (17, 17)"
        )
        assert noiseless_dict[band].shape == (17, 17), (
            "Noiseless image shape should be (17, 17)"
        )
        assert np.sum(images_dict[band]) > 0, (
            f"Band {band} should have positive flux"
        )
        assert np.all(pixel_variance_dict[band] >= 0), (
            "Variance should be non-negative"
        )
        assert np.sum(noiseless_dict[band]) > 0, (
            f"Band {band} noiseless should have positive flux"
        )


def test_create_dataset():
    """Test simulating multiple galaxies from test catalog"""
    # Load good galaxies
    catalog = COSMOSWebCatalog(
        catalog_path=str(TEST_CATALOG_PATH),
        galaxies_only=True,
        filter_invalid_mags=True,
    )

    # Initialize simulator
    sim = GalaxySim(
        survey_name="HSC",
        catalog=catalog,
        image_size=17,
        random_seed=42,
        max_fft_size=512,
        snr_threshold=50,
        catalog_columns=CATALOG_COLUMNS,
    )

    sim.create_dataset(
        output_dir=OUTPUT_DIR,
        num_workers=1,
        filter_high_snr=True,
        max_galaxies=10,
    )
