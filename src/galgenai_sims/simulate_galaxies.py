"""
Galaxy Image Simulator using Galsim
Simulates HSC and HST observations of galaxies from the
COSMOSWeb catalog
"""

import csv
import gc
import numpy as np
import galsim
from pathlib import Path
from astropy.io import fits
from tqdm import tqdm
from multiprocessing import Pool
from surveycodex import get_survey
from surveycodex.utilities import mag2counts, mean_sky_level


def sim_single_band_sersic_galaxy(
    flux, hlr, sersic_n, axis_ratio, position_angle, gsparams=None
):
    """
    Create galaxy light profile from catalog parameters

    Parameters:
    -----------
    flux : float
        Flux in photons
    hlr : float
        Half-light radius in arcsec
    sersic_n : float
        Sersic index
    axis_ratio : float
        b/a ratio (default: 1.0 for circular)
    position_angle : float
        PA in degrees
    gsparams : galsim.GSParams, optional
        GSParams object for FFT settings

    Returns:
    --------
    galsim.GSObject
        Galaxy profile
    """
    # Create Sersic profile with GSParams
    gal = galsim.Sersic(
        n=sersic_n, half_light_radius=hlr, flux=flux, gsparams=gsparams
    )

    # Apply shear for ellipticity
    if axis_ratio < 1.0:
        g = (1 - axis_ratio) / (1 + axis_ratio)  # reduced shear
        gal = gal.shear(g=g, beta=position_angle * galsim.degrees)

    return gal


def sim_single_band_bulge_disk_galaxy(
    flux_bulge,
    flux_disk,
    hlr_bulge,
    hlr_disk,
    bulge_axratio,
    disk_axratio,
    position_angle,
    gsparams=None,
):
    """
    Create bulge+disk galaxy light profile from catalog parameters

    The disk is modeled as an Exponential profile (Sersic n=1).
    The bulge is modeled as a DeVaucouleurs profile (Sersic n=4).

    If flux_disk is 0, simulates bulge-only galaxy (no disk component).

    Parameters:
    -----------
    flux_bulge : float
        Bulge flux in photons
    flux_disk : float
        Disk flux in photons (set to 0 for bulge-only)
    hlr_bulge : float
        Bulge half-light radius in arcsec
    hlr_disk : float
        Disk half-light radius in arcsec
    bulge_axratio : float
        Bulge axis ratio (b/a)
    disk_axratio : float
        Disk axis ratio (b/a)
    position_angle : float
        Position angle in degrees (applied to combined bulge+disk)
    gsparams : galsim.GSParams, optional
        GSParams object for FFT settings

    Returns:
    --------
    galsim.GSObject
        Combined bulge+disk galaxy profile (or bulge-only if flux_disk=0)
    """
    # Create bulge (DeVaucouleurs profile, Sersic n=4)
    bulge = galsim.DeVaucouleurs(
        half_light_radius=hlr_bulge, flux=flux_bulge, gsparams=gsparams
    )

    # Apply ellipticity to bulge
    # Clip axis ratio to valid range [0, 1] to handle catalog edge cases
    q_bulge = np.clip(bulge_axratio, 0.0, 1.0)
    g_bulge = (1.0 - q_bulge) / (1.0 + q_bulge)
    bulge = bulge.shear(g=g_bulge, beta=0.0 * galsim.degrees)

    # Check if we have a disk component
    if flux_disk > 0:
        # Create disk component (Exponential profile, Sersic n=1)
        disk = galsim.Exponential(
            half_light_radius=hlr_disk, flux=flux_disk, gsparams=gsparams
        )

        # Apply ellipticity to disk
        # Clip axis ratio to valid range [0, 1] to handle catalog edge cases
        q_disk = np.clip(disk_axratio, 0.0, 1.0)
        g_disk = (1.0 - q_disk) / (1.0 + q_disk)
        disk = disk.shear(g=g_disk, beta=0.0 * galsim.degrees)

        # Combine bulge and disk
        gal = galsim.Add([bulge, disk])
    else:
        # Bulge-only galaxy
        gal = bulge

    # Rotate by position angle
    gal = gal.rotate(position_angle * galsim.degrees)

    return gal


def _save_galaxy_fits(
    image_array, var_array, noiseless_array, path, filter_names
):
    """
    Save image and inverse-variance arrays as a FITS file.

    HDU[0] (PrimaryHDU) : empty; header documents the file layout
    HDU[1] (ImageHDU)   : image,  shape (N_bands, H, W), float32
    HDU[2] (ImageHDU)   : ivar,   shape (N_bands, H, W), float32
                          Inverse-variance: 1 / pixel_variance.
    HDU[3] (ImageHDU)   : mask,   shape (N_bands, H, W), uint32
                          Bitmask; 0 = unmasked.
    HDU[4] (ImageHDU)   : noiseless, shape (N_bands, H, W), float32
                          Noiseless galaxy image.

    Parameters
    ----------
    image_array : np.ndarray, shape (N_bands, H, W), dtype float32
    var_array : np.ndarray, shape (N_bands, H, W), dtype float32
        Per-pixel Poisson variance (galaxy signal +
        sky background counts).
    noiseless_array : np.ndarray, shape (N_bands, H, W), dtype float32
        Noiseless galaxy image.
    path : Path or str
    filter_names : list of str
        Band labels written into the FITS headers
        (BAND0, BAND1, ...).
    """
    primary = fits.PrimaryHDU()
    primary.header["COMMENT"] = "HDU[1] = IMAGE  (N_bands, H, W) float32"
    primary.header["COMMENT"] = (
        "HDU[2] = IVAR   (N_bands, H, W) float32  1/variance"
    )
    primary.header["COMMENT"] = (
        "HDU[3] = MASK   (N_bands, H, W) uint32   0=unmasked"
    )
    primary.header["COMMENT"] = "HDU[4] = NOISELESS (N_bands, H, W) float32"
    primary.header["NBANDS"] = (
        len(filter_names),
        "Number of photometric bands",
    )
    for i, name in enumerate(filter_names):
        primary.header[f"BAND{i}"] = name

    hdu_image = fits.ImageHDU(image_array, name="IMAGE")
    hdu_image.header["BUNIT"] = "electron/s"
    for i, name in enumerate(filter_names):
        hdu_image.header[f"BAND{i}"] = name

    ivar_array = np.where(var_array > 0, 1.0 / var_array, 0.0).astype(
        np.float32
    )
    hdu_ivar = fits.ImageHDU(ivar_array, name="IVAR")
    hdu_ivar.header["BUNIT"] = "(electron/s)^-2"
    hdu_ivar.header["COMMENT"] = "Inverse variance: 1/var, zero where var<=0"
    for i, name in enumerate(filter_names):
        hdu_ivar.header[f"BAND{i}"] = name

    mask_array = np.zeros(image_array.shape, dtype=np.uint32)
    hdu_mask = fits.ImageHDU(mask_array, name="MASK")
    hdu_mask.header["BUNIT"] = "bitmask"
    hdu_mask.header["COMMENT"] = "0 = unmasked; no bits set (simulated data)"
    for i, name in enumerate(filter_names):
        hdu_mask.header[f"BAND{i}"] = name

    hdu_noiseless = fits.ImageHDU(noiseless_array, name="NOISELESS")
    hdu_noiseless.header["BUNIT"] = "electron/s"

    fits.HDUList(
        [primary, hdu_image, hdu_ivar, hdu_mask, hdu_noiseless]
    ).writeto(str(path), overwrite=True)


def _process_chunk(
    galaxy_rows_chunk,
    images_path,
    filter_names,
    sim_kwargs,
    worker_seed,
    show_progress=False,
):
    """
    Worker function: creates GalaxySim and generates images for chunk.

    Generates images for a chunk of galaxy rows, writes FITS files,
    and returns metadata rows. Each worker has an independent RNG.

    Parameters
    ----------
    galaxy_rows_chunk : list of dict
    images_path : str
        Directory where FITS files are written (passed as str for
        pickling).
    filter_names : list of str
    sim_kwargs : dict
        Keyword arguments forwarded to GalaxySim (survey_name,
        image_size).
    worker_seed : int
        Per-worker random seed for independent RNG stream.

    Returns
    -------
    tuple (list of dict, int)
        (metadata_rows, failed_count)
    """
    images_path = Path(images_path)
    sim = GalaxySim(**sim_kwargs, random_seed=worker_seed)
    catalog_columns = sim.catalog_columns

    metadata_rows = []
    failed_count = 0

    iterator = (
        tqdm(galaxy_rows_chunk, desc="Galaxies", leave=True)
        if show_progress
        else galaxy_rows_chunk
    )
    for i, gr in enumerate(iterator):
        try:
            images_dict, pixel_variance_dict, noiseless_dict, galaxy_params = (
                sim.generate_image_from_row(gr, filter_names)
            )

            image_array = np.stack(
                [images_dict[b] for b in filter_names], axis=0
            ).astype(np.float32)
            var_array = np.stack(
                [pixel_variance_dict[b] for b in filter_names], axis=0
            ).astype(np.float32)
            noiseless_array = np.stack(
                [noiseless_dict[b] for b in filter_names], axis=0
            ).astype(np.float32)

            galaxy_id = int(gr[catalog_columns["galid"]])
            filename = f"galaxy_{galaxy_id}.fits"
            _save_galaxy_fits(
                image_array,
                var_array,
                noiseless_array,
                images_path / filename,
                filter_names,
            )

            # Get the first filter's params (geometry is
            # same across all filters)
            first_filter_params = galaxy_params[filter_names[0]]

            # Build metadata dict based on galaxy type
            metadata_dict = {
                "filename": filename,
                catalog_columns["galid"]: galaxy_id,
                catalog_columns["ra"]: float(gr[catalog_columns["ra"]]),
                catalog_columns["dec"]: float(gr[catalog_columns["dec"]]),
                catalog_columns["snr"]: float(gr[catalog_columns["snr"]]),
                catalog_columns["redshift_col"]: float(
                    gr[catalog_columns["redshift_col"]]
                ),
            }

            # Add magnitude and morphology columns based on galaxy type
            if sim.galaxy_type == "sersic":
                # Single Sersic profile: save total magnitudes and Sersic params
                metadata_dict.update({
                    next(
                        col
                        for col in catalog_columns["mag_cols"]
                        if col.endswith(b)
                    ): galaxy_params[b]["mag"]
                    for b in filter_names
                })
                metadata_dict.update({
                    catalog_columns["hlr"]: first_filter_params["hlr"],
                    catalog_columns["sersic_n"]: first_filter_params["sersic_n"],
                    catalog_columns["sersic_ratio"]: first_filter_params["sersic_ratio"],
                    catalog_columns["sersic_angle"]: first_filter_params["sersic_angle"],
                })
            elif sim.galaxy_type == "bulge+disk":
                # Bulge+disk: save separate bulge and disk magnitudes and params
                metadata_dict.update({
                    next(
                        col
                        for col in catalog_columns["mag_bulge_cols"]
                        if col.endswith(b)
                    ): galaxy_params[b]["mag_bulge"]
                    for b in filter_names
                })
                metadata_dict.update({
                    next(
                        col
                        for col in catalog_columns["mag_disk_cols"]
                        if col.endswith(b)
                    ): galaxy_params[b]["mag_disk"]
                    for b in filter_names
                })
                metadata_dict.update({
                    catalog_columns["radius_bulge"]: first_filter_params["hlr_bulge"],
                    catalog_columns["radius_disk"]: first_filter_params["hlr_disk"],
                    catalog_columns["bulge_axratio"]: first_filter_params["bulge_axratio"],
                    catalog_columns["disk_axratio"]: first_filter_params["disk_axratio"],
                    catalog_columns["angle_bd"]: first_filter_params["position_angle"],
                })

            metadata_rows.append(metadata_dict)

        except Exception as e:
            print(f"\nWarning: Failed galaxy {gr.get('id', '?')}: {e}")
            failed_count += 1

        if (i + 1) % 500 == 0:
            gc.collect()

    return metadata_rows, failed_count


def _process_chunk_args(args):
    """Adapter so pool.imap_unordered can call _process_chunk with a
    tuple."""
    return _process_chunk(*args)


class GalaxySim:
    """Galaxy image simulator using Galsim"""

    def __init__(
        self,
        catalog=None,
        survey_name="HSC",
        image_size=53,
        random_seed=None,
        max_fft_size=512,
        catalog_columns=None,
        snr_threshold=50,
        galaxy_type="sersic",
        disk_mag_threshold=50.0,
    ):
        """
        Initialize the simulator

        Parameters
        ----------
        catalog : COSMOSWebCatalog, optional
            Galaxy catalog to use (default: None)
        survey_name : str, optional
            Survey name for pixel scale and filter definitions
            (default: 'HSC')
        image_size : int, optional
            Size of simulated images in pixels (default: 53)
        random_seed : int, optional
            Random seed for reproducibility. If None, uses 12345
            (default: None)
        max_fft_size : int, optional
            Maximum FFT size for Galsim operations (default: 512)
        catalog_columns : dict, optional
            Mapping of parameter names to catalog column names
            (should include 'mag_cols' and 'snr' keys)
        snr_threshold : float, optional
            Minimum SNR threshold for filtering galaxies (default: 50)
        galaxy_type : str, optional
            Type of galaxy profile: 'sersic' or 'bulge+disk'
            (default: 'sersic')
        disk_mag_threshold : float, optional
            For bulge+disk mode: clip disk magnitudes to this value.
            If disk mag >= threshold, simulate bulge-only (no disk).
            This parameter is required because of bulge COSMOS-Web has 999
            as a sentinel value for no detection but for the disk component
            there are extremely large values for non-detection.
        """
        self.catalog = catalog
        self.survey = get_survey(survey_name=survey_name)
        self.survey_name = survey_name
        self.image_size = image_size
        self.random_seed = random_seed
        self.max_fft_size = max_fft_size
        self.catalog_columns = catalog_columns
        self.snr_threshold = snr_threshold
        self.galaxy_type = galaxy_type
        self.disk_mag_threshold = disk_mag_threshold
        self.rng = galsim.BaseDeviate(random_seed or 12345)
        self.gsparams = galsim.GSParams(maximum_fft_size=max_fft_size)

    def get_psf(self, psf_type, psf_params, gsparams=None):
        """
        Create PSF for the instrument

        Parameters:
        -----------
        psf_type : str
            Type of PSF ('moffat' or 'gaussian')
        psf_params : dict
            PSF parameters (fwhm required, beta for Moffat)
        gsparams : galsim.GSParams, optional
            GSParams object for FFT settings

        Returns:
        --------
        galsim.GSObject
            PSF profile
        """
        if "fwhm" not in psf_params.keys():
            raise ValueError(
                "fwhm parameter is required for psf. Add it to psf_params."
            )
        if psf_type == "moffat":
            if "beta" not in psf_params.keys():
                raise ValueError(
                    "beta parameter is required for moffat psf. "
                    "Add it to psf_params."
                )
            psf = galsim.Moffat(
                beta=psf_params["beta"],
                fwhm=psf_params["fwhm"],
                gsparams=gsparams,
            )
        elif psf_type == "gaussian":
            psf = galsim.Gaussian(
                fwhm=psf_params["fwhm"],
                gsparams=gsparams,
            )
        else:
            raise ValueError("not yet implimented")

        return psf

    def simulate_galaxy_single_band(
        self,
        galaxy_params_filter,
        psf_params_filter,
        filter_name,
        psf_type="moffat",
        add_noise=None,
        galaxy_type="sersic",
        gsparams=None,
    ):
        """
        Simulate a single-band galaxy image

        Parameters:
        -----------
        galaxy_params_filter : dict
            Galaxy parameters. For 'sersic' type, required keys:
            'mag', 'hlr', 'sersic_n', 'sersic_ratio', 'sersic_angle'.
            For 'bulge+disk' type, required keys:
            'mag_bulge', 'mag_disk', 'hlr_bulge', 'hlr_disk',
            'bulge_axratio', 'disk_axratio', 'position_angle'
        psf_params_filter : dict
            PSF parameters with keys 'fwhm' (required) and 'beta' (for
            Moffat PSF)
        filter_name : str
            Name of the filter band (e.g., 'g', 'r', 'i')
        psf_type : str, optional
            Type of PSF to use: 'moffat' or 'gaussian'
            (default: 'moffat')
        add_noise : str or None, optional
            Type of noise: 'galaxy', 'background', 'all', or None
            (default: None)
        galaxy_type : str, optional
            Type of galaxy profile: 'sersic' or 'bulge+disk'
            (default: 'sersic')
        gsparams : galsim.GSParams, optional
            GSParams for FFT settings (default: None)
        """
        if gsparams is None:
            gsparams = self.gsparams

        # Get filter object and compute sky level
        filter = self.survey.get_filter(filter_name)
        sky_level = mean_sky_level(self.survey, filter).to_value("electron")

        # Create galaxy profile
        if galaxy_type == "sersic":
            # Get galaxy flux
            gal_flux = mag2counts(
                galaxy_params_filter["mag"], survey=self.survey, filter=filter
            )
            galaxy = sim_single_band_sersic_galaxy(
                flux=gal_flux.value,
                hlr=galaxy_params_filter["hlr"],
                sersic_n=galaxy_params_filter["sersic_n"],
                axis_ratio=galaxy_params_filter["sersic_ratio"],
                position_angle=galaxy_params_filter["sersic_angle"],
                gsparams=gsparams,
            )
        elif galaxy_type == "bulge+disk":
            # Get bulge flux from magnitude
            bulge_flux = mag2counts(
                galaxy_params_filter["mag_bulge"], survey=self.survey, filter=filter
            )

            # Check if disk magnitude is at/above threshold (no detectable disk)
            # If so, simulate bulge-only by setting disk flux to zero
            if galaxy_params_filter["mag_disk"] >= self.disk_mag_threshold:
                disk_flux_value = 0.0
            else:
                disk_flux = mag2counts(
                    galaxy_params_filter["mag_disk"], survey=self.survey, filter=filter
                )
                disk_flux_value = disk_flux.value

            galaxy = sim_single_band_bulge_disk_galaxy(
                flux_bulge=bulge_flux.value,
                flux_disk=disk_flux_value,
                hlr_bulge=galaxy_params_filter["hlr_bulge"],
                hlr_disk=galaxy_params_filter["hlr_disk"],
                bulge_axratio=galaxy_params_filter["bulge_axratio"],
                disk_axratio=galaxy_params_filter["disk_axratio"],
                position_angle=galaxy_params_filter["position_angle"],
                gsparams=gsparams,
            )
        else:
            raise ValueError(
                f"galaxy_type should be either sersic or"
                f" bulge+disk, got {galaxy_type}"
            )

        # Create PSF
        psf = self.get_psf(psf_type, psf_params_filter, gsparams=gsparams)

        # Convolve galaxy with PSF (both already have gsparams)
        gal_conv = galsim.Convolve([galaxy, psf], gsparams=gsparams)

        # Draw noiseless image (expected counts per
        # pixel, used for ivar)
        image = gal_conv.drawImage(
            nx=self.image_size,
            ny=self.image_size,
            scale=self.survey.pixel_scale.to_value("arcsec"),
        )

        noiseless_image = image.array.copy()

        # Compute per-pixel inverse variance before adding noise.
        # For Poisson statistics, variance = expected counts.
        pixel_variance = np.zeros(image.array.shape, dtype=np.float32)

        if add_noise in ["galaxy", "all"]:
            galaxy_noise = galsim.PoissonNoise(rng=self.rng, sky_level=0.0)
            image.addNoise(galaxy_noise)
            pixel_variance += image.array

        if add_noise in ["background", "all"]:
            background_noise = galsim.PoissonNoise(
                rng=self.rng, sky_level=sky_level
            )
            noise_image = galsim.Image(self.image_size, self.image_size)
            noise_image.addNoise(background_noise)
            image += noise_image
            pixel_variance += sky_level

        return image, pixel_variance, noiseless_image

    def simulate_galaxy(
        self,
        galaxy_params_multiband,
        psf_params_multiband,
        psf_type="moffat",
        add_noise=None,
        galaxy_type="sersic",
        gsparams=None,
    ):
        """
        Simulate multiband galaxy observations.

        Does not use catalog col names so it is
        catalog-independent.

        Parameters:
        -----------
        galaxy_params_multiband : dict
            Dictionary with filter names as keys and galaxy parameters
            as values. Each filter's parameters must use standardized
            keys:
            For 'sersic' profiles:
            - 'mag': magnitude in the band
            - 'hlr': half-light radius in arcsec
            - 'sersic_n': Sersic index
            - 'sersic_ratio': axis ratio (b/a)
            - 'sersic_angle': position angle in degrees
            For 'bulge+disk' profiles:
            - 'mag_bulge': bulge magnitude in the band
            - 'mag_disk': disk magnitude in the band
            - 'hlr_bulge': bulge half-light radius in arcsec
            - 'hlr_disk': disk half-light radius in arcsec
            - 'bulge_axratio': bulge axis ratio (b/a)
            - 'disk_axratio': disk axis ratio (b/a)
            - 'position_angle': position angle in degrees
        psf_params_multiband : dict
            Dictionary with filter names as keys and PSF parameters as
            values. Each filter's parameters should contain 'fwhm'
            (required) and 'beta' (for Moffat PSF).
        psf_type : str, optional
            Type of PSF to use: 'moffat' or 'gaussian'
            (default: 'moffat')
        add_noise : str or None, optional
            Type of noise: 'galaxy', 'background', 'all', or None
            (default: None)
        galaxy_type : str, optional
            Type of galaxy profile: 'sersic' or 'bulge+disk'
            (default: 'sersic')
        gsparams : galsim.GSParams, optional
            GSParams for FFT settings (default: None)

        Returns:
        --------
        tuple : (image, pixel var, noiseless image)
            - multi_band_image: Dictionary of galsim.Image objects
              keyed by filter name
            - multi_band_pixel_variance: Dictionary of variance arrays
              keyed by filter name
            - multi_band_noiseless: Dictionary of noiseless image arrays
              keyed by filter name
        """
        if add_noise is not None:
            if add_noise not in ["galaxy", "background", "all"]:
                raise ValueError(
                    "add_noise must be galaxy/background/all or "
                    f"None, got {add_noise}"
                )

        if gsparams is None:
            gsparams = self.gsparams

        multi_band_image = {}
        multi_band_pixel_variance = {}
        multi_band_noiseless = {}

        for filter_name, band_params in galaxy_params_multiband.items():
            if filter_name not in self.survey.available_filters:
                raise ValueError(f"Filter '{filter_name}' not found in Survey")

            if filter_name not in psf_params_multiband:
                raise ValueError(
                    f"PSF params for filter '{filter_name}' not "
                    "provided in psf_params_multiband"
                )

            image, pixel_variance, noiseless_image = (
                self.simulate_galaxy_single_band(
                    galaxy_params_filter=band_params,
                    psf_params_filter=psf_params_multiband[filter_name],
                    filter_name=filter_name,
                    psf_type=psf_type,
                    add_noise=add_noise,
                    galaxy_type=galaxy_type,
                    gsparams=gsparams,
                )
            )

            multi_band_image[filter_name] = image
            multi_band_pixel_variance[filter_name] = pixel_variance
            multi_band_noiseless[filter_name] = noiseless_image

        return (
            multi_band_image,
            multi_band_pixel_variance,
            multi_band_noiseless,
        )

    def generate_image_from_row(self, galaxy_row, filter_names=None, galaxy_type=None):
        """
        Generate multi-band images from a catalog row.

        Parameters:
        -----------
        galaxy_row : astropy.table.Row
            Single row from catalog
        filter_names : list, optional
            List of filter names to simulate. If None, uses all
            available filters
        galaxy_type : str, optional
            Type of galaxy profile: 'sersic' or 'bulge+disk'.
            If None, uses self.galaxy_type (default: None)

        Returns:
        --------
        tuple : (images_dict, pixel_variance_dict, noiseless_dict,
                 galaxy_params_multi_band)
            - images_dict: Dictionary with filter names as keys and
              image arrays as values
            - pixel_variance_dict: Dictionary with filter names as keys
              and variance arrays as values
            - noiseless_dict: Dictionary with filter names as keys and
              noiseless image arrays as values
            - galaxy_params_multi_band: Dictionary with galaxy
              parameters for each filter
        """
        if filter_names is None:
            filter_names = self.survey.available_filters

        if galaxy_type is None:
            galaxy_type = self.galaxy_type

        galaxy_params_multi_band = {}
        psf_params_multi_band = {}

        for filter_name in filter_names:
            filter_obj = self.survey.get_filter(filter_name)

            # Extract from catalog using catalog column names
            # but create dict with standardized parameter names
            if galaxy_type == "sersic":
                # Find the magnitude column that corresponds to this filter
                mag_col = next(
                    col
                    for col in self.catalog_columns["mag_cols"]
                    if col.endswith(filter_name)
                )
                galaxy_params_multi_band[filter_name] = {
                    "mag": float(galaxy_row[mag_col]),
                    "hlr": float(galaxy_row[self.catalog_columns["hlr"]] * 3600),
                    "sersic_n": float(
                        galaxy_row[self.catalog_columns["sersic_n"]]
                    ),
                    "sersic_ratio": float(
                        galaxy_row[self.catalog_columns["sersic_ratio"]]
                    ),
                    "sersic_angle": float(
                        galaxy_row[self.catalog_columns["sersic_angle"]]
                    ),
                }
            elif galaxy_type == "bulge+disk":
                # Find the bulge and disk magnitude columns for this filter
                mag_bulge_col = next(
                    col
                    for col in self.catalog_columns["mag_bulge_cols"]
                    if col.endswith(filter_name)
                )
                mag_disk_col = next(
                    col
                    for col in self.catalog_columns["mag_disk_cols"]
                    if col.endswith(filter_name)
                )
                # Clip disk magnitude to threshold to handle sentinel values
                # (e.g., 1e308 for galaxies with no detectable disk)
                disk_mag_raw = float(galaxy_row[mag_disk_col])
                disk_mag_clipped = min(disk_mag_raw, self.disk_mag_threshold)

                galaxy_params_multi_band[filter_name] = {
                    "mag_bulge": float(galaxy_row[mag_bulge_col]),
                    "mag_disk": disk_mag_clipped,
                    "hlr_bulge": float(
                        galaxy_row[self.catalog_columns["radius_bulge"]] * 3600
                    ),
                    "hlr_disk": float(
                        galaxy_row[self.catalog_columns["radius_disk"]] * 3600
                    ),
                    "bulge_axratio": float(
                        galaxy_row[self.catalog_columns["bulge_axratio"]]
                    ),
                    "disk_axratio": float(
                        galaxy_row[self.catalog_columns["disk_axratio"]]
                    ),
                    "position_angle": float(
                        galaxy_row[self.catalog_columns["angle_bd"]]
                    ),
                }
            else:
                raise ValueError(
                    f"galaxy_type should be either 'sersic' or "
                    f"'bulge+disk', got {galaxy_type}"
                )

            psf_params_multi_band[filter_name] = {
                "fwhm": filter_obj.psf_fwhm.value,
                "beta": 3.0,
            }

        # Generate images
        multi_band_images, multi_band_pixel_variance, multi_band_noiseless = (
            self.simulate_galaxy(
                galaxy_params_multi_band,
                psf_params_multi_band,
                psf_type="moffat",
                add_noise="all",
                galaxy_type=galaxy_type,
            )
        )

        images_dict = {
            band: img.array.copy() for band, img in multi_band_images.items()
        }
        pixel_variance_dict = {
            band: pv.copy() for band, pv in multi_band_pixel_variance.items()
        }
        noiseless_dict = {
            band: noiseless.copy()
            for band, noiseless in multi_band_noiseless.items()
        }
        return (
            images_dict,
            pixel_variance_dict,
            noiseless_dict,
            galaxy_params_multi_band,
        )

    def filter_high_snr_galaxies(self, inplace=True):
        """
        Filter galaxies with high SNR.

        Uses self.snr_threshold and self.catalog_columns['snr']

        Returns:
        --------
        filtered_data : astropy.table.Table
            Filtered catalog data
        """
        if self.catalog is None:
            raise ValueError("No catalog loaded. Please set self.catalog")

        if self.catalog_columns is None or "snr" not in self.catalog_columns:
            raise ValueError(
                "catalog_columns must be set and contain 'snr' key"
            )

        snr_column = self.catalog_columns["snr"]
        print(f"\nFiltering galaxies with SNR > {self.snr_threshold}...")
        filtered = self.catalog.data[
            self.catalog.data[snr_column] > self.snr_threshold
        ]
        print(
            f"Found {len(filtered)} galaxies with SNR > {self.snr_threshold}"
        )

        if inplace:
            self.catalog.data = filtered
        return filtered

    def create_dataset(
        self,
        output_dir,
        filter_names=None,
        num_workers=1,
        filter_high_snr=True,
        max_galaxies=None,
    ):
        """
        Generate galaxy images and save as FITS.

        All galaxies are stored in output_dir/images/.
        Train/validation/test split is applied at runtime when loading
        the dataset (see cosmos_dataset.load_fits_dataset).

        When num_workers > 1, the catalog is divided into num_workers
        chunks that are processed in parallel via multiprocessing. Each
        worker creates its own GalaxySim with independent seed.

        Parameters
        ----------
        output_dir : str or Path
            Directory where to save the dataset
        filter_names : list of str, optional
            List of filter names to process. If None, uses all available
            filters
        num_workers : int, optional
            Number of parallel workers (default: 1)
        filter_high_snr : bool, optional
            If True, filter galaxies by SNR threshold.
            If False, use all catalog data (default: False)
        max_galaxies : int, optional
            Maximum number of galaxies to process.
            If None, process all (default: None)
        """
        if self.catalog is None:
            raise ValueError("No catalog loaded. Please set self.catalog")

        if filter_names is None:
            filter_names = self.survey.available_filters

        # Get catalog data - either filtered or all
        if filter_high_snr:
            self.filter_high_snr_galaxies()

        data = self.catalog.data

        # Optionally limit number of galaxies
        if max_galaxies is not None and max_galaxies < len(data):
            print(f"Limiting to {max_galaxies} galaxies...")
            rng = np.random.default_rng(self.random_seed)
            indices = rng.choice(len(data), max_galaxies, replace=False)
            data = data[indices]

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        images_path = output_dir / "images"
        images_path.mkdir(parents=True, exist_ok=True)

        sim_kwargs = {
            "survey_name": self.survey.name,
            "image_size": self.image_size,
            "max_fft_size": self.max_fft_size,
            "catalog_columns": self.catalog_columns,
            "galaxy_type": self.galaxy_type,
        }

        print(
            f"\nProcessing {len(data)} galaxies with "
            f"{num_workers} worker(s)..."
        )

        galaxy_rows = [
            dict(zip(row.colnames, row, strict=False)) for row in data
        ]
        if num_workers != 1:
            chunks = np.array_split(galaxy_rows, num_workers)
        else:
            chunks = [galaxy_rows]
        chunk_args = [
            (
                list(chunk),
                str(images_path),
                filter_names,
                sim_kwargs,
                self.random_seed + worker_idx
                if self.random_seed
                else worker_idx,
            )
            for worker_idx, chunk in enumerate(chunks)
            if len(chunk) > 0
        ]

        if num_workers > 1:
            with Pool(processes=num_workers) as pool:
                results = list(
                    tqdm(
                        pool.imap_unordered(_process_chunk_args, chunk_args),
                        total=len(chunk_args),
                        desc="Generating galaxies",
                    )
                )
        else:
            results = (
                [_process_chunk(*chunk_args[0], show_progress=True)]
                if chunk_args
                else []
            )

        all_metadata_rows = []
        total_failed = 0
        for metadata_rows, failed_count in results:
            all_metadata_rows.extend(metadata_rows)
            total_failed += failed_count

        all_metadata_rows.sort(key=lambda r: r["id"])

        csv_path = output_dir / "metadata.csv"
        with open(csv_path, "w", newline="") as csvfile:
            csv_writer = csv.DictWriter(
                csvfile, fieldnames=all_metadata_rows[0].keys()
            )
            csv_writer.writeheader()
            csv_writer.writerows(all_metadata_rows)

        if total_failed > 0:
            print(f"Warning: Failed to generate {total_failed} galaxies")
        print(f"{len(all_metadata_rows)} galaxies saved to {output_dir}")
