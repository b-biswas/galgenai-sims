"""
Generate Galaxy Images Dataset as FITS files
"""

import yaml
from pathlib import Path

from galgenai_sims.config import load_sim_config
from galgenai_sims import COSMOSWebCatalog, GalaxySim


def main():
    config_dict = load_sim_config()

    # Load all required config values
    try:
        cosmos_config = config_dict["cosmos"]
        catalog_path = cosmos_config["catalog_path"]
        output_dir = cosmos_config["hf_dataset_path"]
        snr_threshold = cosmos_config["HST_snr_threshold"]
        image_size = cosmos_config["image_size"]
        max_galaxies = cosmos_config["max_galaxies"]
        num_workers = cosmos_config["num_workers"]
        random_seed = cosmos_config["seed"]  # Fixed seed for reproducibility
        max_fft_size = cosmos_config["max_fft_size"]
        catalog_columns = cosmos_config["catalog_columns"]
        galaxy_type = cosmos_config.get("galaxy_type", "sersic")  # Default to sersic
        disk_mag_threshold = cosmos_config.get("disk_mag_threshold", 50.0)  # Default to 50.0
    except KeyError as e:
        raise ValueError(
            f"Missing required config value: {e}. "
            "Please ensure all required values are present in the"
            " config file under 'cosmos' section."
        ) from e

    # Create output directory and save config
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    config_to_save = {
        "dataset_generation": {
            "catalog_path": catalog_path,
            "hf_dataset_path": output_dir,
            "snr_threshold": snr_threshold,
            "image_size": image_size,
            "max_galaxies": max_galaxies,
            "num_workers": num_workers,
            "seed": random_seed,
            "max_fft_size": max_fft_size,
            "survey_name": "HSC",
            "catalog_columns": catalog_columns,
            "galaxy_type": galaxy_type,
            "disk_mag_threshold": disk_mag_threshold,
        }
    }

    config_save_path = output_path / "generation_config.yaml"
    print(f" Saving generation config to :     {config_save_path}")

    with open(config_save_path, "w") as f:
        yaml.dump(config_to_save, f, default_flow_style=False, sort_keys=False)
    print(f"\nSaved generation config to: {config_save_path}")

    # Load from B+D HDU if using bulge+disk model
    use_bulge_disk = (galaxy_type == "bulge+disk")

    catalog = COSMOSWebCatalog(
        catalog_path=catalog_path,
        required_columns_only=True,
        use_bulge_disk=use_bulge_disk
    )

    print(
        f"\nInitializing GalaxySim (image_size={image_size},"
        f" max_fft_size={max_fft_size}, galaxy_type={galaxy_type})..."
    )
    sim = GalaxySim(
        catalog=catalog,
        survey_name="HSC",
        image_size=image_size,
        random_seed=random_seed,
        max_fft_size=max_fft_size,
        catalog_columns=catalog_columns,
        snr_threshold=snr_threshold,
        galaxy_type=galaxy_type,
        disk_mag_threshold=disk_mag_threshold,
    )

    filter_names = sim.survey.available_filters
    print(f"Simulating in bands: {filter_names}")

    sim.create_dataset(
        output_dir=output_dir,
        filter_names=filter_names,
        num_workers=num_workers,
        filter_high_snr=True,
        max_galaxies=max_galaxies,
    )


if __name__ == "__main__":
    main()
