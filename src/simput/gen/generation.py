from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal
from uuid import uuid4

from loguru import logger

from src.simput.gen.background import background
from src.simput.gen.image import simput_image
from src.simput.agn import get_fluxes
from src.simput.gen.pointsource import simput_ps
from src.simput.utils import merge_simputs
from src.xmm_utils.file_utils import compress_gzip


def create_background(
    instrument_name: Literal["epn", "emos1", "emos2"],
    emin: float,
    emax: float,
    run_dir: Path,
    spectrum_file: Path,
) -> list[Path]:
    output_files = [
        background(
            run_dir=run_dir,
            spectrum_file=spectrum_file,
            instrument_name=instrument_name,
            emin=emin,
            emax=emax,
        )
    ]

    return output_files


def create_agn_sources(
    emin: float,
    emax: float,
    run_dir: Path,
    img_settings: dict,
    xspec_file: Path,
):
    output_files = []

    for _ in range(img_settings["n_gen"]):
        # Use the current time as id, such that clashes don't happen
        unique_id = uuid4().int
        output_file_path = run_dir / f"agn_{unique_id}_p0_{emin}ev_p1_{emax}ev.simput"
        simput_files: list[Path] = []

        # Get the fluxes from the agn distribution
        fluxes = get_fluxes(img_settings["agn_counts_file"])

        # TODO: make an option to make agns that are close together
        if img_settings["deblending_n_gen"] > 0:
            # TODO:
            # img_settings["deblending_min_sep"]
            # img_settings["deblending_max_sep"]
            # img_settings["deblending_max_flux_delta"]
            pass

        for i, flux in enumerate(fluxes):
            logger.info(f"Creating AGN with flux={flux}")
            output_file = run_dir / f"ps_{unique_id}_{i}.simput"
            output_file = simput_ps(
                emin=emin,
                emax=emax,
                output_file=output_file,
                src_flux=flux,
                xspec_file=xspec_file,
                offset="random",
            )
            simput_files.append(output_file)
        output_file = merge_simputs(simput_files=simput_files, output_file=output_file_path)
        output_files.append(output_file)

        for file in simput_files:
            file.unlink(missing_ok=True)

    return output_files


def simput_generate(
    emin: float,
    emax: float,
    mode: str,
    img_settings: dict,
    tmp_dir: Path,
    output_dir: Path,
    spectrum_file: Path,
) -> None:
    with TemporaryDirectory(dir=tmp_dir) as temp:
        run_dir = Path(temp)

        file_names = []

        if mode == "agn":
            file_names = create_agn_sources(
                emin=emin,
                emax=emax,
                run_dir=run_dir,
                img_settings=img_settings,
                xspec_file=spectrum_file,
            )

        if mode == "img":
            file_names = simput_image(
                emin=emin,
                emax=emax,
                run_dir=run_dir,
                img_settings=img_settings,
                xspec_file=spectrum_file,
            )
            img_path_in = img_settings["img_path"]
            tng_name = img_path_in.parts[-3]
            snapshot_num = img_path_in.parts[-2]
            output_dir = output_dir / tng_name / snapshot_num
            output_dir.mkdir(parents=True, exist_ok=True)

        if mode == "bkg":
            file_names = create_background(
                instrument_name=img_settings["instrument_name"],
                emin=emin,
                emax=emax,
                run_dir=run_dir,
                spectrum_file=spectrum_file,
            )

        for file_name in file_names:
            # Compress the simput file and move it to the correct output dir
            compressed_file = output_dir / f"{file_name.name}.gz"
            if compressed_file.exists():
                logger.warning(f"SIMPUT file {compressed_file.resolve()} already exists, skipping.")
            else:
                compress_gzip(in_file_path=file_name, out_file_path=compressed_file)
            file_name.unlink(missing_ok=True)