import tomllib
from argparse import ArgumentParser
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from datetime import datetime
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import numpy as np
from loguru import logger
from tqdm import tqdm
from xspec import Model, Xset

from src.config import EnergySettings, EnvironmentCfg, MultiprocessingCfg, SimputCfg
from src.tools.files import compress_targz, decompress_targz
from src.tools.run_utils import configure_logger, load_satellites
from src.xmm.tools import get_background_spectrum

logger.remove()


def _generate_img_simputs(
    tmp_dir: Path,
    executor: ProcessPoolExecutor,
    simput_cfg: SimputCfg,
    env_cfg: EnvironmentCfg,
    energies: EnergySettings,
) -> None:
    from src.simput.image import simput_image

    img_fs: dict[Future, Path] = {}

    img_path = simput_cfg.simput_dir / "img"
    img_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng()

    xspec_file = tmp_dir / "spectrum.xcm"
    Model("phabs*power", setPars={1: 0.04, 2: 2.0, 3: 0.01})
    Xset.save(f"{xspec_file}")

    if simput_cfg.fits_compressed.exists():
        logger.info(f"Found compressed FITS files in {simput_cfg.fits_compressed}")
        decompress_targz(
            in_file_path=simput_cfg.fits_compressed,
            out_file_dir=simput_cfg.fits_dir,
            tar_options="--strip-components=1",
        )

    amount_img = simput_cfg.img.n_gen
    fits_glob = simput_cfg.fits_dir.rglob("*.fits")
    in_files = fits_glob if amount_img == -1 else islice(fits_glob, amount_img)
    for in_file in in_files:
        tng_set, snapshot_num = in_file.parts[-3], in_file.parts[-2]
        # Check how many files have already been generated and how many are left to generate
        simput_glob = (img_path / tng_set / snapshot_num).glob(f"{in_file.stem}*")
        already_created = len(list(islice(simput_glob, simput_cfg.num_img_sample)))
        missing = simput_cfg.num_img_sample - already_created

        # None are left to be generated => Skip
        if missing < 1:
            logger.info(
                f"Won't generate any images for {in_file.name}. "
                "(Requested/Found: {simput_cfg.num_img_sample}/{already_created})"
            )
            if env_cfg.consume_data:
                in_file.unlink()
            continue

        logger.info(f"START\tGenerate {missing} images for {in_file.relative_to(simput_cfg.fits_dir)}")

        zoom = np.round(
            rng.uniform(
                simput_cfg.zoom_range[0],
                simput_cfg.zoom_range[1],
                missing,
            ),
            2,
        )
        sigma_b = np.round(
            rng.uniform(
                simput_cfg.sigma_b_range[0],
                simput_cfg.sigma_b_range[1],
                missing,
            ),
            2,
        )

        offset_x = np.round(
            rng.normal(
                -simput_cfg.offset_std,
                simput_cfg.offset_std,
                missing,
            ),
            2,
        )
        offset_y = np.round(
            rng.normal(
                -simput_cfg.offset_std,
                simput_cfg.offset_std,
                missing,
            ),
            2,
        )

        fs = executor.submit(
            simput_image,
            emin=energies.emin,
            emax=energies.emax,
            run_dir=Path(mkdtemp(dir=tmp_dir, prefix="img_")),
            xspec_file=xspec_file,
            consume_data=env_cfg.consume_data,
            img_path_in=in_file,
            zooms=zoom,
            sigmas_b=sigma_b,
            offsets_x=offset_x,
            offsets_y=offset_y,
            output_dir=img_path / tng_set / snapshot_num,
        )

        img_fs[fs] = in_file

    for future in tqdm(as_completed(img_fs), desc="Creating SIMPUTs for IMG", total=len(img_fs)):
        out_files = future.result()
        in_file = img_fs[future]
        logger.success(f"Created {len(out_files)} SIMPUTs for {in_file.relative_to(simput_cfg.fits_dir)}")
    logger.success("DONE\tGenerating SIMPUT for mode IMG.")


def _generate_bkg_simputs(
    tmp_dir: Path,
    executor: ProcessPoolExecutor,
    simput_cfg: SimputCfg,
    env_cfg: EnvironmentCfg,
    energies: EnergySettings,
    satellites: list,
) -> None:
    from src.simput.background import create_background

    bkg_fs: dict[Future, Path] = {}

    bkg_path = simput_cfg.simput_dir / "bkg"
    bkg_path.mkdir(parents=True, exist_ok=True)

    spectrum_fs = {}
    for sat in satellites:
        for name, instrument in sat:
            if not instrument.use:
                continue

            filter_abbrv = instrument.filter_abbrv
            fs = executor.submit(
                get_background_spectrum,
                instrument_name=name,
                spectrum_dir=env_cfg.working_dir / "spectrums",
                filter_abbr=filter_abbrv,
            )
            spectrum_fs[fs] = name

    logger.info("START\tGetting spectrum files.")
    for future in tqdm(as_completed(spectrum_fs), desc="Getting spectrum files", total=len(spectrum_fs)):
        spectrum_file = future.result()
        name = spectrum_fs[future]
        logger.success(f"Got spectrum file for {name} --> START Generation of BKG simputs.")
        fs = executor.submit(
            create_background,
            spectrum_file=spectrum_file,
            instrument_name=name,
            run_dir=Path(mkdtemp(dir=tmp_dir, prefix="bkg_")),
            output_dir=bkg_path,
            emin=energies.emin,
            emax=energies.emax,
        )
        bkg_fs[fs] = name
    logger.success("DONE\tGetting spectrum files.")

    for _ in tqdm(as_completed(bkg_fs), desc="Creating SIMPUTs for BKG", total=len(bkg_fs)):
        # Do nothing. Logging is already taken care of and there is nothing else to do.
        pass
    logger.success("DONE\tGenerating SIMPUT for mode BKG.")


def _generate_agn_simputs(
    tmp_dir: Path,
    executor: ProcessPoolExecutor,
    simput_cfg: SimputCfg,
    energies: EnergySettings,
    agn_counts_file: Path | None,
) -> None:
    skip_agn = agn_counts_file is None or (agn_counts_file.exists() and agn_counts_file.is_dir())
    if skip_agn:
        logger.warning(f"{agn_counts_file} does not exist! Won't create any AGN SIMPUTs.")
    else:
        from src.simput.agn import create_agn, get_s_n_from_file
        from src.xmm.tools import get_fov

        agn_fs = []

        logger.info("START\tGenerating SIMPUT for mode 'agn'...")
        agn_path = simput_cfg.simput_dir / "agn"
        agn_path.mkdir(parents=True, exist_ok=True)

        logger.info(f"Will generate {simput_cfg.agn.n_gen} AGNs")
        rng = np.random.default_rng()
        xspec_file = tmp_dir / "spectrum.xcm"
        Model("phabs*power", setPars={1: 0.04, 2: 2.0, 3: 0.001})
        Xset.save(f"{xspec_file}")
        s, n = get_s_n_from_file(agn_counts_file)
        n = n * np.pi * 0.25**2
        d = np.flip(np.ediff1d(np.flip(n)))
        d_sum = np.sum(d)
        p = d / d_sum

        star_counts = np.round(d_sum + rng.uniform(-1, 1, simput_cfg.agn.n_gen) * np.sqrt(d_sum)).astype(int)
        fov = get_fov("epn")

        for star_count in star_counts:
            counts = np.bincount(rng.choice(range(len(p)), size=star_count, p=p), minlength=len(p))
            indices = np.flatnonzero(counts)
            fluxes = [rng.uniform(low=s[i], high=s[i + 1], size=counts[i]) for i in indices]
            fluxes = np.concatenate(fluxes)
            offsets = rng.uniform(low=-fov / 2.0, high=fov / 2.0, size=(fluxes.shape[0], 2))
            fs = executor.submit(
                create_agn,
                fluxes=fluxes,
                offsets=offsets,
                emin=energies.emin,
                emax=energies.emax,
                run_dir=Path(mkdtemp(dir=tmp_dir, prefix="agn_")),
                output_dir=agn_path,
                xspec_file=xspec_file,
            )

            agn_fs.append(fs)

        for future in tqdm(as_completed(agn_fs), desc="Creating SIMPUTs for AGN", total=len(agn_fs)):
            out_files = future.result()
            logger.success(
                f"Created AGN SIMPUTs: [{', '.join([out_file.relative_to(agn_path) for out_file in out_files])}]"
            )
        logger.success("DONE\tGenerating SIMPUT for mode AGN.")


def generate_simputs(
    config_path: Path,
    agn_counts_file: Path | None,
) -> None:
    with open(config_path, "rb") as file:
        cfg: dict[str, dict] = tomllib.load(file)

    env_cfg = EnvironmentCfg(**cfg.pop("environment"))
    simput_cfg = SimputCfg(**cfg.pop("simput"), working_dir=env_cfg.working_dir, output_dir=env_cfg.output_dir)
    energies = EnergySettings(**cfg.pop("energy"))
    mp_cfg = MultiprocessingCfg(**cfg.pop("multiprocessing"))
    satellites = load_satellites(cfg.pop("instruments"))

    configure_logger(
        log_dir=env_cfg.log_dir,
        log_name="02_generate_simput_{time}.log",
        enqueue=True,
        debug=env_cfg.debug,
        verbose=env_cfg.verbose,
    )

    tar_and_compress = env_cfg.working_dir != env_cfg.output_dir

    logger.info(f"Found satellites with instruments: {satellites}")

    starttime = datetime.now()
    with TemporaryDirectory(prefix="simput_") as tmp_dir:
        tmp_dir = Path(tmp_dir)

        with ProcessPoolExecutor(max_workers=mp_cfg.num_cores) as executor:
            if simput_cfg.img.n_gen != 0:
                # TODO Check if files have been downloaded
                _generate_img_simputs(
                    tmp_dir=tmp_dir,
                    executor=executor,
                    simput_cfg=simput_cfg,
                    env_cfg=env_cfg,
                    energies=energies,
                )
                if tar_and_compress:
                    logger.info(
                        f"IMG SIMPUTs will be compressed and moved to {simput_cfg.img_compressed}. "
                        f"Existing file will be overwritten."
                    )
                    simput_cfg.img_compressed.unlink(missing_ok=True)
                    executor.submit(
                        compress_targz,
                        in_path=simput_cfg.simput_dir / "img",
                        out_file_path=simput_cfg.img_compressed,
                        remove_files=True,
                    )

            if simput_cfg.bkg.n_gen:
                _generate_bkg_simputs(
                    tmp_dir=tmp_dir,
                    executor=executor,
                    simput_cfg=simput_cfg,
                    env_cfg=env_cfg,
                    energies=energies,
                    satellites=satellites,
                )
                if tar_and_compress:
                    logger.info(
                        f"BKG SIMPUTs will be compressed and moved to {simput_cfg.bkg_compressed}. "
                        f"Existing file will be overwritten."
                    )
                    simput_cfg.bkg_compressed.unlink(missing_ok=True)
                    executor.submit(
                        compress_targz,
                        in_path=simput_cfg.simput_dir / "bkg",
                        out_file_path=simput_cfg.bkg_compressed,
                        remove_files=True,
                    )

            if simput_cfg.agn.n_gen > 0:
                _generate_agn_simputs(
                    tmp_dir=tmp_dir,
                    executor=executor,
                    simput_cfg=simput_cfg,
                    energies=energies,
                    agn_counts_file=agn_counts_file,
                )
                if tar_and_compress:
                    logger.info(
                        f"AGN SIMPUTs will be compressed and moved to {simput_cfg.agn_compressed} . "
                        f"Existing file will be overwritten."
                    )
                    simput_cfg.agn_compressed.unlink(missing_ok=True)
                    executor.submit(
                        compress_targz,
                        in_path=simput_cfg.simput_dir / "agn",
                        out_file_path=simput_cfg.agn_compressed,
                        remove_files=True,
                    )
    endtime = datetime.now()
    logger.info(f"Duration: {endtime - starttime}")


if __name__ == "__main__":
    parser = ArgumentParser(prog="", description="")
    parser.add_argument(
        "-a",
        "--agn_counts_file",
        default=Path(__file__).parent.resolve() / "res" / "agn_counts.cgi",
        type=Path,
        help="Path to agn_counts_cgi.",
    )
    parser.add_argument(
        "-p",
        "--config_path",
        type=Path,
        default=Path(__file__).parent.resolve() / "config.toml",
        help="Path to config file.",
    )

    args = parser.parse_args()

    generate_simputs(
        config_path=args.config_path,
        agn_counts_file=args.agn_counts_file,
    )
