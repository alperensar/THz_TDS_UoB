# tds_preprocessing_CLI.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Main CLI for processing time domain signal using TDS_preprocessing class via GUI
#
# author@: Alperen Sari - PhDing at the University of Birmingham
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# TO DO: independence from constant folder structures
# TO DO: FP term calc from file with epsilon value
from __future__ import annotations

import gc
import sys
import typer
import logging
from tqdm import tqdm
from pathlib import Path
from pydantic import ValidationError
from typing import Tuple, List, Optional

import pickle
import logging
import scipy.io
import numpy as np
import pandas as pd
import tkinter as tk
import matplotlib.pyplot as plt
from tkinter import filedialog

from tds_fft_processor import THzFFTprocessor 
from tds_processing_GUI import THzInteractivePlotter
from config import load_and_validate_config, AppConfig

EPS_VAL = 1e-20
app = typer.Typer()


def ensure_reference_file(pulse_dir: Path, cfg: AppConfig):
    """ 
    Determine (sample, reference) pairs inside a given pulse directory. 
    If there is seperate reference.txt or reference.csv use it otherwise reference embedded in sample folder. 
    """
    ref_name = cfg.paths.reference_filename
    allowed_exts = {f".{ext.lower()}" for ext in cfg.file_handling.extensions}

    data_files = [f for f in pulse_dir.iterdir() if f.is_file() and f.suffix.lower() in allowed_exts]

    if not data_files:
        raise RuntimeError(f"{pulse_dir}: no data files found with extensions {allowed_exts}.")

    pairs: List[Tuple[Path, Path]] = []
    ref_path = pulse_dir / ref_name if ref_name else None
    if ref_path and ref_path.exists():
        logging.info("📌 %s: reference file '%s' found. All files will be paired with it.", pulse_dir.name, ref_name,)
        sample_files = [f for f in data_files if f != ref_path]
        if not sample_files:
            raise RuntimeError(f"{pulse_dir}: only reference file found; no sample files.")
        for f in sample_files:
            pairs.append((f, ref_path))
    else:
        logging.info("📌 %s: no dedicated reference file found. Using self-reference mode: %s", pulse_dir.name, cfg.file_handling.allow_self_reference,)
        if not cfg.file_handling.allow_self_reference:
            raise RuntimeError(f"{pulse_dir}: no '{ref_name}' found and allow_self_reference=False. "  "Cannot determine reference for samples.")
        for f in data_files:
            pairs.append((f, f))

    return pairs  # [(sample_path, reference_path)]

def load_thickness_fp(file: Path, req_cols: AppConfig):
    """
    Loads the FP/Thickness file.
    - Creates template if missing.
    - Validates columns based on config.
    - Removes duplicate materials (keeps the LAST entry).
    """
    try:
        col_name = req_cols[0]
    except (AttributeError, IndexError):
        logging.error("Config 'required_cols_list' is invalid.")
        return pd.DataFrame()

    if not file.exists():
        logging.warning(f"⚠️ FP/Thickness file not found. Creating a template at: {file}")
        df_empty = pd.DataFrame(columns=req_cols)
        df_empty.to_csv(file, index=False, sep="\t") 
        logging.warning("Template file created. Continuing with empty database.")
        return df_empty

    try:
        df = pd.read_csv(file, sep=None, engine="python")
    except Exception as e:
        logging.error(f"Failed to read FP file: {e}")
        return pd.DataFrame()

    if not set(req_cols).issubset(df.columns):
        missing = set(req_cols) - set(df.columns)
        raise ValueError(f"File {file} is missing required columns: {missing}")

    if not set(req_cols).issubset(df.columns):
        missing = set(req_cols) - set(df.columns)
        raise ValueError(f"File {file} is missing required columns: {missing}")
    
    if not df.empty:
        df = df[~df[col_name].astype(str).str.strip().str.startswith("#")]

    # normalize columns and duplicate cleaning by keeping last material_name
    df["_norm_name"] = df[col_name].astype(str).str.strip().str.lower()
    df = df.drop_duplicates(subset=["_norm_name"], keep="last")

    numeric_cols = req_cols[1:] 
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df

def run_processing_pipeline(cfg: AppConfig):
    """Main THz-TDS FFT processing pipeline driven by YAML configuration."""
    
    root_dir = Path(cfg.paths.root_dir).expanduser() if cfg.paths.root_dir else None
    if root_dir is None or not root_dir.exists():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        try:
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askdirectory(title="Select the main (root) directory (pulse folders will be searched)")
        finally:
            try:
                root.destroy()
            except:
                pass
            gc.collect()

        if not selected:
            raise RuntimeError("🚫 No folder selected — aborting pipeline.")
        root_dir = Path(selected)

    # 1) Load thickness map and FP pulse if exists otherwise create it
    req_cols = cfg.file_handling.required_cols
    if len(req_cols) < 2:
        raise ValueError("Config 'required_cols_list' must have at least [Material, Thickness].")

    # col_name = req_cols[0]
    col_thick = req_cols[1]
    col_eps = req_cols[3] if len(req_cols) > 3 else None
    col_sigma = req_cols[2] if len(req_cols) > 2 else None

    # 1) Load thickness map and FP pulse if exists otherwise create it
    database_file = root_dir / cfg.paths.thickness_file
    df_fp = load_thickness_fp(database_file, req_cols)


    pulse_dirs = [d for d in root_dir.rglob("*") if d.is_dir() and d.name.lower() in {name.lower() for name in cfg.paths.pulse_dir_names}]
    if not pulse_dirs:
        logging.warning("No pulse directories found under %s", root_dir)
        return
    
    pulse_dirs = sorted(pulse_dirs)
    logging.info(f"📁 Found {len(pulse_dirs)} pulse folders.")

    # Iterate over pulse folders
    for pulse_idx, p_dir in enumerate(pulse_dirs): # [23:]
        
        print(f"[{pulse_idx}] Processing")

        # Index skipping options
        if cfg.run.process_only_index is not None and pulse_idx != cfg.run.process_only_index:
            continue
        if cfg.run.start_from_index is not None and pulse_idx < cfg.run.start_from_index:
            continue

        # Identify Material & Instrument
        try:
            # relative path parts: 0 -> material, 1 -> instrument
            parts = p_dir.relative_to(root_dir).parts
            material_name = parts[0]
            instrument_name = parts[1] if len(parts) > 1 else "Unknown"
        except Exception:
            logging.error("Invalid folder structure in %s", p_dir)
            continue
        
        logging.info(f"--- Processing: {material_name} ({instrument_name}) ---")

        sigma_um = 0.0
        eps_guess = 1.0
        base_thickness_mm = None
        override_fp_terms = None 

        search_key = material_name.lower()

        if not df_fp.empty and "_norm_name" in df_fp.columns:
            match_row = df_fp[df_fp["_norm_name"] == search_key]

        if not match_row.empty:
            row_data = match_row.iloc[0]
            val_t = row_data[col_thick]

            if pd.notna(val_t):
                base_thickness_mm = float(val_t)
                
            if col_eps and pd.notna(row_data[col_eps]):
                eps_guess = float(row_data[col_eps])
                
            if col_sigma and pd.notna(row_data[col_sigma]):
                sigma_um = float(row_data[col_sigma])
        else:
            logging.warning(f"Material '{material_name}' not found in database file.")

        try:
            overrides_dict = cfg.material_overrides.model_dump()
        except AttributeError:
            overrides_dict = cfg.material_overrides.dict()

        ov_match = None
        for key, val in overrides_dict.items():
            if key.lower() == search_key:
                ov_match = val
                break

        if ov_match:
            logging.info(f"Applying YAML overrides for '{material_name}': {ov_match}")
            
            if "measured_thickness_mm" in ov_match:
                base_thickness_mm = float(ov_match["measured_thickness_mm"])
            if "eps" in ov_match:
                eps_guess = float(ov_match["eps"])
            if "sigma_um" in ov_match:
                sigma_um = float(ov_match["sigma_um"])
            if "fp_terms" in ov_match:
                override_fp_terms = int(ov_match["fp_terms"])

        if base_thickness_mm is None:
            err_msg = (f"CRITICAL: Thickness for '{material_name}' is UNKNOWN. "
                    f"Please add it to '{database_file.name}' or YAML config.")
            logging.error(err_msg)
            continue
            # raise ValueError(err_msg)

        logging.info(f"FINAL PARAMS -> Thickness: {base_thickness_mm} mm, \u03B5: {eps_guess}, \u03C3: {sigma_um}")

        try:
            profiles_dict = cfg.instrument_profiles.model_dump()
        except AttributeError:
            profiles_dict = cfg.instrument_profiles.dict()

        if instrument_name not in profiles_dict:
            raise ValueError(f"Instrument '{instrument_name}' not defined in config.instrument_profiles. Available: {list(profiles_dict.keys())}")

        profile = profiles_dict[instrument_name]
        unwrap_min, unwrap_max = profile.get('unwrap_range')
        extraction_min, extraction_max = profile.get('extraction_range')

        # Apply optional material-specific overrides inside profile
        if "custom_material_limits" in profile and profile["custom_material_limits"]:
             if material_name in profile["custom_material_limits"]:
                mlim = profile["custom_material_limits"][material_name]
                # Sözlükten değerleri güvenli çekme
                unwrap_min = float(mlim.get("unwrap_min", unwrap_min))
                unwrap_max = float(mlim.get("unwrap_max", unwrap_max))
                extraction_min = float(mlim.get("extraction_min", extraction_min))
                extraction_max = float(mlim.get("extraction_max", extraction_max))
                
        # 6) Find sample/reference file pairs inside p_dir
        subdirs = [d for d in p_dir.iterdir() if d.is_dir() and d.name.lower() in {s.lower() for s in cfg.paths.subdir_types}]
        target_dirs = subdirs if subdirs else [p_dir]
        
        # Unwrap data
        unwrap_log_path = (root_dir / cfg.paths.unwrap_log) if cfg.run.save_unwrap_log else None

        for subdir in target_dirs:
            try:
                file_pairs = ensure_reference_file(subdir, cfg)
            except RuntimeError as e:
                logging.error("Skipping %s: %s", subdir, e)
                continue
        
            logging.info("[%s] Files to process: %d", subdir.name, len(file_pairs))

            # 7) Process each (sample, reference) pair
            for pair_idx, (sample_path, reference_path) in enumerate(tqdm(file_pairs, desc=f"[{subdir.name}]", ncols=cfg.logging.tqdm_width)):
                try:
                    # FP term count: override from YAML or let TDS_preprocessing handle it internally
                    fp_count = override_fp_terms if override_fp_terms is not None else 0

                    # Sample/Ref column logic
                    sample_column = 1
                    reference_column = 2 if sample_path == reference_path else 1
                        
                    # Build TDS_preprocessing instance (unified time → FFT → H(f) pipeline)
                    freq_processed = THzFFTprocessor(
                        sample_file=        Path(sample_path),
                        reference_file=     Path(reference_path),
                        sample_name=        material_name,
                        instrument_name=    instrument_name,
                        sample_column=      sample_column,
                        reference_column=   reference_column,
                        sigma=              sigma_um,
                        measured_thickness= base_thickness_mm,
                        time_unit=          cfg.file_handling.time_unit,
                        unwrap_range_selec= cfg.unwrap.use_gui,
                        preset=             [extraction_min, extraction_max, fp_count, unwrap_min, unwrap_max],
                        unwrap_file_root=   unwrap_log_path,
                        eps_guess=          eps_guess,
                        cfg=                cfg)

                    # GUI based updated freq_processed
                    THzInteractivePlotter(freq_processed) 
                    plt.show(block=True)

                    if freq_processed.use_backup_unwrap:
                        freq_processed.unwrapped_phase = freq_processed.zero_unwrapped_phase_backup
                    else:
                        freq_processed.unwrapped_phase = freq_processed.zero_unwrapped_phase

                    no_save_mode =cfg.saving.no_save_mode # Saving Logic (Legacy / Plotting)
                    
                    # Prepare frequency-domain data dictionary (legacy content)
                    if cfg.run.save_outputs and not no_save_mode and (cfg.saving.save_mat or cfg.saving.save_txt):
                        mag_db = 20 * np.log10(np.abs(freq_processed.measured_H) + EPS_VAL)

                        fmin = float(np.min(freq_processed.freqs_interest))
                        fmax = float(np.max(freq_processed.freqs_interest))
                        mask = (freq_processed.fft_freqs >= fmin) & (freq_processed.fft_freqs <= fmax)

                        # data for matlab
                        mat_data = {
                            "sample_signal_time":       np.asarray(freq_processed.sample.time_data),
                            "sample_signal_y":          np.asarray(freq_processed.sample.y_data),
                            "reference_signal_time":    np.asarray(freq_processed.reference.time_data),
                            "reference_signal_y":       np.asarray(freq_processed.reference.y_data),
                            "fft_sample":               np.asarray(freq_processed.fft_sample),
                            "fft_reference":            np.asarray(freq_processed.fft_reference),
                            "fft_freq":                 np.asarray(freq_processed.fft_freqs),
                            "measured_H":               np.asarray(freq_processed.measured_H),
                            "fft_freq_masked":          np.asarray(freq_processed.fft_freqs[freq_processed.x_min_index:freq_processed.x_max_index]),
                            "measured_H_masked":        np.asarray(freq_processed.measured_H[freq_processed.x_min_index:freq_processed.x_max_index]),
                            "magH":                     mag_db,
                            "phaseH":                   np.asarray(freq_processed.zero_unwrapped_phase),
                            "magH_masked":              mag_db[freq_processed.x_min_index:freq_processed.x_max_index],
                            "phaseH_masked":            np.asarray(freq_processed.zero_unwrapped_phase[freq_processed.x_min_index:freq_processed.x_max_index]),
                            "freq_mask_boolean":        mask.astype(int),
                            "dynamic_range_db":         np.asarray(freq_processed.DR_dB_spectrum),
                            "alpha_max":                np.asarray(freq_processed.alpha_max),}
                                                 
                        base_dir = p_dir.parent / cfg.paths.base_FFT_folder
                        base_dir.mkdir(parents=True, exist_ok=True)

                        base_name = sample_path.stem

                        if cfg.saving.save_mat:
                            mat_path = base_dir / f"{base_name}.mat"
                            scipy.io.savemat(str(mat_path), mat_data)
                            logging.info("Saved MAT: %s", mat_path)

                        if cfg.saving.save_txt:
                            txt_path = base_dir / f"{base_name}.txt"

                            # Flatten all arrays to same length table with NaN padding
                            headers = list(mat_data.keys())
                            cols = [np.asarray(v).reshape(-1) for v in mat_data.values()]
                            max_len = max(len(c) for c in cols)
                            table = np.full((max_len, len(cols)), np.nan + 0j, dtype=complex)
                            for i, c in enumerate(cols):
                                table[:len(c), i] = c

                            header_line = "\t".join(headers)
                            np.savetxt(txt_path, table, delimiter="\t", header=header_line, comments="", fmt="%.18e",)
                            logging.info("Saved TXT: %s", txt_path)

                    # Pickle saving (processor object)
                    if cfg.saving.save_pickle and not no_save_mode and cfg.saving.save_pickle:
                        out_dir = subdir.parent / cfg.paths.pickle_folder
                        out_dir.mkdir(parents=True, exist_ok=True)

                        pkl_name = f"{sample_path.stem}.pkl"
                        if override_fp_terms:
                            pkl_name = f"fp{override_fp_terms}_" + pkl_name

                        with open(out_dir / pkl_name, "wb") as pf:
                            pickle.dump(freq_processed, pf)
                        logging.info("✔️ Pickled processor → %s", out_dir / pkl_name)

                except Exception as e:
                    logging.error("❌ Error while processing %s: %s", sample_path.name, e)

@app.command()
def main(
    config:         Path = typer.Option("config.yaml",         "--config",         help="Path to the main YAML configuration file.", exists=True, readable=True),
    root_dir:       Optional[Path] = typer.Option(None,        "--root-dir",       help="Override paths.root_dir (main dataset directory)"),
    reference_name: Optional[str] = typer.Option(None,         "--reference-name", help="Override paths.reference_filename."),
    no_save:        bool = typer.Option(False,                 "--no-save",        help="Disable all saving (MAT/TXT/PKL)."),
    start_from:     Optional[int] = typer.Option(None,         "--start",          help="Start processing from this index, overrides run.start_from_index"),
    overrides:      Optional[List[str]] = typer.Argument(None,                     help="CLI overrides in 'key.subkey=value' format (e.g., 'run.interactive=True')."),):
    """
    THz-TDS FFT Processing Pipeline:
     - Find all pulse folders and then process files inside those folders if they are .txt or .csv
     - Loads YAML configuration, applies CLI overrides, and runs the signal → FFT → H(ω) → unwrap → save pipeline.
    """
    all_overrides = list(overrides) if overrides else []

    if root_dir:                all_overrides.append(f"paths.root_dir={root_dir}")
    if reference_name:          all_overrides.append(f"paths.reference_filename={reference_name}")
    if start_from is not None:  all_overrides.append(f"run.start_from_index={start_from}")
    if no_save:
        all_overrides += [
            "saving.save_mat    =False",
            "saving.save_txt    =False",
            "saving.save_pickle =False",
            "saving.no_save_mode=True",]
    
    if all_overrides:
        typer.echo(f"Applying overrides: {all_overrides}")

    try:
        # Load and validate the configuration
        cfg: AppConfig = load_and_validate_config(config, all_overrides)
        typer.echo(typer.style("Configuration loaded and validated successfully.", fg=typer.colors.GREEN))
   
    except (FileNotFoundError, ValidationError) as e:
         # Pydantic validation error
         typer.echo(typer.style(f"CONFIGURATION ERROR:\n{e}", fg=typer.colors.RED), err=True)
         sys.exit(1)
    
    except Exception as e:
        typer.echo(typer.style(f"An unexpected error occurred: {e}", fg=typer.colors.RED), err=True)
        sys.exit(1)

    typer.echo("📌 Configuration loaded successfully.")
    typer.echo(f"   → root_dir: {cfg.paths.root_dir}")
    typer.echo(f"   → reference: {cfg.paths.reference_filename}")
    typer.echo(f"   → start_from_index: {cfg.run.start_from_index}")
    typer.echo(f"   → no_save_mode: {getattr(cfg.saving, 'no_save_mode', False)}")
    typer.echo("--------------------------------------------------------")

    run_processing_pipeline(cfg)

    typer.echo("✔️ Processing completed.")

if __name__ == "__main__":
    app()