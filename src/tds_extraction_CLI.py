# tds_extraction_CLI.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Main CLI for Parameter Extraction (n, k) using TDS_Extraction logic
# Handles file discovery, configuration loading, and parallel execution.
#
# author@: Alperen Sari - PhDing at the University of Birmingham
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
from __future__ import annotations

import gc
import sys
import copy
import typer
import logging
from tqdm import tqdm
from pathlib import Path
from tkinter import filedialog
from typing import Optional, List
from pydantic import ValidationError

import pickle
import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from multiprocessing import Process, Queue, cpu_count
from matplotlib.widgets import Button, TextBox, CheckButtons

import tds_extraction_core as core
from config import load_and_validate_config, AppConfig

# try:
#     from mpi4py import MPI
#     MPI_AVAILABLE = True
# except ImportError:
#     MPI_AVAILABLE = False

app = typer.Typer(help="THz-TDS Parameter Extraction CLI")
logger = logging.getLogger("TDS_CLI")
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

THZ_TO_FREQ = 1E12

def L_key_um(L_m: float) -> int:
    return int(round(float(L_m) * 1e6))   # m -> µm (int)

def um_to_m(key_um: int) -> float:
    return float(key_um) * 1e-6           # µm -> m

def um_to_mm(key_um: int) -> float:
    return float(key_um) * 1e-3           # µm -> mm

class ThicknessSelector:
    def __init__(self, tv_scores: dict, nominal_L_m: float, best_L_m: float, tag: str):
        self.apply_to_all   = False
        self.tv_scores      = tv_scores
        self.nominal_L_mm   = nominal_L_m * 1000.0
        self.best_L_mm      = best_L_m * 1000.0
        self.selected_L_mm  = self.best_L_mm
        self.tag            = tag

        sorted_keys_um = sorted(self.tv_scores.keys())          # key = int(µm)
        self.l_vals_mm = [um_to_mm(k) for k in sorted_keys_um]  # µm -> mm
        self.tv_vals   = [self.tv_scores[k] for k in sorted_keys_um]

        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        plt.subplots_adjust(bottom=0.25)
        self.plot_graph()
        
        # 1. Text Box
        axbox = plt.axes([0.15, 0.15, 0.3, 0.05])
        self.text_box = TextBox(axbox, 'Thickness (mm): ', initial=f"{self.selected_L_mm:.4f}")
        self.text_box.on_submit(self.submit_text)

        # 2. Confirm Button
        axbtn = plt.axes([0.55, 0.15, 0.2, 0.05])
        self.btn = Button(axbtn, 'Confirm & Analyze', color='lightgreen', hovercolor='0.95')
        self.btn.on_clicked(self.confirm)

        # 3. APPLY TO ALL CHECKBOX
        axcheck = plt.axes([0.15, 0.05, 0.3, 0.05], frameon=False)
        self.check = CheckButtons(axcheck, ['Apply offset to all remaining'], [False])
        self.check.on_clicked(self.toggle_apply)

        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        plt.show(block=True)

    def toggle_apply(self, label):
        self.apply_to_all = not self.apply_to_all

    def plot_graph(self):
        self.ax.clear()
        self.ax.set_title(f"Thickness Selection: {self.tag}")
        self.ax.set_xlabel("Thickness (mm)")
        self.ax.set_ylabel("Total Variation (Score)")
        self.ax.grid(True, linestyle='--', alpha=0.5)

        # 1. Sweep Blue Line
        self.ax.plot(self.l_vals_mm, self.tv_vals, 'o-', color='navy', alpha=0.6, label='Sweep Data', picker=5)

        # 2. Nominal Value Red Line
        self.ax.axvline(self.nominal_L_mm, color='red', linestyle='--', linewidth=2, label=f'Nominal ({self.nominal_L_mm:.3f} mm)')

        # 3. Best Green Star
        best_key = L_key_um(self.best_L_mm / 1000.0)   # mm -> m -> key_um
        best_tv  = self.tv_scores[best_key]
        self.ax.plot(self.best_L_mm, best_tv, 'g*', markersize=15, label=f'Auto-Best ({self.best_L_mm:.4f} mm)')

        # 4. Selected
        if self.selected_L_mm in self.l_vals_mm:
            sel_key = L_key_um(self.selected_L_mm / 1000.0)
            sel_tv  = self.tv_scores.get(sel_key, np.nan)

        if self.selected_L_mm in self.l_vals_mm and np.isfinite(sel_tv):
            self.ax.plot(self.selected_L_mm, sel_tv, 's', color='orange', markersize=12, label='Selected')
        else:
            self.ax.axvline(self.selected_L_mm, color='orange', linestyle='-', linewidth=2, label='User Selected')

        #     self.ax.plot(self.selected_L_mm, sel_tv, 's', color='orange', markersize=12, label='Selected')
        # else:
        #     self.ax.axvline(self.selected_L_mm, color='orange', linestyle='-', linewidth=2, label='User Selected')

        self.ax.legend()
        self.fig.canvas.draw()

    def on_click(self, event):
        if event.inaxes != self.ax: return

        clicked_x = event.xdata
        if clicked_x:
            nearest = min(self.l_vals_mm, key=lambda x: abs(x - clicked_x))
            self.update_selection(nearest)

    def submit_text(self, text):
        try:
            val = float(text)
            self.update_selection(val)
        except ValueError:
            pass

    def update_selection(self, val_mm):
        self.selected_L_mm = val_mm
        self.text_box.set_val(f"{self.selected_L_mm:.4f}")
        self.plot_graph()

    def confirm(self, event):
        try:
            self.fig.canvas.stop_event_loop()
            self.fig.set_visible(False) 
            plt.close(self.fig)
            
        except Exception as e:
            print(f"Error closing figure: {e}")
            plt.close('all')

def run_extraction_pipeline(pkl_path: Path, output_root: Path, backend: str, cfg: AppConfig, global_state: dict):
    """
    Loads a pickle. 
    IF 'Apply to All' is active -> Skips sweep, calculates thickness from offset.
    ELSE -> Runs sweep (Step 1), opens GUI, gets user selection.
    FINALLY -> Runs parameter extraction (Step 2) and saves results.
    """
    try:
        with open(pkl_path, "rb") as f:
            pair_org = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load {pkl_path}: {e}")
        return

    # Önceki pencereleri temizle (GUI donmasını önler)
    plt.close('all')

    pe_cfg = cfg.param_estimation
    algo_step1 = pe_cfg.step1_algorithm  
    algo_step2 = pe_cfg.step2_algorithm  
    bounds_info = getattr(pe_cfg, 'optimization_bounds', {}) 
    base_out = Path(pkl_path).parent.parent / cfg.paths.processed_dir_names
    current_file_stem = pkl_path.stem

    # Processing Modes Loop (Roughness / FP)
    for rough, fp in pe_cfg.processing_modes:
        pair = copy.deepcopy(pair_org)
        
        tag = f"{'R' if rough else 'NR'}_{'FP' if fp else 'NFP'}"
        out_dir = base_out / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        measurement_name = f"{pkl_path.stem}_{tag}"
        
        logger.info(f"Processing {measurement_name} | Roughness: {rough} | FP: {fp}")

        # Değişkenleri başlat
        final_L_m = None
        tv_scores = {}
        thickness_sweep = None

        # ---------------------------------------------------------------------------
        # MANTIK AYRIMI: "APPLY TO ALL" VAR MI?
        # ---------------------------------------------------------------------------
        
        # DURUM 1: "Hepsine Uygula" AÇIK -> HIZLI YOL (Sweep ve GUI Yok)
        if global_state.get('apply_all', False) and global_state.get('offset_mm') is not None:
            nominal_L       = pair.measured_thickness_m
            offset_val      = global_state['offset_mm']
            
            # Direkt hesapla
            calculated_L_mm = (nominal_L * 1000.0) + offset_val
            final_L_m       = calculated_L_mm / 1000.0
            
            logger.info(f"⏭️ Skipping Step 1. Auto-applying offset ({offset_val:.4f} mm). Target L: {calculated_L_mm:.4f} mm")

        # DURUM 2: "Hepsine Uygula" KAPALI -> YAVAŞ YOL (Sweep Yap + GUI Sor)
        else:
            # --- STEP 1: SWEEP ---
            center_L= pair.measured_thickness_m
            res     = pe_cfg.scanning_resolution
            range_m = pe_cfg.scanning_range_mm * 1e-3
            start_L = center_L - range_m
            stop_L  = center_L + range_m

            if res % 2 == 0: res += 1 
            thickness_sweep = np.linspace(start_L, stop_L, res)
            thickness_sweep[res // 2] = center_L 
            
            logger.info(f"Sweeping {len(thickness_sweep)} points via {algo_step1}. Range: {start_L*1e3:.4f} - {stop_L*1e3:.4f} mm")

            results_data = {}   
            worker_cfg_step1 = {
                'algo':         algo_step1,         
                'whole_freq':   True,         
                'roughness':    rough, 
                'fp':           fp,
                'bounds':       bounds_info}
        
            # --- BACKEND EXECUTION ---
            if backend == "serial":
                for L in tqdm(thickness_sweep, desc="Serial Sweep", leave=False):
                    _, data = core.run_param_extraction((pair, L, worker_cfg_step1))
                    if data:
                        results_data[L] = data
                        
            elif backend == "multiprocessing":
                q_task = Queue()
                q_result = Queue()
                
                for L in thickness_sweep:
                    q_task.put((pair, L, worker_cfg_step1, 0))
                
                try:
                    num_cores = int(pe_cfg.calculation_cores) if isinstance(pe_cfg.calculation_cores, int) else cpu_count()-1
                except:
                    num_cores = 2

                workers = []
                for _ in range(num_cores):
                    p = Process(target=core.worker_process, args=(q_task, q_result))
                    p.start()
                    workers.append(p)
                    q_task.put(None)

                for _ in tqdm(range(len(thickness_sweep)), desc=f"Multi {tag}", leave=False):
                    L_out, data = q_result.get()
                    if data:
                        results_data[L_out] = data
                
                for p in workers:
                    if p.is_alive(): p.terminate()
                    p.join()

            # --- GUI SELECTION ---
            if not results_data:
                logger.warning("No valid results generated from Sweep. Skipping selection.")
                continue # Diğer moda geç
            else:
                tv_scores_raw = core.calculate_total_variation(results_data)

                if len(tv_scores_raw) > 0 and isinstance(next(iter(tv_scores_raw.keys())), float):
                    tv_scores = {L_key_um(L_m): score for L_m, score in tv_scores_raw.items()}
                else:
                    tv_scores = tv_scores_raw

                auto_best_key = min(tv_scores, key=tv_scores.get)   # int (µm)
                auto_best_L_m = um_to_m(auto_best_key)              # m
                nominal_L     = pair.measured_thickness_m

                print(f"Nominal: {nominal_L*1000:.6f} mm | Auto-Best: {auto_best_L_m*1000:.6f} mm")

                # tv_scores   = core.calculate_total_variation(results_data)
                # auto_best_L = min(tv_scores, key=tv_scores.get) 
                # nominal_L   = pair.measured_thickness_m

                # print(f"\n--- Selection for {measurement_name} ---")
                # print(f"Nominal: {nominal_L*1000:.6f} mm | Auto-Best: {auto_best_L*1000:.6f} mm")

                try:
                    # GUI'yi aç
                    selector = ThicknessSelector(tv_scores, nominal_L, auto_best_L_m, tag)
                    final_L_mm = selector.selected_L_mm
                    final_L_m  = final_L_mm / 1000.0
                    
                    logger.info(f"👉 USER SELECTED: {final_L_mm:.6f} mm")
                    
                    # Kullanıcı "Apply to All" dediyse STATE güncelle
                    if selector.apply_to_all:
                        offset = final_L_mm - (nominal_L * 1000.0)
                        global_state['apply_all'] = True
                        global_state['offset_mm'] = offset
                        logger.info(f"🔒 'Apply to All' ACTIVATED. Offset {offset:.4f} mm saved.")
                
                except Exception as e:
                    logger.error(f"GUI Error: {e}. Defaulting to auto-best.")
                    final_L_m = auto_best_L_m
                
                # GUI kapandıktan sonra temizlik
                plt.close('all')

        # ---------------------------------------------------------------------------
        # STEP 3: FINAL OPTIMIZATION (ORTAK ALAN)
        # ---------------------------------------------------------------------------
        if final_L_m is not None:
            logger.info(f"🚀 Running Final Optimization (Step 2) with L={final_L_m*1000:.6f} mm using {algo_step2}...")
            
            worker_cfg_step2 = {
                'algo':         algo_step2, 
                'whole_freq':   True,
                'roughness':    rough, 
                'fp':           fp,
                'bounds':       bounds_info}
            
            _, final_data = core.run_param_extraction((pair, final_L_m, worker_cfg_step2))

            if final_data:
                n_opt, k_opt, H_calc, _, cost, bounds = final_data
                
                offset_val = global_state.get('offset_mm', 0.0)
                if offset_val is None: offset_val = 0.0

                meta_algos = {
                    "step1_sweep_algo":     algo_step1,
                    "step2_final_algo":     algo_step2,
                    "user_offset_applied":  str(global_state.get('apply_all', False)),
                    "offset_val_mm":        f"{offset_val:.6f}",
                    "roughness":            str(rough),
                    "fp":                   str(fp)}

                if not getattr(cfg.saving, 'no_save_mode', False):
                    core.save_comprehensive_results(
                        out_dir=    out_dir, 
                        tag=        tag, 
                        pair=       pair, 
                        best_L=     final_L_m,
                        n=          n_opt, 
                        k=          k_opt, 
                        H_model=    H_calc, 
                        bounds=     bounds, 
                        meta_algos= meta_algos, 
                        cfg=        cfg, 
                        file_stem=  current_file_stem, 
                        thickness_sweep_m=thickness_sweep,
                        tv_scores=  tv_scores) # Apply all ise tv_scores boştur, sorun olmaz.
                    logger.info("✅ Results saved.")
            else:
                logger.error(f"❌ Final optimization failed for {measurement_name}")

@app.command()
def main(
    config:         Path = typer.Option("config.yaml",          "--config",         help="Path to the main YAML configuration file.", exists=True, readable=True),
    root_dir:       Optional[Path] = typer.Option("/Users/alp/Desktop/Toptica_UoB_ROOT" ,         "--root-dir",       help="Path to the RAWDATA folder containing pulsefft subfolders."),
    backend:        str = typer.Option(None,                    "--backend",        help="Execution mode: 'serial' or 'multiprocessing'."),
    no_save:        bool = typer.Option(False,                  "--no-save",        help="Disable all saving (MAT/TXT/PKL)."),
    overrides:      Optional[List[str]] = typer.Argument(None,                      help="CLI overrides in 'key.subkey=value' format (e.g., 'run.interactive=True')."),):
    """
    Batch Process THz-TDS Pickles to extract n and k.
    """
    all_overrides = list(overrides) if overrides else []

    if root_dir is None or not root_dir.exists():
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        try:
            root = tk.Tk()
            root.withdraw()
            selected = filedialog.askdirectory(title="Select RAWDATA Root Directory")
        finally:
            try:
                root.destroy()
            except:
                pass
            # to remove tkinter from memory
            gc.collect()

        if not selected:
            typer.echo(typer.style("🚫 No directory selected. Exiting.", fg=typer.colors.RED), err=True)
            sys.exit(1)
        root_dir = Path(selected)
        all_overrides.append(f"paths.root_dir={root_dir}")

    if backend:                 all_overrides.append(f"param_estimation.backend={backend}")
    if no_save:
        all_overrides += [
            "saving.save_mat    =False",
            "saving.save_txt    =False",
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

    # Find pickle files
    logger.info(f"Scanning {root_dir} for .pkl files...")
    pkl_files = list(root_dir.rglob(str(cfg.paths.pickle_folder) + "/*.pkl"))  
    if not pkl_files:
        logger.warning(f"🚫 No .pkl files found inside {str(cfg.paths.pickle_folder)} subdirectories.")
        raise typer.Exit()
    if cfg.param_estimation.pkl_prefix is not None:
        [f for f in pkl_files if f.name.startswith(str(cfg.param_estimation.pkl_prefix))]
        if not pkl_files:
            logger.warning(f"🚫 No .pkl files found with ID {cfg.param_estimation.pkl_prefix}.")
            raise typer.Exit()
        
    logger.info(f"Found {len(pkl_files)} files. Starting batch processing...")
    logger.info(f"Configuration: Backend={backend}, Res={cfg.param_estimation.scanning_resolution}, Range=+/-{cfg.param_estimation.scanning_range_mm}mm")
    
    extraction_state = {'apply_all': False, 'offset_mm': None}
    # last_folder = None

    for idx, pkl in enumerate(pkl_files):

        if cfg.run.start_from_index is not None and idx < cfg.run.start_from_index:
            continue
        if cfg.run.process_only_index is not None and idx != cfg.run.process_only_index:
            continue
        
        # current_folder = pkl.parent
        
        # if last_folder is not None and current_folder != last_folder:
        #     logger.info(f"📂 Folder changed to: {current_folder.name}")
        #     logger.warning("🔄 Resetting 'Apply to All' state for the new folder.")
        extraction_state = {'apply_all': False, 'offset_mm': None}
        
        # last_folder = current_folder
        plt.close('all')

        # logger.info(f"[{idx}/{len(pkl_files)}] Processing: {pkl.name}")
        run_extraction_pipeline(pkl, root_dir, backend, cfg, extraction_state)
    
    typer.echo("✔️ Processing completed.")

if __name__ == "__main__":
    app()