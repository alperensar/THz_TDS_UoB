# tds_extraction_core.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Core logic for extracting optical parameters (n, k) from THz-TDS data
# Contains physics models, optimization routines, and parallel processing workers.
#
# author@: Alperen Sari - PhDing at the University of Birmingham
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# TO DO: instead of TV use different metric
# TO DO: Covergence Maps
# TO DO: Same formula for utils and saving DR and alpha_max
from __future__ import annotations
import gc

import pickle
import logging
import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.constants import c
from multiprocessing import Queue
from scipy.optimize import least_squares, minimize
from typing import Tuple, Optional, Dict, Any

import utils
logger = logging.getLogger(__name__)

ERR = 1E-6          # error for optimisation
EPS_VAL = 1e-20     # Small constant to avoid log(0)
PHASE_WEIGHT = 1.0  # Weight factor for phase error in least squares

def plot_final_results(save_path: Path, freqs: np.ndarray, n: np.ndarray, k: np.ndarray, 
                       H_meas: np.ndarray, H_model: np.ndarray, alpha: np.ndarray, 
                       alpha_max: np.ndarray, error: np.ndarray,
                       bounds: Optional[Tuple] = None):
    
    freq_THz = freqs / 1e12 
    
    # --- STYLE SETTINGS ---
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 10
    
    grid_kwargs = {"visible": True, "linestyle": '--', "alpha": 0.4}
    legend_kwargs = {"loc": "best", "fontsize": 10, "framealpha": 0.9}
    label_fontsize = 11
    title_fontsize = 12

    # Reduced height by ~40%
    fig, axs = plt.subplots(3, 2, figsize=(12, 8.5))
    fig.suptitle(f"Optimization Results: {save_path.stem}", fontsize=14, y=0.98)
    
    # --- Row 1: Optical Constants ---
    # (a) n
    ax_n = axs[0, 0]
    ax_n.plot(freq_THz, n, 'r-', linewidth=1.2, label='Extracted n') 
    if bounds:
        ax_n.fill_between(freq_THz, bounds[0], bounds[1], color='red', alpha=0.1, label='Bounds')
    ax_n.set_ylabel("n", fontsize=label_fontsize)
    ax_n.set_title("(a) Refractive Index", fontsize=title_fontsize)
    ax_n.grid(**grid_kwargs)
    ax_n.legend(**legend_kwargs)
    
    # (b) k
    ax_k = axs[0, 1]
    ax_k.plot(freq_THz, k, 'b-', linewidth=1.2, label='Extracted k')
    if bounds:
        ax_k.fill_between(freq_THz, bounds[2], bounds[3], color='blue', alpha=0.1, label='Bounds')
    ax_k.set_ylabel("k", fontsize=label_fontsize)
    ax_k.set_title("(b) Extinction Coeff", fontsize=title_fontsize)
    ax_k.grid(**grid_kwargs)
    ax_k.legend(**legend_kwargs)
    
    # --- Row 2: Transfer Function ---
    # (c) Magnitude
    ax_mag = axs[1, 0]
    ax_mag.plot(freq_THz, np.abs(H_meas), 'k-', alpha=0.6, linewidth=1.0, label='Meas')
    ax_mag.plot(freq_THz, np.abs(H_model), 'r--', linewidth=1.2, label='Model')
    ax_mag.set_yscale('log')
    ax_mag.set_ylabel("|H(f)|", fontsize=label_fontsize)
    ax_mag.set_title("(c) Magnitude", fontsize=title_fontsize)
    ax_mag.grid(**grid_kwargs)
    ax_mag.legend(**legend_kwargs)

    # (d) Phase
    ax_phs = axs[1, 1]
    ax_phs.plot(freq_THz, np.unwrap(np.angle(H_meas)), 'k-', alpha=0.6, linewidth=1.0, label='Meas')
    ax_phs.plot(freq_THz, np.unwrap(np.angle(H_model)), 'r--', linewidth=1.2, label='Model')
    ax_phs.set_ylabel("Phase (rad)", fontsize=label_fontsize)
    ax_phs.set_title("(d) Phase", fontsize=title_fontsize)
    ax_phs.grid(**grid_kwargs)
    ax_phs.legend(**legend_kwargs)

    # --- Row 3: Absorption & Error ---
    # (e) Absorption
    ax_alp = axs[2, 0]
    ax_alp.plot(freq_THz, alpha, 'r-', linewidth=1.2, label=r'$\alpha$')
    if alpha_max is not None:
        ax_alp.plot(freq_THz, alpha_max, 'b-', linewidth=1.0, alpha=0.8, label=r'$\alpha_{max}$')
    
    ax_alp.set_yscale('log') 
    ax_alp.set_ylabel(r"$\alpha$ (cm$^{-1}$)", fontsize=label_fontsize)
    ax_alp.set_xlabel("Frequency (THz)", fontsize=label_fontsize)
    ax_alp.set_title("(e) Absorption", fontsize=title_fontsize)
    ax_alp.grid(**grid_kwargs)
    ax_alp.legend(**legend_kwargs)
    
    # Auto-scaling logic for Alpha
    try:
        valid_vals = []
        if alpha is not None:
            valid_vals.append(alpha[np.isfinite(alpha) & (alpha > 0)])
        if alpha_max is not None:
            valid_vals.append(alpha_max[np.isfinite(alpha_max) & (alpha_max > 0)])
        
        if valid_vals:
            all_data = np.concatenate(valid_vals)
            if all_data.size > 0:
                y_min = np.min(all_data)
                y_max = np.max(all_data)
                if y_min > 0 and y_max > y_min:
                    ax_alp.set_ylim(y_min * 0.5, y_max * 2.0)
    except:
        pass

    # (f) Residual Error
    ax_err = axs[2, 1]
    ax_err.plot(freq_THz, error, 'm-', linewidth=1.2, label='Error')
    ax_err.set_ylabel("Cost", fontsize=label_fontsize)
    ax_err.set_xlabel("Frequency (THz)", fontsize=label_fontsize)
    ax_err.set_title("(f) Residual Error", fontsize=title_fontsize)
    ax_err.grid(**grid_kwargs)
    
    # Common X-axis limits
    x_min, x_max = np.min(freq_THz), np.max(freq_THz)
    for ax in axs.flat:
        ax.set_xlim(x_min, x_max)
        ax.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save PDF
    plt.savefig(save_path.with_suffix(".pdf"), format='pdf', bbox_inches='tight', pad_inches=0.1)
    
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(2)
    except Exception as e:
        print(f"Plot display skipped: {e}")

    plt.close('all') 
    fig.clf()
    gc.collect()

def calculate_physics_metrics(pair: Any, best_L: float, n: np.ndarray, k: np.ndarray, H_model: np.ndarray) -> Dict[str, Any]:
    """Centralized function to perform all physics calculations once. """
    freqs = pair.freqs_interest
    H_meas = pair.measured_H[pair.x_min_index : pair.x_max_index + 1]

    if H_meas.shape[0] != freqs.shape[0]:
        raise ValueError(f"H_meas ({H_meas.shape[0]}) != freqs ({freqs.shape[0]}). ROI mismatch.")

    # 1. Initialize Seed Parameters
    n_S, k_S = init_params_guess(pair, freqs, best_L)
    c_cm = c * 100.0  # cm/s

    # 2. Final Permittivity & Alpha
    # Permittivity: ε = (n - i k)^2 = (n^2-k^2) - i(2nk)
    eps_r  = n**2 - k**2
    eps_i  = 2.0 * n * k
    
    # Absorption (cm^-1)
    alpha   = (4.0 * np.pi * freqs * k)   / c_cm

    # 3. Seed Permittivity & Alpha
    epsS_r = n_S**2 - k_S**2
    epsS_i = 2.0 * n_S * k_S
    alpha_S = (4.0 * np.pi * freqs * k_S) / c_cm

    # 4. Loss tangent (tanδ = ε''/ε')
    denom_final = np.where(np.abs(eps_r) > 1e-30, eps_r, np.nan)
    denom_seed  = np.where(np.abs(epsS_r) > 1e-30, epsS_r, np.nan)
    loss_tan    = eps_i  / denom_final
    loss_tan_S  = epsS_i / denom_seed

    # 5. Dynamic Range → alpha_max (L cm)
    noise_floor_dB  = float(getattr(pair, "noise_floor_db", -60.0))
    noise_floor_val = 10**(noise_floor_dB/20)
    
    ref_amp = np.abs(pair.fft_reference[pair.x_min_index : pair.x_max_index + 1])
    DR      = ref_amp / (noise_floor_val + EPS_VAL)
    DR_dB   = 20.0 * np.log10(DR + EPS_VAL)
    
    alpha_max = (2.0 / (best_L * 100.0)) * np.log(DR)

    # 6. Deltas (Final - Seed)
    delta_N      = n - n_S
    delta_K      = k - k_S
    delta_alpha  = alpha - alpha_S
    delta_eps_r  = eps_r - epsS_r
    delta_eps_i  = eps_i - epsS_i

    # 7. Model performance & Errors
    err_complex = H_meas - H_model
    err_abs     = np.abs(err_complex)

    # 8. Smoothness metrics (TV)
    tv_n     = float(np.sum(np.abs(np.diff(n))))
    tv_k     = float(np.sum(np.abs(np.diff(k))))
    tv_total = tv_n + tv_k

    # Pack everything into a dictionary
    return {
        'freqs':        freqs,
        'H_meas':       H_meas,
        'H_model':      H_model,
        
        # Parameters
        'n':            n,
        'k':            k,
        'alpha':        alpha,
        'eps_r':        eps_r,
        'eps_i':        eps_i,
        'loss_tan':     loss_tan,
        
        # INITIALS
        'n_S':          n_S,
        'k_S':          k_S,
        'alpha_S':      alpha_S,
        'epsS_r':       epsS_r,
        'epsS_i':       epsS_i,
        'loss_tan_S':   loss_tan_S,
        
        # DR Limits
        'DR':           DR,
        'DR_dB':        DR_dB,
        'alpha_max':    alpha_max,
        
        # Deltas
        'delta_N':      delta_N,
        'delta_K':      delta_K,
        'delta_alpha':  delta_alpha,
        'delta_eps_r':  delta_eps_r,
        'delta_eps_i':  delta_eps_i,
        
        # Errors
        'err_complex':  err_complex,
        'err_abs':      err_abs,
        
        # TV
        'tv_n':         tv_n,
        'tv_k':         tv_k,
        'tv_total':     tv_total}

def save_full_excel(out_dir: Path, tag: str, best_L: float, metrics: Dict[str, Any], file_stem: str, bounds: Optional[Tuple] = None):
    """ Saves the Excel file using the pre-calculated metrics dictionary. """
    m = metrics 
    freqs = m['freqs']

    data = {
        'freq_Hz':       freqs,
        'thickness_m':   np.full(freqs.shape, best_L, dtype=float),

        # FINAL
        'n_final':       m['n'],
        'k_final':       m['k'],
        'alpha_final':   m['alpha'],
        'eps_r_final':   m['eps_r'],
        'eps_i_final':   m['eps_i'],
        'loss_tan_final': m['loss_tan'],

        # INITIALS
        'n_seed':        m['n_S'],
        'k_seed':        m['k_S'],
        'alpha_seed':    m['alpha_S'],
        'eps_r_seed':    m['epsS_r'],
        'eps_i_seed':    m['epsS_i'],
        'loss_tan_seed': m['loss_tan_S'],

        # LIMIT
        'alpha_max':     m['alpha_max'],

        # DELTA
        'delta_n':       m['delta_N'],
        'delta_k':       m['delta_K'],
        'delta_alpha':   m['delta_alpha'],
        'delta_eps_r':   m['delta_eps_r'],
        'delta_eps_i':   m['delta_eps_i'],

        # H (complex components)
        'H_meas_real':   m['H_meas'].real,
        'H_meas_imag':   m['H_meas'].imag,
        'H_model_real':  m['H_model'].real,
        'H_model_imag':  m['H_model'].imag,

        # H (mag/phase)
        'H_meas_mag':    np.abs(m['H_meas']),
        'H_meas_phi':    np.unwrap(np.angle(m['H_meas'])),
        'H_model_mag':   np.abs(m['H_model']),
        'H_model_phi':   np.unwrap(np.angle(m['H_model'])),

        # Error
        'error_abs':     m['err_abs'],
        'error_real':    m['err_complex'].real,
        'error_imag':    m['err_complex'].imag,

        # Smoothness metrics
        'TV_n':          np.full(freqs.shape, m['tv_n'], dtype=float),
        'TV_k':          np.full(freqs.shape, m['tv_k'], dtype=float),
        'TV_total':      np.full(freqs.shape, m['tv_total'], dtype=float),}

    df = pd.DataFrame(data)

    out_dir.mkdir(parents=True, exist_ok=True)
    filename = out_dir / f"{file_stem}_{tag}_L{best_L*1000:.2f}mm.xlsx"

    try:
        df.to_excel(filename, index=False)
        logger.info(f"Saved EXCEL: {filename.name}")
    except Exception as e:
        logger.error(f"Excel save failed: {e}")

    # Pass the unpacked data to plot function
    plot_final_results(filename, freqs, m['n'], m['k'], m['H_meas'], m['H_model'], m['alpha'], m['alpha_max'], m['err_abs'], bounds)

def save_comprehensive_results(out_dir: Path, tag: str, pair: Any, best_L: float, n: np.ndarray,
    k: np.ndarray,  H_model: np.ndarray, bounds: Optional[Tuple], meta_algos: Dict[str, str],
    cfg: Any, file_stem: str, thickness_sweep_m: Optional[np.ndarray] = None, tv_scores: Optional[Dict[float, float]] = None):
    """ Master save function: Excel+Plot + MAT + TXT + PKL"""
    # 0) Calculate Physics Metrics (Centralized)
    metrics = calculate_physics_metrics(pair, best_L, n, k, H_model)

    # 1) Excel & Plot (Using calculated metrics)
    save_full_excel(out_dir, tag, best_L, metrics, file_stem, bounds)

    # Respect no_save_mode (MAT/TXT/PKL disabled)
    no_save_mode = bool(getattr(getattr(cfg, "saving", object()), "no_save_mode", False))
    if no_save_mode:
        logger.info("no_save_mode=True -> Skipping MAT/TXT/PKL saving.")
        return

    saving_cfg = getattr(cfg, "saving", object())
    save_mat = bool(getattr(saving_cfg, "save_mat", True))
    save_txt = bool(getattr(saving_cfg, "save_txt", True))
    save_pickle = bool(getattr(saving_cfg, "save_pickle", True))
    complex_mode = str(getattr(saving_cfg, "complex_mode", "both")).lower()

    # 3) Sweep TV formatting for MATLAB
    tv_sweep = np.array([])
    if tv_scores:
        tv_L = np.array(sorted(tv_scores.keys()), dtype=float)  # meters
        tv_V = np.array([tv_scores[L] for L in tv_L], dtype=float)
        tv_sweep = np.column_stack([tv_L, tv_V])  # Nx2 [L_m, TV]

    thickness_sweep_vec = (np.asarray(thickness_sweep_m, dtype=float) if thickness_sweep_m is not None else np.array([]))

    # 4) MAT/TXT base name
    base_name = f"{file_stem}_{tag}"
    m = metrics # Short handle

    # 5) MAT saving
    if save_mat:
        mat_data = {
            # Time domain
            "sample_time":      np.asarray(pair.sample.time_data),
            "sample_time_amp":  np.asarray(pair.sample.y_data),
            "ref_time":         np.asarray(pair.reference.time_data),
            "ref_time_amp":     np.asarray(pair.reference.y_data),

            # Full frequency domain
            "fft_freq_full":    np.asarray(pair.fft_freqs),
            "fft_sample_full":  np.asarray(pair.fft_sample),
            "fft_ref_full":     np.asarray(pair.fft_reference),

            # ROI / extracted
            "freq_roi":         m['freqs'],
            "n_opt":            np.asarray(m['n']),
            "k_opt":            np.asarray(m['k']),
            "alpha":            np.asarray(m['alpha']), # Replaces alpha_cm

            # Expanded Physics Parameters
            "eps_real":         np.asarray(m['eps_r']),
            "eps_imag":         np.asarray(m['eps_i']),
            "eps_complex":      np.asarray(m['eps_r'] - 1j*m['eps_i']),
            "loss_tan":         np.asarray(m['loss_tan']),

            "DR":               np.asarray(m['DR']),
            "DR_dB":            np.asarray(m['DR_dB']),
            "alpha_max":        np.asarray(m['alpha_max']),

            "H_meas":           np.asarray(m['H_meas']),
            "H_model":          np.asarray(m['H_model']),

            "thickness_mm":     float(best_L * 1000.0),

            # Residuals & Errors
            "error_complex":    np.asarray(m['err_complex']), 
            "error_abs":        np.asarray(m['err_abs']), 
            "residual_mag":     np.abs(m['H_meas']) - np.abs(m['H_model']),
            "residual_phase":   np.unwrap(np.angle(m['H_meas'])) - np.unwrap(np.angle(m['H_model'])),

            # Sweep metadata
            "thickness_sweep_m": thickness_sweep_vec,
            "tv_sweep":          tv_sweep,

            "meta_algos":        meta_algos, }

        try:
            scipy.io.savemat(str(out_dir / f"{base_name}_L{best_L*1000:.2f}mm.mat"), mat_data)
            logger.info(f"Saved MAT: {base_name}.mat")
        except Exception as e:
            logger.error(f"MAT error: {e}") 

    # 6) TXT saving helper: complex expansion
    def _expand_complex(vec: np.ndarray, name: str):
        vec = np.asarray(vec)
        cols = []
        headers = []
        if np.iscomplexobj(vec):
            if complex_mode == "real_imag":
                cols += [vec.real, vec.imag]
                headers += [f"{name}_real", f"{name}_imag"]
            elif complex_mode == "mag_phase":
                cols += [np.abs(vec), np.unwrap(np.angle(vec))]
                headers += [f"{name}_mag", f"{name}_phase"]
            else:  # both
                cols += [vec.real, vec.imag, np.abs(vec), np.unwrap(np.angle(vec))]
                headers += [f"{name}_real", f"{name}_imag", f"{name}_mag", f"{name}_phase"]
        else:
            cols += [vec.astype(float)]
            headers += [name]
        return headers, cols

    # 7) TXT saving (tabular)
    if save_txt:
        try:
            headers = []
            cols = []

            # always scalar/real vectors first
            for name, vec in [
                ("freq_roi_Hz", m['freqs']),
                ("n_opt",       m['n']),
                ("k_opt",       m['k']),
                ("alpha_cm",    m['alpha']),
                ("eps_r",       m['eps_r']),
                ("eps_i",       m['eps_i']),
                ("loss_tan",    m['loss_tan']),
                ("DR",          m['DR']),
                ("alpha_max",   m['alpha_max']),]:
                h, c = _expand_complex(vec, name)
                headers += h
                cols += c

            # optionally include H_meas / H_model (complex)
            h, c = _expand_complex(m['H_meas'], "H_meas")
            headers += h
            cols += c

            h, c = _expand_complex(m['H_model'], "H_model")
            headers += h
            cols += c

            # Stack
            cols = [np.asarray(v).reshape(-1) for v in cols]
            max_len = max(len(v) for v in cols)
            table = np.full((max_len, len(cols)), np.nan, dtype=float)

            for j, v in enumerate(cols):
                table[: len(v), j] = v.astype(float)

            txt_path = out_dir / f"{base_name}_L{best_L*1000:.2f}mm.txt"
            header_line = "\t".join(headers)
            np.savetxt(
                txt_path,
                table,
                delimiter="\t",
                header=header_line,
                comments="",
                fmt="%.18e",
            )
            logger.info(f"Saved TXT: {txt_path}")
        except Exception as e:
            logger.error(f"TXT error: {e}")

    # 8) PKL saving (store extracted results in pair and dump)
    if save_pickle:
        try:
            pair.extracted_results = {
                "n": np.asarray(n),
                "k": np.asarray(k),
                "L_m": float(best_L),
                "tag": str(tag),
                "meta_algos": dict(meta_algos),
            }
            pkl_path = out_dir / f"{base_name}_extracted_L{best_L*1000:.2f}mm.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump(pair, f)
            logger.info(f"Saved PKL: {pkl_path.name}")
        except Exception as e:
            logger.error(f"PKL error: {e}")

# PHYSICS MODELS
def init_params_guess(pair: Any, freqs_array: np.ndarray, L: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimates initial guesses for refractive index (ns_guess) and extinction coefficient (ks_guess) at a specific frequency.
    Mira's book 2.23a and 2.23b initializing the complex refractive index
    """
    omega_L = 2 * np.pi * freqs_array * L
    omega_L[omega_L == 0] = EPS_VAL
    co = c / omega_L
    
    indices = utils.find_index(pair.fft_freqs, freqs_array)

    phase = pair.unwrapped_phase[indices]
    ns_guess = 1 - co * phase

    ln_arg = (4 * ns_guess) / ((ns_guess + 1)**2 + EPS_VAL)
    H_abs = np.abs(pair.measured_H[indices]) + EPS_VAL
    ks_guess = co * (np.log(ln_arg) - np.log(H_abs))
    
    return ns_guess, ks_guess

def sigma_sum_FP(pair: Any, ns: np.ndarray, ks: np.ndarray, L: float, omega: np.ndarray, roughness: bool = False) -> np.ndarray:
    """Computes the Fabry–Pérot multiple reflection sum using a geometric series."""
    n_tilde = ns - 1j * ks
    total = np.zeros_like(n_tilde, dtype=complex)

    R = ((n_tilde - 1) / (n_tilde + 1))**2

    rough_val = pair.sigma_um * 1E-6 if roughness else 0.0
    
    # Limit FP pulses to avoid infinite loops if not set
    fp_limit = pair.fp_term_count or 0
    for m in range(max(1, fp_limit + 1)):
        phase_term = np.exp(-1j * 2 * m * n_tilde * omega * L / c)
        envelope = np.exp(- ((2 * m + 1)**2) * (n_tilde**2) * omega**2 * rough_val**2 / (2 * c**2))
        total += (R**m) * phase_term * envelope

    return total

def calculate_model_H(pair: Any, ns: np.ndarray, ks: np.ndarray, L: float, roughness: bool = False, FP: bool = False, omega: np.ndarray = None) -> np.ndarray:
    """
    Calculates the theoretical Transfer Function H(w) based on the physical model.
    Modes: Standard, Roughness corrected, FP corrected, or both.
    """
    c2 = 2 * c**2
    
    if omega is None:
        omega = 2 * np.pi * pair.freqs_interest
    
    omega2 = omega**2 

    n_tilde = ns - 1j * ks
    n_tilde2 = n_tilde**2
    coeff = 4 * n_tilde / ((n_tilde + 1)**2) 

    exponent1_term = -1j * (n_tilde - 1) * omega * L / c

    if roughness:
        sigma = pair.sigma_um * 1E-6
        sigma2 = sigma**2
        exponent2_term = np.exp((sigma2 * omega2) / c2)
        
        if FP:
            fp_term = sigma_sum_FP(pair, ns, ks, L, omega, roughness=True)
            H_model = coeff * np.exp(exponent1_term) * exponent2_term * fp_term
        else:
            H_model = coeff * np.exp(exponent1_term) * exponent2_term * np.exp(- n_tilde2 * omega2 * sigma2 / c2)
    
    else:
        if FP:                                                     
            fp_term = sigma_sum_FP(pair, ns, ks, L, omega, roughness=False)
            H_model = coeff * np.exp(exponent1_term) * fp_term        
        else:
            H_model = coeff * np.exp(exponent1_term) 
            
    return H_model

def errorFun_Nelder_Mead_total(params_vector: np.ndarray, pair: Any, freqs_interest: np.ndarray, L: float, roughness: bool = False, FP: bool = False) -> float:
    """Computes total error. """
    num_freqs = len(freqs_interest)

    if params_vector.size != 2 * num_freqs:
        raise ValueError(f"Vector size mismatch. Expected {2*num_freqs}, got {params_vector.size}")

    ns = params_vector[:num_freqs]
    ks = params_vector[num_freqs:]

    full_model = calculate_model_H(pair, ns, ks, L, roughness, FP)

    low_bound = pair.x_min_index
    upp_bound = pair.x_max_index
    measured_H_roi = pair.measured_H[low_bound:upp_bound+1]
    
    diff = full_model - measured_H_roi
    total_error = np.sum(np.abs(diff))
    
    return total_error

def errorFun_Nelder_Mead_individual(params_pair: np.ndarray, pair: Any, freq_point: float, L: float, roughness: bool, FP: bool, idx: int) -> float:
    """ Computes error for a SINGLE frequency point."""
    n_val, k_val = params_pair
    
    omega = 2 * np.pi * freq_point
    n_tilde = n_val - 1j * k_val
    
    coeff = 4 * n_tilde / ((n_tilde + 1)**2)
    exponent = -1j * (n_tilde - 1) * omega * L / c
    H_model = coeff * np.exp(exponent)
    
    meas_H = pair.measured_H[pair.x_min_index + idx]
    
    return np.abs(H_model - meas_H)

def residual_mag_phase(H_meas, H_model, phase_weight=1.0, w=None, unwrap=False):
    mag = np.log(np.abs(H_meas) + EPS_VAL) - np.log(np.abs(H_model) + EPS_VAL)
    ph  = np.angle(H_meas * np.conj(H_model))

    if unwrap and np.ndim(ph) == 1:
        ph = np.unwrap(ph)

    if w is None:
        return np.concatenate([mag, phase_weight * ph])
    else:
        return np.concatenate([w * mag, w * (phase_weight * ph)])

def errorFun_least_squares(params_vector, pair, freqs_array, L, roughness, FP):
    """errorFun for least square method"""
    num_freqs = len(freqs_array)
    ns_array = params_vector[:num_freqs]
    ks_array = params_vector[num_freqs:]

    H_meas = pair.measured_H[pair.x_min_index : pair.x_max_index + 1]
    H_model = calculate_model_H(pair, ns_array, ks_array, L, roughness, FP)

    amp = np.abs(H_meas)
    med = np.median(amp) + EPS_VAL
    w = np.clip(amp / med, 0.2, 3.0)

    return residual_mag_phase(H_meas, H_model, phase_weight=PHASE_WEIGHT, w=w, unwrap=True)

def run_param_extraction(args: Tuple[Any, float, Dict]) -> Tuple[float, Optional[Tuple]]:
    """  Master function to run extraction based on configuration. """
    try:
        pair, L, cfg = args

        freqs = pair.freqs_interest
        num_freqs = len(freqs)

        algo       = cfg.get('algo', 'Optimise')
        whole_freq = cfg.get('whole_freq', True)
        rough      = cfg.get('roughness', False)
        fp         = cfg.get('fp', False)

        bounds_cfg = cfg.get('bounds', {})

        def get_val(obj, key, default):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # ONLY percentage windows (defaults if not given)
        n_window = float(bounds_cfg.n_bound)  # e.g. 0.02 => ±2%
        k_window = float(bounds_cfg.k_bound)  # e.g. 0.40 => ±40%

        # Initial Guess
        ns_guess, ks_guess = init_params_guess(pair, freqs, L)
        if len(ns_guess) != num_freqs:
            ns_guess = np.resize(ns_guess, num_freqs)
            ks_guess = np.resize(ks_guess, num_freqs)

        ns_guess = np.asarray(ns_guess, dtype=float)
        ks_guess = np.asarray(ks_guess, dtype=float)

        initial_guess = np.concatenate([ns_guess, ks_guess])

        cost_val = 0.0
        bounds_tuple = None
        n_final, k_final = None, None

        # NELDER–MEAD (no bounds)
        if algo == "Nelder-Mead":
            if whole_freq:
                res = minimize(errorFun_Nelder_Mead_total, initial_guess, args=(pair, freqs, L, rough, fp),
                    method='Nelder-Mead', options={'maxiter': 10000, 'xatol': ERR})
                n_final  = res.x[:num_freqs]
                k_final  = res.x[num_freqs:]
                cost_val = res.fun
                bounds_tuple = None

            else:
                n_list, k_list = [], []
                total_cost = 0.0

                for i, f in enumerate(freqs):
                    p0 = [float(ns_guess[i]), float(ks_guess[i])]
                    res = minimize(errorFun_Nelder_Mead_individual, p0, args=(pair, f, L, rough, fp, i),
                        method='Nelder-Mead', options={'maxiter': 1000, 'xatol': ERR, 'fatol': ERR})
                    
                    n_list.append(res.x[0])
                    k_list.append(res.x[1])
                    total_cost += res.fun

                n_final = np.array(n_list, dtype=float)
                k_final = np.array(k_list, dtype=float)
                cost_val = total_cost
                bounds_tuple = None

        # OPTIMISE (bounds = % windows)
        else:
            # A) GLOBAL (WHOLE FREQUENCY)
            if whole_freq:
                H_meas_all = pair.measured_H[pair.x_min_index : pair.x_max_index + 1]
                assert H_meas_all.shape[0] == num_freqs, (H_meas_all.shape[0], num_freqs)

                # Build per-frequency bounds around the SEED (ns_guess, ks_guess)
                # n bounds: ± n_window
                n_delta = np.maximum(np.abs(ns_guess) * n_window, 1e-6)
                n_lb = ns_guess - n_delta
                n_ub = ns_guess + n_delta

                # keep n physically > 0 (tiny clamp)
                n_lb = np.maximum(n_lb, 1e-6)

                # k bounds: ± k_window * max(|k|, k_floor)
                k_floor = 1e-6
                k_scale = np.maximum(np.abs(ks_guess), k_floor)
                k_delta = k_scale * k_window + 0.03
                k_lb = ks_guess - k_delta
                k_ub = ks_guess + k_delta

                # Combine into solver bounds
                lb = np.concatenate([n_lb, k_lb])
                ub = np.concatenate([n_ub, k_ub])

                def global_cost(p):
                    ns_curr = p[:num_freqs]
                    ks_curr = p[num_freqs:]
                    H_sim = calculate_model_H(pair, ns_curr, ks_curr, L, rough, fp)

                    amp = np.abs(H_meas_all)
                    med = np.median(amp) + EPS_VAL
                    w = np.clip(amp / med, 0.2, 3.0)

                    return residual_mag_phase(H_meas_all, H_sim, phase_weight=PHASE_WEIGHT, w=w, unwrap=True)

                res = least_squares(global_cost, initial_guess, bounds=(lb, ub), method='trf', loss='soft_l1', verbose=0)

                n_final = res.x[:num_freqs]
                k_final = res.x[num_freqs:]
                cost_val = res.cost

                # naming you asked for:
                n_bound = (n_lb, n_ub)
                k_bound = (k_lb, k_ub)
                bounds_tuple = (n_bound[0], n_bound[1], k_bound[0], k_bound[1])

            # B) POINT-BY-POINT (bounds = % window around previous solution)
            else:
                n_final = np.zeros(num_freqs, dtype=float)
                k_final = np.zeros(num_freqs, dtype=float)

                n_lb_h = np.zeros(num_freqs, dtype=float)
                n_ub_h = np.zeros(num_freqs, dtype=float)
                k_lb_h = np.zeros(num_freqs, dtype=float)
                k_ub_h = np.zeros(num_freqs, dtype=float)

                current_n = float(ns_guess[0])
                current_k = float(ks_guess[0])

                total_cost = 0.0

                for i in range(num_freqs):
                    seed_n = float(ns_guess[i])
                    seed_k = float(ks_guess[i])

                    # chaining reference
                    if i == 0:
                        ref_n, ref_k = seed_n, seed_k
                    else:
                        ref_n, ref_k = current_n, current_k

                    # bounds around ref using % windows
                    dn = max(abs(ref_n) * n_window, 1e-6)
                    n_lb = max(ref_n - dn, 1e-6)
                    n_ub = ref_n + dn

                    k_floor = 1e-6
                    dk = max(max(abs(ref_k), k_floor) * k_window, 1e-6)
                    k_lb = ref_k - dk
                    k_ub = ref_k + dk

                    n_lb_h[i] = n_lb
                    n_ub_h[i] = n_ub
                    k_lb_h[i] = k_lb
                    k_ub_h[i] = k_ub

                    bounds = (np.array([n_lb, k_lb]), np.array([n_ub, k_ub]))
                    p0 = np.array([ref_n, ref_k], dtype=float)

                    freq_point = freqs[i]
                    H_meas_i = pair.measured_H[pair.x_min_index + i]

                    def single_point_cost(p):
                        n_val, k_val = float(p[0]), float(p[1])
                        omega_i = 2 * np.pi * float(freq_point)

                        H_sim_i = calculate_model_H(pair, n_val, k_val, L, rough, fp, omega=omega_i)
                        return residual_mag_phase( np.array([H_meas_i]), np.array([H_sim_i]), phase_weight=PHASE_WEIGHT, w=None, unwrap=True)

                    try:
                        res_i = least_squares(single_point_cost, p0,  bounds=bounds,method='trf', loss='soft_l1', max_nfev=80, verbose=0)
                        new_n, new_k = float(res_i.x[0]), float(res_i.x[1])
                        total_cost += float(res_i.cost)
                    except Exception:
                        new_n, new_k = ref_n, ref_k

                    # update
                    n_final[i] = new_n
                    k_final[i] = new_k
                    current_n, current_k = new_n, new_k

                cost_val = total_cost

                n_bound = (n_lb_h, n_ub_h)
                k_bound = (k_lb_h, k_ub_h)
                bounds_tuple = (n_bound[0], n_bound[1], k_bound[0], k_bound[1])

        H_calc = calculate_model_H(pair, n_final, k_final, L, rough, fp)
        return L, (n_final, k_final, H_calc, L, cost_val, bounds_tuple)

    except Exception as e:
        print(f"Extraction failed for L={L}: {e}")
        return L, None
    
def worker_process(task_queue: Queue, result_queue: Queue):
    """Worker for multiprocessing."""
    while True:
        task = task_queue.get()
        if task is None:
            break

        pair, L, cfg, retry = task

        try:
            L_out, data = run_param_extraction((pair, L, cfg))
            result_queue.put((L_out, data))
        except Exception:
            if retry < 1:
                task_queue.put((pair, L, cfg, retry + 1))
            else:
                result_queue.put((L, None))

def L_key_um(L_m: float) -> int:
    # metre → mikrometre (µm)
    return int(round(float(L_m) * 1e6))

def calculate_total_variation(data_dict: dict) -> dict[int, float]:
    tv_dict = {}
    for L, val in data_dict.items():
        if val is None:
            continue
        n_val, k_val = val[0], val[1]
        tv = np.sum(np.abs(np.diff(n_val))) + np.sum(np.abs(np.diff(k_val)))
        tv_dict[L_key_um(L)] = float(tv)
    return tv_dict


# def errorFun_least_squares(params_vector: np.ndarray, pair: Any, freqs_array: np.ndarray, L: float, roughness: bool, FP: bool) -> np.ndarray:
#     """Residual function with PHASE PRIORITY."""
#     num_freqs = len(freqs_array)
#     ns_array = params_vector[:num_freqs]
#     ks_array = params_vector[num_freqs:]
    
#     H_meas = pair.measured_H[pair.x_min_index : pair.x_max_index + 1]
#     H_model = calculate_model_H(pair, ns_array, ks_array, L, roughness, FP)
    
#     # Logaritmik Fark
#     log_meas  = np.log(H_meas + EPS_VAL)
#     log_model = np.log(H_model + EPS_VAL)
#     diff = log_meas - log_model
        
#     res_real = diff.real       # Genlik Hatası (Kappa/Alpha için)
#     res_imag = diff.imag * PHASE_WEIGHT # Faz Hatası (n için)
    
#     return np.concatenate([res_real, res_imag])

# ERROR FUNCTIONS
# def compute_mag_phase_errors(pair: Any, full_model, freqs_interest: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
#     """Error calculation between measured and model H in terms of magnitude and phase for Nelder_Mead."""
#     low_bound = pair.x_min_index
#     upp_bound = pair.x_max_index

#     measured_H_roi = pair.measured_H[low_bound:upp_bound+1]
#     zero_phase_roi = pair.unwrapped_phase[low_bound:upp_bound+1]

#     # Magnitude error
#     if full_model.shape[0] != measured_H_roi.shape[0]:
#         raise ValueError(f"full_model len={full_model.shape[0]} is not match  measured_H_roi len={measured_H_roi.shape[0]}.")
#     M_omega = np.abs(full_model) - np.abs(measured_H_roi)

#     # Phase unwrapping for model
#     raw_phase = np.angle(full_model)
#     unwrapped_phase, _ = utils.unwrap_phase(raw_phase)

#     i1 = utils.find_index(freqs_interest, pair.unwrap_fit_min)
#     i2 = utils.find_index(freqs_interest, pair.unwrap_fit_max)
#     popt = np.polyfit(freqs_interest[i1:i2+1], unwrapped_phase[i1:i2+1], 1)
#     interpolated_phase = unwrapped_phase - np.abs(popt[1])
   
#     A_omega = interpolated_phase - zero_phase_roi

#     return M_omega, A_omega

# def errorFun_Nelder_Mead_total(pair: Any, params_vector: np.ndarray, freqs_interest: np.ndarray,
#     L: float, roughness: bool = False, FP: bool = False, verbose: bool = False) -> float:
#     """
#     Computes the total error (magnitude + phase) for a given set of parameters using manual unwrapping.
#     This outputs the error function that is to be minimized via Nelder-Mead
#     As in Mira Naftaly's book calculate for whole frequency range not individual as in previous
#     """
#     freqs_interest = np.asarray(freqs_interest)
#     num_freqs = len(freqs_interest)

#     if params_vector.size != 2 * num_freqs:
#         raise ValueError("Parameter vector size does not match expected size.")

#     # n(f) and k(f) vectors
#     ns = params_vector[:num_freqs]
#     ks = params_vector[num_freqs:]

#     # Model H(f) calculation
#     full_model = calculate_model_H(pair, ns, ks, L, roughness, FP)

#     # Amplitude and Phase errors
#     M_omega, A_omega = compute_mag_phase_errors(pair, full_model, freqs_interest)

#     # Toplam hata (L1 normu)
#     mag_error_sum = np.sum(np.abs(M_omega))
#     phase_error_sum = np.sum(np.abs(A_omega))
#     total_error = mag_error_sum + phase_error_sum

#     if verbose:
#         print(f"M_omega Sum: {mag_error_sum:.6f} | A_omega Sum: {phase_error_sum:.6f} " f"| Total: {total_error:.6f}")

#     return total_error

# def errorFun_Nelder_Mead_individual(pair: Any, ns: float, ks: float, freqs_interest: np.ndarray, 
#     L: float, roughness: bool = False, FP: bool = False, verbose: bool = False) -> float:
#     """ 
#      Computes the total error (magnitude + phase) for a given set of parameters using manual unwrapping.
#      This outputs the error function that is to be minimized via Nelder-Mead
#     """
#     freqs_interest = np.asarray(freqs_interest)
#     num_freqs = len(freqs_interest)

#     ns_vec = np.full(num_freqs, float(ns), dtype=float)
#     ks_vec = np.full(num_freqs, float(ks), dtype=float)

#     full_model = calculate_model_H(pair, ns_vec, ks_vec, L, roughness, FP)
#     M_omega, A_omega = compute_mag_phase_errors(pair, full_model, freqs_interest)

#     mag_error_sum = np.sum(np.abs(M_omega))
#     phase_error_sum = np.sum(np.abs(A_omega))
#     total_error = mag_error_sum + phase_error_sum

#     if verbose:
#         print(f"[errorFun] M_sum={mag_error_sum:.6f} | A_sum={phase_error_sum:.6f} " f"| Total={total_error:.6f}")

#     return total_error
# --- ERROR FUNCTIONS (DÜZELTİLMİŞ SIRALAMA) ---

# def plot_tv_sweep(out_dir: Path, tag: str, tv_scores: dict, best_L: float):
#     """
#     Plots the Total Variation (smoothness metric) vs Thickness.
#     Helps to visualize if the minimum is distinct or flat.
#     """
#     l_vals_m = sorted(tv_scores.keys())
#     l_vals_mm = [l * 1000.0 for l in l_vals_m]
#     tv_vals = [tv_scores[l] for l in l_vals_m]
    
#     best_L_mm = best_L * 1000.0
#     best_tv = tv_scores[best_L]

#     plt.figure(figsize=(10, 6))
#     plt.plot(l_vals_mm, tv_vals, 'o-', color='navy', alpha=0.7, label='TV Score')
#     plt.plot(best_L_mm, best_tv, 'r*', markersize=15, label=f"Selected: {best_L_mm:.4f} mm")
    
#     plt.xlabel("Thickness (mm)")
#     plt.ylabel("Total Variation (Lower is Smoother)")
#     plt.title(f"Thickness Optimization Sweep: {tag}")
#     plt.grid(True, which='both', linestyle='--', alpha=0.5)
#     plt.legend()
    
#     filename = out_dir / f"{tag}_TV_Selection_L={best_L_mm:.4f}mm.png"
#     plt.savefig(filename, dpi=300)
#     plt.close()
#     logger.info(f"Saved TV Sweep plot to: {filename}")
