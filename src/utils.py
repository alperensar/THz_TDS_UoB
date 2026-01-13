# utils.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# author@: Alperen Sari - PhDing at the University of Birmingham
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
import os

import numpy as np
from scipy.constants import c
from scipy.signal import find_peaks

THZ_TO_HZ = 1e12
ERROR_EPSILON = 1E-20


def find_index(array, target):
    """Finds the index/indices in 'array' closest to the value(s) in 'target'."""
    array = np.asarray(array)

    if np.isscalar(target):
        return np.abs(array - target).argmin()
    else:
        # If 'target' is a list or array
        target = np.asarray(target)
        diffs = np.abs(array[:, np.newaxis] - target)
        return diffs.argmin(axis=0)
    
def update_indices(self):
    """It is updating extraction and unwrapping ranges on variable"""
    self.x_min_index = find_index(self.fft_freqs, self.f_min_extract)
    self.x_max_index = find_index(self.fft_freqs, self.f_max_extract)
    self.unwrap_min_idx = find_index(self.fft_freqs, self.unwrap_fit_min)
    self.unwrap_max_idx = find_index(self.fft_freqs, self.unwrap_fit_max)
    self.freqs_interest = self.fft_freqs[self.x_min_index:self.x_max_index+1]   # Add 1 such that the final index is inclusive
    self.unwrap_freqs_interest = self.fft_freqs[self.unwrap_min_idx:self.unwrap_max_idx+1]

def recalculate_absorption(self):
    """
    Recalculates Refractive Index (n), Absorption (alpha), and Max Absorption (alpha_max).
    following the 10.1364/OL.30.000029
    Methodology:
    1. Refractive Index (n): Calculated from unwrapped phase delay.
    2. Dynamic Range (DR): Estimated from reference signal and noise floor.
    3. Absorption (alpha): Derived from Transmission and Fresnel losses.
    4. Max Absorption (alpha_max): Theoretical limit based on DR.
    """
    freq_Hz = self.fft_freqs         # whole fequency range 
    mask_pos = freq_Hz > 0.0
    measured_thickness_cm = self.measured_thickness_mm / 10.0

    # Phase selection (Primary vs Backup Method)
    if self.use_backup_unwrap and hasattr(self, "zero_unwrapped_phase_backup"):
        unwrapped_phase = self.zero_unwrapped_phase_backup
    else:
        unwrapped_phase = self.zero_unwrapped_phase

    # Calculate refractive index
    n = np.full_like(freq_Hz, np.nan, dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        # Formula: n = 1 - (c * phi) / (omega * d)
        n[mask_pos] = 1.0 - (c * unwrapped_phase[mask_pos]) / (2.0 * np.pi * freq_Hz[mask_pos] * self.measured_thickness_m)
    n[(~np.isfinite(n)) | (n <= 0.0)] = np.nan

    # Noise floor & Dynamic range (DR)
    noise_floor_db = float(getattr(self, "noise_floor_db", -80.0))

    Eref_mag = np.abs(self.fft_reference) + ERROR_EPSILON
    DR_dB_spectrum = 20.0 * np.log10(Eref_mag) - noise_floor_db
    
    # Dynamic Range Amplitude (Linear Scale), must be >= 1
    DR_amp = np.maximum(10.0 ** (DR_dB_spectrum / 20.0), 1.0)

    # Fresnel F2 is for Transmission -> Absorption correction
    F2 = ((n + 1.0) ** 2) / (4.0 * n)
    # Fresnel F3 is for max DR calculation
    F3 = (4.0 * n) / ((n + 1.0) ** 2)

    # Calculate alpha & alpha_max
    amp_H = np.abs(self.measured_H)
    arg2 = np.clip(amp_H * F2, ERROR_EPSILON, None)                # For alpha
    arg3 = np.clip(DR_amp * F3, ERROR_EPSILON, None)               # For alpha_max
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = (-2.0 / measured_thickness_cm) * np.log(arg2)      # [cm^-1]
        alpha_max = ( 2.0 / measured_thickness_cm) * np.log(arg3)  # [cm^-1]

    # Cleanup invalid values
    alpha[~np.isfinite(alpha)] = np.nan
    alpha[alpha < 0] = 0.0
    alpha_max[~np.isfinite(alpha_max)] = np.nan

    # 6) Create Validity Mask
    valid_mask = mask_pos & np.isfinite(n) & np.isfinite(alpha) & np.isfinite(alpha_max)

    # Store results
    self.current_n = n
    self.current_alpha = alpha
    self.current_alpha_max = alpha_max
    self.valid_mask = valid_mask
    self.DR_dB_spectrum = DR_dB_spectrum
    self.alpha_max = alpha_max

    # Update FP pulse locations based on new 'n'
    update_fp_from_n_avg(self)

def update_fp_from_n_avg(self):
    """
    Calculates the number of Fabry–Perot echoes and their timings using the average 
    refractive index from recalculate_absorption
    """
    n_arr = self.current_n
    freq_Hz = self.fft_freqs         # whole fequency range
    
    mask_band = (freq_Hz >= self.f_min_extract) & (freq_Hz <= self.f_max_extract)
    mask_valid = mask_band & np.isfinite(n_arr) & (n_arr > 0)
    if not np.any(mask_valid):
        self.fp_term_count  = 0
        self.fp_n_avg       = None
        self.fp_delta_t     = None
        self.fp_t_peak      = None
        self.fp_echo_times  = np.array([])
        return

    n_avg = np.nanmean(n_arr[mask_valid])

    thickness_m = self.measured_thickness_m
    if thickness_m <= 0 or n_avg <= 0:
        self.fp_term_count  = 0
        self.fp_n_avg       = None
        self.fp_delta_t     = None
        self.fp_t_peak      = None
        self.fp_echo_times  = np.array([])
        return

    delta_t = (2.0 * thickness_m * n_avg) / c    # Δt (seconds)
    x = np.asarray(self.time_data, dtype=float)  # Time axis (seconds)

    # Sample signal
    if hasattr(self, "sample") and hasattr(self.sample, "y_data"):
        y = self.sample.y_data
    elif hasattr(self, "sample_signal") and hasattr(self.sample_signal, "y_data"):
        y = self.sample_signal.y_data
    else:
        return

    if x.size == 0 or y.size == 0 or x.size != y.size:
        return

    # Main peak detection
    peaks, _ = find_peaks(np.abs(y))
    if len(peaks) == 0:
        self.fp_term_count  = 0
        self.fp_n_avg       = n_avg
        self.fp_delta_t     = delta_t
        self.fp_t_peak      = None
        self.fp_echo_times  = np.array([])
        return

    main_idx = peaks[np.argmax(np.abs(y[peaks]))]
    t_peak = float(x[main_idx])
    t_end  = float(x[-1])
    temporal_length = t_end - t_peak

    if temporal_length <= 0 or delta_t <= 0:
        self.fp_term_count  = 0
        self.fp_n_avg       = n_avg
        self.fp_delta_t     = delta_t
        self.fp_t_peak      = t_peak
        self.fp_echo_times  = np.array([])
        return

    # Echos
    raw_count = temporal_length / delta_t
    echo_count = int(np.floor(raw_count))
    echo_count = max(echo_count, 0)

    self.fp_term_count  = echo_count
    self.fp_n_avg       = n_avg
    self.fp_delta_t     = delta_t
    self.fp_t_peak      = t_peak

    if echo_count > 0:
        idx = np.arange(1, echo_count + 1, dtype=float)
        self.fp_echo_times = t_peak + idx * delta_t
    else:
        self.fp_echo_times = np.array([])

def update_unwrap_log(self, log_file: str):
    """Updates the log file with unwrap AND extraction limits using a dictionary to avoid duplicates."""
    data = {}

    # 1. Read existing data
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue

    # 1. Read existing data
    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue
                
                parts = line.split('\t')

                if len(parts) >= 3:
                    name = parts[0]
                    u_min, u_max = parts[1], parts[2]
                    e_min = parts[3] if len(parts) > 3 else "0.0000"
                    e_max = parts[4] if len(parts) > 4 else "0.0000"
                    
                    data[name] = (u_min, u_max, e_min, e_max)

    # 2. Update current sample
    val_u_min = f"{self.unwrap_fit_min / THZ_TO_HZ:.4f}"
    val_u_max = f"{self.unwrap_fit_max / THZ_TO_HZ:.4f}"
    val_e_min = f"{self.f_min_extract  / THZ_TO_HZ:.4f}"
    val_e_max = f"{self.f_max_extract  / THZ_TO_HZ:.4f}"

    data[self.sample_name] = (val_u_min, val_u_max, val_e_min, val_e_max)

    # 3. Write back (Sorted)
    header = "# Material\tUnwrapMin(THz)\tUnwrapMax(THz)\tExtMin(THz)\tExtMax(THz)\n"
    
    # 3. Write back
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(header)
        for name, (u_mn, u_mx, e_mn, e_mx) in sorted(data.items()):
            f.write(f"{name}\t{u_mn}\t{u_mx}\t{e_mn}\t{e_mx}\n")

def unwrap_phase(self, peak_height=2, peak_distance=1):
    """Manual unwrapping of the phase based on peaks."""
    phase_array = self.measured_phase
    peaks, _ = find_peaks(phase_array, height=peak_height, distance=peak_distance)
    
    peak_mask = np.zeros_like(phase_array, dtype=int)
    peak_mask[peaks] = 1

    phase_jumps = np.cumsum(peak_mask)
    self.measured_unwrapped_phase = phase_array - phase_jumps * (2 * np.pi)
    self.measured_phase_peaks = peaks

def phase_extrapolation(self):
    """
    Fits a line to unwrapped phase (and backup unwrap) in specified frequency range.
    Extrapolation of unwrapped phase to zero intercept:-
    """
    idx_start = self.unwrap_min_idx
    idx_end   = self.unwrap_max_idx
    freqs_seg = self.fft_freqs[idx_start:idx_end]

    # Primary method using our own unwrapping algorithm
    phase_seg = self.measured_unwrapped_phase[idx_start:idx_end]
    if len(freqs_seg) > 1:
        # Remove DC offset (intercept) from both unwrapped signals
        self.popt = np.polyfit(freqs_seg, phase_seg, 1)
        self.zero_unwrapped_phase = self.measured_unwrapped_phase - self.popt[1]
    else:
        self.popt = [0,0]; 
        self.zero_unwrapped_phase = self.measured_unwrapped_phase

    # Backup method using classic libraries unwrapping
    self.backup_unwrap = np.unwrap(self.measured_phase)
    phase_seg_back = self.backup_unwrap[idx_start:idx_end]
    if len(freqs_seg) > 1:
        # Remove DC offset (intercept) from both unwrapped signals
        self.popt_backup = np.polyfit(freqs_seg, phase_seg_back, 1)
        self.zero_unwrapped_phase_backup = self.backup_unwrap - self.popt_backup[1]
    else:
        self.popt_backup = [0,0]; 
        self.zero_unwrapped_phase_backup = self.backup_unwrap



# WILL BE REMOVED
# def clean_signal_name(file_path):
#     """
#     Extracts and cleans the name of the signal file for display or labeling.
#     """
#     file_name = Path(file_path).name           # OS independent taking path
#     cleaned_name = (file_name.upper()
#                     .replace("FOC", " ")
#                     .replace("COL", " ")
#                     .replace(".TXT", " ")
#                     .replace("_", " ")
#                     .replace("ROGERS", " "))
#     return cleaned_name.strip()

# def configure_axis(ax, ylabel=None, xlabel=None, title=None):
    # """Confugure the axis"""
    # if ylabel:  ax.set_ylabel(ylabel, fontsize=14)
    # if xlabel:  ax.set_xlabel(xlabel, fontsize=14)
    # if title:   ax.set_title(title, fontsize=14)
    # ax.yaxis.set_label_coords(-0.1, 0.5)
    # ax.tick_params(axis='both', labelsize=11)
    # ax.grid(True, linestyle='--', alpha=0.6)