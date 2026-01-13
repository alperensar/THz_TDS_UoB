# tds_fft_processor.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# tds_fft_processor:
#   - THzFFTprocessor: process time domain signals to convert frequency domain
#
# author@: Alperen Sari - PhDing at the University of Birmingham
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# TO DO: check windowing for FFT
from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import utils

THZ_TO_HZ = 1e12
ERROR_EPSILON = 1e-20
TIME_SCALE_PS = 1e-12


@dataclass(slots=True)
class SignalData:
    time_data: np.ndarray
    y_data:   np.ndarray

class THzFFTprocessor:
    """
    - This class apply FFT on both reference and sample signals by enabling selection of the frequency limits both extracted signal and 
    unwrapping range. 
    - Unwrapping range is selected using GUI or automatically using predefined values in the root dir. 
      • time-domain data loading (sample + reference)
      • preprocessing & DC removal
      • FFT analysis
      • Fabry-Perot estimation (for fixed-eps or auto)
      • Transfer function H(f) and phase extraction
      • Unwrapping, extrapolation, frequency range slicing
    """    
    def __init__(self, 
                 sample_file:           str, 
                 reference_file:        str, 
                 sample_name:           str,
                 instrument_name:       str,
                 *, 
                 sample_column:         int = 1, 
                 reference_column:      int = 1, 
                 sigma:                 float, 
                 measured_thickness:    float, 
                 time_unit:             str = "ps", 
                 unwrap_range_selec:    bool = True, 
                 preset=                None, 
                 unwrap_file_root=      None, 
                 eps_guess:             float = 1.0,
                 cfg,):
        """
        sample_file :                Path to sample waveform (.txt or .csv)
        reference_file :             Path to reference waveform
        sample_column :              Index of sample amplitude column (0=time, 1,2,... amplitude)
        reference_column :           Index of reference amplitude column (same as above)
        time_unit : {"ps","ns","s"}  Unit of input time axis
        normalization_coef :         Scalar used to normalize time before FFT (typically 1e-12)
        """
        # paths for the data
        self.sample_path = sample_file
        self.reference_path = reference_file

        # sample name and instrument name assignment
        self.sample_name = sample_name
        self.instrument_name = instrument_name

        # load data
        self.sample = self._load_signal(self.sample_path, sample_column)
        self.reference = self._load_signal(self.reference_path, reference_column)

        # time normalisation
        self.time_unit = time_unit
        self.time_scale = {"ps": 1e-12, "ns": 1e-9, "s": 1.0}[time_unit]
        self.time_data = self.sample.time_data * self.time_scale           

        # FFT calculation
        zero_pad = cfg.fft.zero_pad or 1.0
        # dc_offset = cfg.fft.dc_offset or None
        _, self.fft_reference = self.fft_analysis(self.reference, zero_pad=zero_pad, time_normalisation=self.time_scale, positive=True, dc_offset=None)
        self.fft_freqs, self.fft_sample = self.fft_analysis(self.sample, zero_pad=zero_pad, time_normalisation=self.time_scale, positive=True, dc_offset=None)

        # sigma and thickness assignment
        self.sigma_um = sigma
        self.measured_thickness_mm = measured_thickness
        self.measured_thickness_m = measured_thickness * 1e-3

        # Measured transfer function with safe division and phase calculation
        self.measured_H = np.divide(self.fft_sample, self.fft_reference + ERROR_EPSILON)
        self.measured_phase = np.angle(self.measured_H)

        # Placeholders and default plotting parameters
        self.current_n = None
        self.current_alpha = None
        self.noise_floor_db = -60.0
        self.use_backup_unwrap = False
        self.max_freq_before_noise = 3.0

        # Fabry–Perot placeholders
        self.fp_term_count = 0
        self.fp_n_avg = None                # averaged n
        self.fp_t_peak = None               # ana pulse zamanı (s)
        self.fp_delta_t = None              # Δt (s)
        self.fp_echo_times = np.array([])   # seconds

        self._setup_parameters(preset=preset, unwrap_range_selec=unwrap_range_selec, log_file=unwrap_file_root)

        utils.unwrap_phase(self)            # Unwrapping phase
        utils.phase_extrapolation(self)     # Curve fitting using 2 different method
        utils.recalculate_absorption(self)  # Initial calculation of n and alpha
        # utils.update_fp_from_n_avg(self)    # Number of the FP term calculation using initial n prediction

    def _load_signal(self, path: Path, col: int):
        """Reads csv/txt files."""
        ext = path.suffix.lower()
        try:
            if ext == ".csv":
                df = pd.read_csv(path, sep=None, engine='python', header=None, comment='#')

            elif ext == ".txt":
                raw_lines = path.read_text(encoding="utf-8", errors='replace').splitlines()
                numeric_pattern = re.compile(r'^\s*[+-]?\d')
                skiprows = next(
                    (i for i, line in enumerate(raw_lines)
                    if numeric_pattern.match(line)), None)
                if skiprows is None:
                    raise ValueError(f"No numeric data: {path}")

                df = pd.read_csv(
                    path,
                    sep=r"[,\s]+",
                    engine='python',
                    skiprows=skiprows,
                    header=None,
                    comment='#',
                    skip_blank_lines=True,)
                
            else:
                raise ValueError(f"Invalid file type: {path}")

            if col >= df.shape[1]:
                raise ValueError(f"Column {col} out of bounds.")

            time_data = df.iloc[:, 0].to_numpy(float)
            y_data    = df.iloc[:, col].to_numpy(float)

            return SignalData(time_data=time_data, y_data=y_data)

        except Exception as e:
            raise ValueError(f"Error loading {path}: {e}")

    def fft_analysis(self, 
                     data_obj,
                     zero_pad:              int | None = None, 
                     time_normalisation:    float = 1.0, 
                     logscale:              bool = False, 
                     positive:              bool = True,
                     plot:                  bool = False, 
                     t1:                    float | None = None, 
                     t2:                    float | None = None, 
                     window_type:           str | None = None, 
                     dc_offset:             float | None = None,):
        """
        Calculates the FFT of the signal.
        """
        if not hasattr(data_obj, 'y_data') or len(data_obj.y_data) == 0:
            raise ValueError("Invalid signal object: Data is missing or empty.")

        y_data = data_obj.y_data.copy()
        x_scaled = data_obj.time_data.copy() * time_normalisation

        if dc_offset is not None:
            y_data = self._remove_dc_offset_before_peak(x_scaled, y_data, ps_before=dc_offset)

        if t1 is not None and t2 is not None:
            indices = np.where((x_scaled >= t1) & (x_scaled <= t2))[0]
            if len(indices) == 0:
                raise ValueError("It couldnot find specied time interval in data...")
            else:
                x_scaled = x_scaled[indices]
                y_data = y_data[indices]

        if len(x_scaled) > 1:
            self.time_step = (x_scaled[-1] - x_scaled[0]) / (len(x_scaled) - 1)
        else:
            self.time_step = 1.0

        n = len(y_data)
        norm_factor = n

        # if window_type is not None:
            # window = get_window(window_type, n)
            # self.window_timestep = 0.156
            # y_data = y_data * self.window_timestep
            # norm_factor = sum(window)

        if zero_pad is not None: 
            if zero_pad > len(y_data):
                y_data = np.pad(y_data, (0, zero_pad - len(y_data)), 'constant')
            else:
                power = int(np.ceil(np.log2(len(y_data))))
                zero_pad = 2**power
                y_data = np.pad(y_data, (0, zero_pad - len(y_data)), 'constant')

        # n = len(y_data)
        # fft_signal = np.fft.fft(y_data) / norm_factor
        # fast_freqs = np.fft.fftfreq(n, self.time_step)
        n = len(y_data)
        fs = 1.0 / self.time_step  # Örnekleme Frekansı (Toplam Frekans)

        fft_signal = np.fft.fft(y_data) / norm_factor
        
        # --- DÜZELTME: fftfreq YERİNE linspace ---
        # fftfreq, son noktayı (fs) hariç tutar.
        # linspace ise son noktayı (fs) dahil eder ve aralığı ona göre sıkar.
        # fast_freqs = np.linspace(0, fs, n, endpoint=True)
        if positive:
            num_points = n // 2 
            fft_signal = fft_signal[:num_points]
            fft_signal[1:-1] *= 2 
            nyquist_limit = fs / 2  # Burası 3.2 THz
            fast_freqs = np.linspace(0, nyquist_limit, num_points, endpoint=True)
        else:
            fft_signal = np.fft.fft(y_data) / norm_factor
            nyquist_limit = fs / 2
            fast_freqs = np.linspace(0, fs, n, endpoint=True)
            
        magnitude_half = np.abs(fft_signal)
        freqs_half = fast_freqs 

        if logscale:
            magnitude_half = 20 * np.log10(magnitude_half + ERROR_EPSILON)

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(freqs_half, magnitude_half, linestyle='-', marker='o', markersize=2, linewidth=1)
            plt.ylabel("Magnitude (Log Scale)" if logscale else "Magnitude")
            
            title = f"FFT Spectrum - {self.sample_name or os.path.basename(self.signal_file)}"
            if t1 is not None and t2 is not None:
                title += f" (Window: {t1}-{t2} ps)"

            utils.configure_axis(plt.gca(), ylabel="Magnitude (Arbitrary Units) (Log Scale)" if logscale else "Magnitude (Arbitrary Units)", xlabel="Frequency (THz)", title=title)

            plt.tight_layout()
            plt.show()
            plt.close()

        return freqs_half, fft_signal

    def _remove_dc_offset_before_peak(self, x_arr, y_arr, ps_before=2.0):
        """Calculates DC offset based on the region before the main pulse."""
        peaks, _ = find_peaks(np.abs(y_arr), height=np.max(np.abs(y_arr)) * 0.5)
        if len(peaks) == 0: 
            return y_arr

        t_peak = x_arr[peaks[np.argmax(np.abs(y_arr[peaks]))]]
        t_cutoff = t_peak - ps_before * self.time_scale
        segment = y_arr[x_arr <= t_cutoff]

        # if just first 5 bin ignore it
        if len(segment) < 5: 
            return y_arr
        return y_arr - np.mean(segment)

    def _setup_parameters(self, preset: list = None, unwrap_range_selec: bool = False, log_file: str = None):
        """
        Configures frequency analysis limits and unwrap ranges.
        Prioritizes: Preset > User Input > Auto/Manual Selection.
        Decide analysis band (f_min, f_max) and unwrap-fit band for phase extrapolation.
        """
        # Use values from the files
        if preset and len(preset) == 5:
            f_min_thz, f_max_thz, self.number_fp_pulses, unwrap_fit_min_thz, unwrap_fit_max_thz = preset[:5]
        else:
            # If preset is missing or incomplete, ask user
            p = preset if preset else []

            def _get_val(idx, prompt, dtype):
                """If there is data and it is not None, otherwise request."""
                if idx < len(p) and p[idx] is not None:
                    return dtype(p[idx])
                val = input(prompt)
                return dtype(val) if val.strip() else None

            f_min_thz             = _get_val(0, "Min Analysis Freq (THz): ", float)
            f_max_thz             = _get_val(1, "Max Analysis Freq (THz): ", float)
            self.number_fp_pulses = _get_val(2, "Expected FP Pulses (int): ", int)
            unwrap_fit_min_thz    = _get_val(3, "Unwrap Fit Min (THz): ", float)
            unwrap_fit_max_thz    = _get_val(4, "Unwrap Fit Max (THz): ", float)

        # 2. Convert THz to Hz and Store
        self.f_min_extract, self.f_max_extract = f_min_thz * THZ_TO_HZ, f_max_thz * THZ_TO_HZ
        self.unwrap_fit_min, self.unwrap_fit_max = unwrap_fit_min_thz * THZ_TO_HZ, unwrap_fit_max_thz * THZ_TO_HZ
        utils.update_indices(self)

        # 3. Save unwrap limits to file (if applicable)
        if unwrap_range_selec:
            utils.update_unwrap_log(self, log_file)





















            # do not need to unwrap regions at this level since it is better to use GUI
            # # unwrap_fit_min_thz, unwrap_fit_max_thz = f_min_thz, f_max_thz
            # # Determine Unwrap Range for initial value
            # if unwrap_range_selec:
            #     unwrap_fit_min_thz, unwrap_fit_max_thz = self._resolve_unwrap_limits(log_file)

        # # 🔄 Last Verification
        # while True:
        #     confirm = input(f"✅ Final unwrap range: {unwrap_fit_min_thz:.4f} – {unwrap_fit_max_thz:.4f} THz. Confirm? [Y]/n: ").strip().lower()
        #     if confirm == "" or confirm == "y":
        #         break
        #     elif confirm == "n":
        #         print("🔁 Reopening plot for selection...")
        #         unwrap_fit_min_thz, unwrap_fit_max_thz = self._pick_unwrap()
        #     else:
        #         print("❓ Please enter Y or N.")


    # def _resolve_unwrap_limits(self, log_file: str):
    #     """Interactively decides unwrap limits: Loads from file or opens GUI picker."""
    #     saved_limits = self._get_saved_limits(log_file)
        
    #     if saved_limits:
    #         print(f"🔹 Found saved limits for {self.sample_name}: {saved_limits[0]} - {saved_limits[1]} THz")
    #         choice = input("Use these? [Y]/n/[E]dit: ").strip().lower()
            
    #         if choice == "e":
    #             try:
    #                 vals = input("Enter new limits (min max): ").split()
    #                 return float(vals[0]), float(vals[1])
    #             except ValueError:
    #                 print("⚠️ Invalid input. Opening graphical selection.")
    #         elif choice != "n": # Default is Yes
    #             return saved_limits

    #     # If no record exists or user chose 'n', open graph
    #     return self._pick_unwrap() 

    # def _get_saved_limits(self, log_file: str):
    #     """Reads the log file and returns limits for the current sample if they exist."""
    #     if not log_file or not os.path.exists(log_file):
    #         return None
        
    #     with open(log_file, "r") as f:
    #         for line in f:
    #             parts = line.strip().split()
    #             # Check if the line belongs to the current sample (self.name)
    #             if len(parts) >= 3 and parts[0] == self.sample_name:
    #                 return float(parts[1]), float(parts[2])
    #     return None     

    # def _pick_unwrap(self):
    #     """Interactive GUI to select frequency range (Drag mouse on Magnitude plot)."""
        
    #     selected = [None, None]
    #     phase = self.measured_phase
    #     freqs_thz = self.fft_freqs / 1e12
        
    #     print("\n👉 Drag mouse on the Bottom Plot to select range. Close window when done.")

    #     while True:
    #         fig, (ax_phase, ax_mag) = plt.subplots(2, 1, figsize=(12, 10), sharex=True, gridspec_kw=dict(hspace=0.3))

    #         # 1. Phase Plot
    #         ax_phase.plot(freqs_thz, phase, color='b', label=r'Measured $\angle$H')
    #         ax_phase.set_ylabel("Phase (rad)")
    #         ax_phase.set_title(f"Select Range: {self.sample_name}")
    #         ax_phase.grid(True, alpha=0.3)
    #         ax_phase.legend(loc='lower left', fontsize=10)

    #         # 2. Magnitude Plot
    #         # Not: Unified class'ta fft_reference -> fft_ref, fft_sample -> fft_sample
    #         ref_db = 20 * np.log10(np.abs(self.fft_reference) + 1e-20)
    #         sam_db = 20 * np.log10(np.abs(self.fft_sample) + 1e-20)

    #         ax_mag.plot(freqs_thz, ref_db, 'r', label='Reference FFT')
    #         ax_mag.plot(freqs_thz, sam_db, 'b', label='Sample FFT')
    #         ax_mag.set_ylabel("Magnitude (dB)")
    #         ax_mag.set_xlabel("Frequency (THz)")
    #         ax_mag.legend(loc='lower left')
    #         ax_mag.grid(True, alpha=0.3)

    #         # Callback Function
    #         def onselect(vmin, vmax):
    #             selected[0], selected[1] = min(vmin, vmax), max(vmin, vmax)

    #         # Selector (Sadece alt grafikte çalışır, ama sharex olduğu için hizalamak kolaydır)
    #         span = SpanSelector(ax_mag, onselect, 'horizontal', useblit=True,
    #                             props=dict(alpha=0.3, facecolor='green'))
    #         self._span = span
    #         plt.show(block=True) # Wait for user to close window

    #         # 3. Validation & Correction
    #         if selected[0] is None:
    #             if input("⚠️ No selection made. Retry? [Y/n]: ").lower() == 'n':
    #                 raise RuntimeError("Selection cancelled by user.")
    #         else:
    #             f_min, f_max = selected
    #             print(f"🔹 Selected: {f_min:.4f} - {f_max:.4f} THz")
                
    #             edit = input("Press ENTER to accept or type 'min max' to override: ").strip()
    #             if not edit:
    #                 return f_min, f_max
                
    #             try:
    #                 # Manual values
    #                 vals = list(map(float, edit.split()))
    #                 return vals[0], vals[1]
    #             except ValueError:
    #                 print("⚠️ Invalid input. Using graphical selection.")
    #                 return f_min, f_max


# def _estimate_fp_echo_count(self, eps_guess: float | None = None, cfg=None,):
#     """
#     Estimates number of Fabry-Perot echoes and interactively confirms or adjusts result.
    
#     Two modes:
#         1) eps_guess is given (old behaviour):
#         - Uses eps_guess and thickness_mm to compute Δt_theoretical
#         - Optionally validates with measured peak periodicity (cfg.auto_validate_periodicity)
#         - Does NOT change eps_guess

#         2) eps_guess is not given:
#         - Tries to estimate Δt from the *measured* peak spacings
#         - From Δt and thickness_mm, estimates eps_calc
#         - Stores eps_calc in self.fp_calc_eps
#         - If periodicity is not reliable, falls back to "no FP" (echo_count = 0)
#     """
#     thickness_mm = self.measured_thickness_mm
#     thickness_m = thickness_mm * 1e-3

#     if thickness_mm is None:
#         raise ValueError("thickness_mm must be provided for FP echo estimation.")

#     cfg = cfg or {}
#     min_eps                   = cfg.minimum_eps or 1.0
#     auto_validate             = cfg.auto_validate_periodicity or False
#     tol                       = cfg.periodicity_tolerance or 0.05
#     min_peaks_for_periodicity = cfg.min_peaks_for_periodicity or 2
#     interactive               = cfg.interactive_selection or False

#     dt_min_phys = (2.0 * thickness_m * np.sqrt(min_eps)) / c

#     x_raw = np.asarray(self.time_data, dtype=float)
#     y = np.asarray(self.sample.y_data, dtype=float)

#     if x_raw.size != y.size or x_raw.size == 0:
#         raise RuntimeError("x_data and y_data must be non-empty and have the same length.")

#     # Find peaks: it shold be at least one (orginal signal)
#     peaks, _ = find_peaks(y)
#     if len(peaks) == 0:
#         raise RuntimeError("No peaks found in signal; cannot estimate FP echo count.")

#     # Find main pulse then calculate remaining time
#     main_idx = peaks[np.argmax(y[peaks])]

#     t_end_raw = x_raw[-1]
#     t_peak_raw = x_raw[main_idx]                    # original units (e.g. ps)

#     if t_end_raw <= t_peak_raw:
#         raise RuntimeError("Signal length after main peak is zero or negative.")

#     temporal_length = t_end_raw - t_peak_raw        # seconds

#     # ---------- Measured peak spacings Δt_meas ----------
#     dt_med = None
#     dt_rel_spread = None
#     peaks_after = peaks[peaks > main_idx]
#     if len(peaks_after) >= min_peaks_for_periodicity:
#         t_peaks_sec = x_raw[peaks_after]
#         dt = np.diff(t_peaks_sec)

#         if dt.size >= 2:
#             dt_med = float(np.median(dt))           # robust estimate of Δt_meas
#             if dt_med > 0:
#                 dt_rel_spread = float(np.std(dt) / dt_med)
#             else:
#                 dt_rel_spread = None

#     # ---------- Mode 1: eps_guess is None → estimate eps from Δt_meas ----------
#     eps_from_dt = None
#     delta_t_used = None

#     if eps_guess is None:
#         if dt_med is not None and dt_rel_spread is not None and dt_rel_spread <= tol:                # Use measured period to estimate eps
#             if (0.5 * dt_min_phys) <= dt_med:
#                 eps_raw = (c * dt_med / (2.0 * thickness_m)) ** 2
#                 eps_clamped = float(np.clip(eps_raw, min_eps))
#                 eps_from_dt = eps_clamped

#                 delta_t_used = (2.0 * thickness_m * np.sqrt(eps_clamped)) / c
#         else:
#             # Cannot reliably detect periodic FP; assume no FP contribution
#             eps_from_dt = None
#             delta_t_used = None

#     else:
#         # Cannot reliably detect periodic FP
#         eps_from_dt = None
#         delta_t_used = None
    
#     # ---------- Mode 2: eps_guess is provided (old behaviour + optional validation) ----------
#     if eps_guess is not None:
#         eps_eff = float(max(eps_guess, min_eps))            
#         delta_t_th = (2.0 * thickness_m * np.sqrt(eps_eff)) / c  # seconds
        
#         if delta_t_th <= 0:
#                 delta_t_used = None 
#         else:
#             delta_t_used = delta_t_th
    
    
#         # Optional validation: if measured Δt matches theory within tolerance
#         if auto_validate and dt_med is not None and delta_t_used is not None:
#             rel_err = abs(dt_med - delta_t_th) / delta_t_th
#             if rel_err <= tol:
#                 # Only accept dt_med if it is also physically reasonable
#                 if (0.8 * dt_min_phys) <= dt_med:
#                     delta_t_used = dt_med

#     # Safe Minimum Δt check (using physical limit)
#     # If estimated Δt is way smaller than theoretical minimum, reject it.
#     SAFE_MIN_DT = 0.5 * dt_min_phys

#     # If still no usable Δt (eps_guess None & no periodic pattern), treat as "no FP"
#     if delta_t_used is None or delta_t_used <= SAFE_MIN_DT:
#         self.fp_echo_count = 0
#         self.fp_term_count = 0
#         self.fp_calc_eps = eps_from_dt
#         self.fp_echo_times = np.array([])
#         return

#     # ---------- Final echo count from chosen Δt ----------
#     raw_count = temporal_length / delta_t_used
#     echo_count = int(np.floor(raw_count))
#     self.fp_term_count = echo_count

#     # Store eps estimate
#     if eps_guess is None and eps_from_dt is not None:
#         # Fully automatic mode: store the estimated permittivity
#         self.fp_calc_eps = eps_from_dt
#     elif eps_guess is not None:
#         # If you want, you can also keep track of the eps used
#         self.fp_calc_eps = float(eps_guess)
#     else:
#         self.fp_calc_eps = None

#     if interactive:
#         while True:
#             # _plot_fp_lines expects t_peak in ORIGINAL units (e.g. ps),
#             self._plot_fp_lines(t_peak_raw, delta_t_used, echo_count)

#             print("\n--- Fabry–Perot Echo Estimation ---")
#             print(f"Initial echo count: {echo_count}")
#             print(f"Δt_used           : {delta_t_used * 1e12:.3f} ps")
#             print(f"thickness         : {thickness_mm:.3f} mm")

#             if eps_guess is None:
#                 if eps_from_dt is not None:
#                     print(f"eps_est (from Δt): {eps_from_dt:.4f}")
#                 else:
#                     print("eps_est (from Δt): could not be reliably estimated (no clear periodicity).")
#             else:
#                 print(f"eps_guess (input): {float(eps_guess):.4f}")

#             resp = input("Accept this value? [ENTER/y] or type new integer: ").strip().lower()

#             if resp in ("", "y"):
#                 break

#             try:
#                 new_val = int(resp)
#                 echo_count = max(0, new_val)
#                 self.fp_term_count = echo_count
#             except ValueError:
#                 print("Invalid input. Please press ENTER, 'y' or type an integer.")

#     self.fp_term_count = echo_count
#     if echo_count > 0:
#         dt_raw = delta_t_used  # self.time_data zaten saniye
#         echo_indices = np.arange(1, echo_count + 1)
#         self.fp_echo_times = t_peak_raw + (echo_indices * dt_raw)
#     else:
#         self.fp_echo_times = np.array([])

# def _plot_fp_lines(self, t_peak: float, delta_t: float, echo_count: int,):
#     """Plot the Fabry-Perot echo positions on the time-domain signal."""

#     x = np.array(self.time_data)
#     y = np.array(self.sample.y_data)
#     plt.figure(figsize=(10, 4))
#     plt.plot(x, y, label='Signal')
#     plt.axvline(t_peak, color='red', linestyle='--', label='First Peak')
    
#     for i in range(1, echo_count+1):
#         t_line = t_peak + i * delta_t 
#         plt.axvline(t_line, color='green', linestyle=':', label=f'Delta t x {i}' if i==1 else None)

#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.title("FP Echo Estimation")
#     plt.legend()
#     plt.tight_layout()

# def _estimate_fp_echo_count(self, eps_guess: float | None = None, cfg=None,):
#     """
#     Estimates number of Fabry-Perot echoes and interactively confirms or adjusts result.
    
#     Two modes:
#         1) eps_guess is given (old behaviour):
#         - Uses eps_guess and thickness_mm to compute Δt_theoretical
#         - Optionally validates with measured peak periodicity (cfg.auto_validate_periodicity)
#         - Does NOT change eps_guess

#         2) eps_guess is not given:
#         - Tries to estimate Δt from the *measured* peak spacings
#         - From Δt and thickness_mm, estimates eps_calc
#         - Stores eps_calc in self.fp_calc_eps
#         - If periodicity is not reliable, falls back to "no FP" (echo_count = 0)
#     """
#     thickness_mm = self.measured_thickness_mm
#     thickness_m = thickness_mm * 1e-3

#     if thickness_mm is None:
#         raise ValueError("thickness_mm must be provided for FP echo estimation.")

#     cfg = cfg or {}
#     min_eps                   = cfg.minimum_eps or 1.0
#     auto_validate             = cfg.auto_validate_periodicity or False
#     tol                       = cfg.periodicity_tolerance or 0.05
#     min_peaks_for_periodicity = cfg.min_peaks_for_periodicity or 2
#     interactive               = cfg.interactive_selection or False

#     dt_min_phys = (2.0 * thickness_m * np.sqrt(min_eps)) / c

#     x_raw = np.asarray(self.time_data, dtype=float)
#     y = np.asarray(self.sample.y_data, dtype=float)

#     if x_raw.size != y.size or x_raw.size == 0:
#         raise RuntimeError("x_data and y_data must be non-empty and have the same length.")

#     # Find peaks: it shold be at least one (orginal signal)
#     peaks, _ = find_peaks(y)
#     if len(peaks) == 0:
#         raise RuntimeError("No peaks found in signal; cannot estimate FP echo count.")

#     # Find main pulse then calculate remaining time
#     main_idx = peaks[np.argmax(y[peaks])]

#     t_end_raw = x_raw[-1]
#     t_peak_raw = x_raw[main_idx]                    # original units (e.g. ps)

#     if t_end_raw <= t_peak_raw:
#         raise RuntimeError("Signal length after main peak is zero or negative.")

#     temporal_length = t_end_raw - t_peak_raw        # seconds

#     # ---------- Measured peak spacings Δt_meas ----------
#     dt_med = None
#     dt_rel_spread = None
#     peaks_after = peaks[peaks > main_idx]
#     if len(peaks_after) >= min_peaks_for_periodicity:
#         t_peaks_sec = x_raw[peaks_after]
#         dt = np.diff(t_peaks_sec)

#         if dt.size >= 2:
#             dt_med = float(np.median(dt))           # robust estimate of Δt_meas
#             if dt_med > 0:
#                 dt_rel_spread = float(np.std(dt) / dt_med)
#             else:
#                 dt_rel_spread = None

#     # ---------- Mode 1: eps_guess is None → estimate eps from Δt_meas ----------
#     eps_from_dt = None
#     delta_t_used = None

#     if eps_guess is None:
#         if dt_med is not None and dt_rel_spread is not None and dt_rel_spread <= tol:                # Use measured period to estimate eps
#             if (0.5 * dt_min_phys) <= dt_med:
#                 eps_raw = (c * dt_med / (2.0 * thickness_m)) ** 2
#                 eps_clamped = float(np.clip(eps_raw, min_eps))
#                 eps_from_dt = eps_clamped

#                 delta_t_used = (2.0 * thickness_m * np.sqrt(eps_clamped)) / c
#         else:
#             # Cannot reliably detect periodic FP; assume no FP contribution
#             eps_from_dt = None
#             delta_t_used = None

#     else:
#         # Cannot reliably detect periodic FP
#         eps_from_dt = None
#         delta_t_used = None
    
#     # ---------- Mode 2: eps_guess is provided (old behaviour + optional validation) ----------
#     if eps_guess is not None:
#         eps_eff = float(max(eps_guess, min_eps))            
#         delta_t_th = (2.0 * thickness_m * np.sqrt(eps_eff)) / c  # seconds
        
#         if delta_t_th <= 0:
#                 delta_t_used = None 
#         else:
#             delta_t_used = delta_t_th
    
    
#         # Optional validation: if measured Δt matches theory within tolerance
#         if auto_validate and dt_med is not None and delta_t_used is not None:
#             rel_err = abs(dt_med - delta_t_th) / delta_t_th
#             if rel_err <= tol:
#                 # Only accept dt_med if it is also physically reasonable
#                 if (0.8 * dt_min_phys) <= dt_med:
#                     delta_t_used = dt_med

#     # Safe Minimum Δt check (using physical limit)
#     # If estimated Δt is way smaller than theoretical minimum, reject it.
#     SAFE_MIN_DT = 0.5 * dt_min_phys

#     # If still no usable Δt (eps_guess None & no periodic pattern), treat as "no FP"
#     if delta_t_used is None or delta_t_used <= SAFE_MIN_DT:
#         self.fp_echo_count = 0
#         self.fp_term_count = 0
#         self.fp_calc_eps = eps_from_dt
#         self.fp_echo_times = np.array([])
#         return

#     # ---------- Final echo count from chosen Δt ----------
#     raw_count = temporal_length / delta_t_used
#     echo_count = int(np.floor(raw_count))
#     self.fp_term_count = echo_count

#     # Store eps estimate
#     if eps_guess is None and eps_from_dt is not None:
#         # Fully automatic mode: store the estimated permittivity
#         self.fp_calc_eps = eps_from_dt
#     elif eps_guess is not None:
#         # If you want, you can also keep track of the eps used
#         self.fp_calc_eps = float(eps_guess)
#     else:
#         self.fp_calc_eps = None

#     if interactive:
#         while True:
#             # _plot_fp_lines expects t_peak in ORIGINAL units (e.g. ps),
#             self._plot_fp_lines(t_peak_raw, delta_t_used, echo_count)

#             print("\n--- Fabry–Perot Echo Estimation ---")
#             print(f"Initial echo count: {echo_count}")
#             print(f"Δt_used           : {delta_t_used * 1e12:.3f} ps")
#             print(f"thickness         : {thickness_mm:.3f} mm")

#             if eps_guess is None:
#                 if eps_from_dt is not None:
#                     print(f"eps_est (from Δt): {eps_from_dt:.4f}")
#                 else:
#                     print("eps_est (from Δt): could not be reliably estimated (no clear periodicity).")
#             else:
#                 print(f"eps_guess (input): {float(eps_guess):.4f}")

#             resp = input("Accept this value? [ENTER/y] or type new integer: ").strip().lower()

#             if resp in ("", "y"):
#                 break

#             try:
#                 new_val = int(resp)
#                 echo_count = max(0, new_val)
#                 self.fp_term_count = echo_count
#             except ValueError:
#                 print("Invalid input. Please press ENTER, 'y' or type an integer.")

#     self.fp_term_count = echo_count
#     if echo_count > 0:
#         dt_raw = delta_t_used  # self.time_data zaten saniye
#         echo_indices = np.arange(1, echo_count + 1)
#         self.fp_echo_times = t_peak_raw + (echo_indices * dt_raw)
#     else:
#         self.fp_echo_times = np.array([])

# def _plot_fp_lines(self, t_peak: float, delta_t: float, echo_count: int,):
#     """Plot the Fabry-Perot echo positions on the time-domain signal."""

#     x = np.array(self.time_data)
#     y = np.array(self.sample.y_data)
#     plt.figure(figsize=(10, 4))
#     plt.plot(x, y, label='Signal')
#     plt.axvline(t_peak, color='red', linestyle='--', label='First Peak')
    
#     for i in range(1, echo_count+1):
#         t_line = t_peak + i * delta_t 
#         plt.axvline(t_line, color='green', linestyle=':', label=f'Delta t x {i}' if i==1 else None)

#     plt.xlabel("Time")
#     plt.ylabel("Amplitude")
#     plt.title("FP Echo Estimation")
#     plt.legend()
#     plt.tight_layout()
