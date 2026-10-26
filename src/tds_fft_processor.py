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
GRID_REL_TOLERANCE = 1e-3     # allowed time-grid deviation as a fraction of dt (file rounding)


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
                 fp_terms_override:     int | None = None,
                 cfg,):
        """
        sample_file :                Path to sample waveform (.txt or .csv)
        reference_file :             Path to reference waveform
        sample_column :              Index of sample amplitude column (0=time, 1,2,... amplitude)
        reference_column :           Index of reference amplitude column (same as above)
        time_unit : {"ps","ns","s"}  Unit of input time axis
        fp_terms_override :          Fixed FP order (nonnegative int); None uses the estimate from n_avg
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
        self._validate_signal_pair(self.sample, self.reference)

        # time normalisation
        self.time_unit = time_unit
        self.time_scale = {"ps": 1e-12, "ns": 1e-9, "s": 1.0}[time_unit]
        self.time_data = self.sample.time_data * self.time_scale           

        # FFT calculation
        zero_pad = cfg.fft.zero_pad or 1.0
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
        if fp_terms_override is not None and (isinstance(fp_terms_override, (bool, np.bool_)) or
                not isinstance(fp_terms_override, (int, np.integer)) or fp_terms_override < 0):
            raise ValueError("fp_terms_override must be a nonnegative integer or None.")
        self.fp_terms_override = fp_terms_override
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

    @staticmethod
    def _validate_signal_grid(data, label="Signal"):
        """A standard FFT requires finite values on a uniform increasing grid."""
        time = np.asarray(data.time_data, dtype=float)
        field = np.asarray(data.y_data)
        if time.ndim != 1 or field.ndim != 1 or len(time) != len(field) or len(time) < 2:
            raise ValueError(f"{label}: time and field must be matching 1D arrays with at least two samples.")
        if not np.all(np.isfinite(time)) or not np.all(np.isfinite(field)):
            raise ValueError(f"{label}: time and field must be finite.")
        dt = (time[-1]-time[0])/(len(time)-1)
        # Instruments write time with limited decimals (e.g. 0.15625 ps stored as 0.1562/0.1563),
        # so each point is compared with the ideal grid t0 + k*dt instead of requiring equal steps.
        tolerance = max(abs(dt)*GRID_REL_TOLERANCE, 16*np.spacing(np.max(abs(time))))
        deviation = np.max(np.abs(time - (time[0] + dt*np.arange(len(time)))))
        if dt <= 0 or np.any(np.diff(time) <= 0) or deviation > tolerance:
            raise ValueError(f"{label}: FFT requires a uniform increasing time grid; resample explicitly first.")
        return tolerance

    @classmethod
    def _validate_signal_pair(cls, sample, reference):
        tolerance = max(cls._validate_signal_grid(sample, "Sample"),
                        cls._validate_signal_grid(reference, "Reference"))
        if len(sample.time_data) != len(reference.time_data) or not np.allclose(
                sample.time_data, reference.time_data, rtol=0, atol=tolerance):
            raise ValueError("Sample and reference must use the same physical time grid, including time origin and length; align explicitly before preprocessing.")

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
        self._validate_signal_grid(data_obj)

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
        if zero_pad is not None: 
            if zero_pad > len(y_data):
                y_data = np.pad(y_data, (0, zero_pad - len(y_data)), 'constant')
            else:
                power = int(np.ceil(np.log2(len(y_data))))
                zero_pad = 2**power
                y_data = np.pad(y_data, (0, zero_pad - len(y_data)), 'constant')
        n = len(y_data)
        fft_signal = np.fft.fft(y_data) / norm_factor
        # Use the frequencies of the actual DFT bins. An endpoint-inclusive
        # linspace incorrectly labels the last retained positive bin as Nyquist.
        fast_freqs = np.fft.fftfreq(n, d=self.time_step)
        if positive:
            # Retain DC and all strictly positive bins; even-length Nyquist
            # remains excluded, matching the existing positive-spectrum layout.
            num_points = (n + 1) // 2
            fft_signal = fft_signal[:num_points]
            fft_signal[1:] *= 2
            fast_freqs = fast_freqs[:num_points]
            
        magnitude_half = np.abs(fft_signal)
        freqs_half = fast_freqs 

        if logscale:
            magnitude_half = 20 * np.log10(magnitude_half + ERROR_EPSILON)

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(freqs_half / THZ_TO_HZ, magnitude_half, linestyle='-', marker='o', markersize=2, linewidth=1)
            plt.ylabel("Magnitude (Log Scale)" if logscale else "Magnitude")
            
            title = f"FFT Spectrum - {getattr(self, 'sample_name', None) or 'Signal'}"
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
