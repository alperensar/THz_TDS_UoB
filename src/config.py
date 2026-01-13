# config.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Config system for TDS_FFT processing pipeline.
# Parses YAML, expands environment variables, applies CLI overrides, and validates using Pydantic models.
#
# Structure matches the TDS_FFT.yaml format exactly.
#
# author@: Alperen Sari - PhDing at the University of Birmingham
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
from __future__ import annotations

import os, yaml, ast
from pathlib import Path
from typing import Any, Optional, Dict, List, Union, Tuple

from pydantic import BaseModel, Field, ValidationError, field_validator, ConfigDict

# --- YAML Loading & Override Logic (Unchanged from your code) ---
def parse_overrides(argv: list[str]) -> dict[str, object]:
    """Parses key=value overrides from a list of strings (like sys.argv)."""
    out: dict[str, object] = {}
    for s in argv or []:
        if "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        v = os.path.expanduser(os.path.expandvars(v)).strip() # Expand env vars in override value too
        if v == "": continue

        low = v.lower()
        if low in ("none", "null", "~"):
            out[k] = None
        # EMPTY_TOKEN logic removed for simplicity, can be added back if needed
        else:
            out[k] = v # Keep as string for now, Pydantic/literal_eval will parse later
    return out

def _expand_env(obj: Any) -> Any:
    """Recursively expands $VAR or ${VAR} in strings within nested dicts/lists."""
    if isinstance(obj, dict):
        return {k: _expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env(x) for x in obj]
    if isinstance(obj, str):
        try:
            expanded = os.path.expandvars(obj)
            if expanded == obj and '${' in obj and '}' in obj:
                 import re
                 pattern = r'\$\{([A-Za-z_][A-Za-z0-9_]*)\}'
                 def replace_env(match):
                     var_name = match.group(1)
                     return os.environ.get(var_name, match.group(0)) # Keep original if var not found
                 expanded = re.sub(pattern, replace_env, obj)

            return expanded
        except Exception:
             return obj # Keep original if expansion fails
    return obj

def _apply_overrides(cfg: dict, overrides: dict[str, Any]):
    """Recursively applies dot-notation overrides (e.g., 'loading.limit_files=10')."""
    for k, v in overrides.items():
        d = cfg
        keys = k.split(".")
        try:
            for kk in keys[:-1]:
                if not isinstance(d.get(kk), dict):
                    d[kk] = {}
                d = d[kk]

            final_key = keys[-1]
            if isinstance(v, str):
                vv = v.strip()
                parsed_value = None
                try:
                    parsed_value = ast.literal_eval(vv)
                except (ValueError, SyntaxError): # Handle non-literal strings
                    low = vv.lower()
                    if low == "true": parsed_value = True
                    elif low == "false": parsed_value = False
                    else:
                        try: # Attempt float/int conversion
                            if any(c in vv for c in ('.', 'e', 'E')):
                                parsed_value = float(vv)
                            else:
                                parsed_value = int(vv)
                        except ValueError:
                            parsed_value = vv # Keep as string if all else fails
                v = parsed_value
            
            d[final_key] = v
        except KeyError:
            print(f"Warning: Override key '{k}' path not fully found in config.")
        except Exception as e:
            print(f"Warning: Failed to apply override '{k}={v}': {e}")
    return cfg

def load_config(path: str | Path, cli_overrides_list: list[str] = None) -> dict[str, Any]:
    """Load YAML, expand env vars, apply CLI overrides."""
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    # Step 1: Load base config
    with open(path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    # Step 2: Expand environment variables
    cfg = _expand_env(cfg)

    # Step 3: Parse and apply CLI overrides
    cli_overrides_dict = parse_overrides(cli_overrides_list or [])
    if cli_overrides_dict:
        cfg = _apply_overrides(cfg, cli_overrides_dict)

    return cfg

class OptimizationBounds(BaseModel):
    n_bound:               float = 0.05
    k_bound:               float = 0.60 

class MaterialParams(BaseModel):
    eps:                    Optional[float] = None
    measured_thickness_mm:  Optional[float] = None
    sigma_um:               Optional[float] = None
    fp_terms:               Optional[int] = None

class PathsConfig(BaseModel):
    """Configuration for data sources."""
    model_config = ConfigDict(extra="forbid")

    root_dir:               Path = Field(..., description="Root directory containing pulse folders")
    pulse_dir_names:        List[str] = Field(default_factory=lambda: ["pulse"])
    subdir_types:           List[str] = Field(default_factory=lambda: ["film", "plate"])
    reference_filename:     str = "reference.txt"
    thickness_file:         str = "thickness.txt"
    base_FFT_folder:        str = "FFT_data"
    pickle_folder:          str = "pulsefft"
    processed_dir_names:    str = "Processed_Revision"
    unwrap_log:             str = "unwrap_limits.txt"

    @field_validator("root_dir", mode="before")
    def _resolve_root(cls, v):
        return Path(v).expanduser()

class FileHandlingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_column:          int = 1
    reference_column:       int = 2
    allow_self_reference:   bool = True
    extensions:             List[str] = ["txt", "csv"]
    time_unit:              str = "ps"
    required_cols:          Optional[List[str]]

class ParamEstimationConfig(BaseModel):
    step1_algorithm:        str = Field("Optimise", description="Algorithm for Thickness Sweep")
    step2_algorithm:        str = Field("Optimise", description="Algorithm for Final Extraction")
    backend:                str = "serial"
    scanning_resolution:    int = 10
    scanning_range_mm:      float = 0.05
    calculation_cores:      Union[int, str] = 4
    processing_modes:       List[Tuple[bool, bool]] 
    pkl_prefix:             Optional[str] = None
    optimization_bounds:    OptimizationBounds = Field(default_factory=OptimizationBounds)

    # material_overrides:     Optional[Dict[str, MaterialParams]] = None

class FFTConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    zero_pad:               int = 512
    positive_only:          bool = True
    logscale:               bool = False
    dc_offset_ps:           float = 2.0

class UnwrapConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    use_gui:                bool = True
    auto_save:              bool = True
    auto_load_previous:     bool = True

class SavingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_mat:               bool = True
    save_txt:               bool = True
    save_pickle:            bool = True
    complex_mode:           str = "both"   # real_imag | mag_phase | both
    no_save_mode:           bool = False   # disable all writes

class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verbose:                bool = True
    tqdm_width:             int = 100

class InstrumentProfilesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    # free-form: content varies per instrument

class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    process_only_index:     Optional[int] = None
    start_from_index:       Optional[int] = None
    save_outputs:           bool = True
    save_unwrap_log:        bool = True
    continue_on_error:      bool = True

class MaterialOverridesConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    # free-form: material: { eps: .., measured_thickness_mm: .. }

class FilterConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    processed_dir_name:     str = "Processed_Revision"
    nr_dir_name:            str = "NR_FP"
    r_dir_name:             str = "R_FP"
    output_dir_name:        str = "ProcessedFilter_Revision"
    log_name:               str = "TDSFilter.log"

    window_default:         int = 3
    trial_iterations:       List[int] = Field(default_factory=lambda: [5, 2])

    n_col:                  int =2
    k_col:                  int =3
    verbose:                bool = True

    window_n_default:       int =3
    iter_n_default:         int =5
    window_k_default:       int =3
    iter_k_default:         int =5

    # GUI slider limits
    window_min:             int =0
    window_max:             int =51
    iter_min:               int =0
    iter_max:               int =20

    freq_col_name:          str ="freq_Hz"

    fmin_thz_default:       float =0.2
    fmax_thz_default:       float =5

    # --- Validators ---
    @field_validator("window_default")
    @classmethod
    def _window_must_be_odd_and_ge_3(cls, v: int) -> int:
        if v < 3 or (v % 2 == 0):
            raise ValueError("window_default must be an odd integer >= 3")
        return v

    @field_validator("trial_iterations")
    @classmethod
    def _iters_must_be_positive(cls, v: List[int]) -> List[int]:
        if not v or any(int(x) < 1 for x in v):
            raise ValueError("trial_iterations must be a non-empty list of ints >= 1")
        return [int(x) for x in v]

class MergeConfig(BaseModel):
    model_config = ConfigDict(extra="allow")

class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    
class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    paths:                  PathsConfig
    file_handling:          FileHandlingConfig
    fft:                    FFTConfig
    unwrap:                 UnwrapConfig
    saving:                 SavingConfig
    logging:                LoggingConfig
    instrument_profiles:    InstrumentProfilesConfig = InstrumentProfilesConfig()
    run:                    RunConfig
    material_overrides:     MaterialOverridesConfig = MaterialOverridesConfig()
    param_estimation:       ParamEstimationConfig = Field(default_factory=ParamEstimationConfig)
    filter:                 FilterConfig
    plot:                   PlotConfig
    merge:                  MergeConfig

def load_and_validate_config(cfg_path: str | Path, cli_args: List[str] = None) -> AppConfig:
    raw = load_config(cfg_path, cli_args)
    try:
        return AppConfig.model_validate(raw)
    except ValidationError as e:
        print("ERROR: Configuration validation failed:")
        print(e)
        raise



# class FPEstimationConfig(BaseModel):
#     model_config = ConfigDict(extra="forbid")

#     enabled:                    bool = True
#     minimum_eps:                float = 1.0
#     auto_validate_periodicity:  bool = True
#     periodicity_tolerance:      float = 0.05
#     allow_override_from_yaml:   bool = True
#     interactive_selection:      bool = True
#     min_peaks_for_periodicity:  int=2