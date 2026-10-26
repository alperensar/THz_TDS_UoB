# THz-TDS Dielectric Parameter Extraction Algorithm

**Version: 0.2.0 — RoE revision with the corrected surface-scattering model.**

> [!CAUTION]
> **Updated surface-scattering model.** This version corrects excess attenuation and reference-plane errors in the earlier formulation, with validation against CST simulations. For the current model derivation, see **Publication 1 (RoE, under review)** in [Citations & References](#-citations--references). Details are given in [Disclosure & Historical Note](#-disclosure--historical-note).

This software is an in-house developed algorithm at the University of Birmingham designed to extract complex dielectric parameters from **Terahertz Time-Domain Spectroscopy (THz-TDS)** measurements. It features a dual-stage workflow, advanced optimization routines, and a unique surface roughness-induced scattering compensation model based on Geometric Optics.

## 🚀 Key Features

- **Two-Stage Extraction:** Conversion from measured time-domain signals to frequency-domain parameter extraction.
- **Optimization:** Nelder-Mead simplex and least-squares optimization routines.
- **Surface Roughness Modeling:** Accounts for surface-scattering attenuation using a GO-based model, subject to its validity conditions.
- **Parallel Processing:** Serial and multiprocessing execution for parameter extraction.
- **Interactive Workflow:** GUI controls for preprocessing parameters and visual inspection before saving results.

## 📦 Installation & Setup

Use **Python 3.12** for the committed dependency lock. Run the following commands from the repository directory.

```bash
# Create and activate a virtual environment (macOS/Linux).
python3.12 -m venv .venv
source .venv/bin/activate

# Install the locked runtime dependencies.
python -m pip install --upgrade pip
python -m pip install "poetry==2.2.1"
poetry install --only main --no-root

# Install the Excel writer required by result exports.
python -m pip install "openpyxl>=3.1,<4"
```

On Windows PowerShell, create the environment with `py -3.12 -m venv .venv` and activate it with `.\.venv\Scripts\Activate.ps1`; then run the same installation commands starting at `python -m pip install --upgrade pip`.

`poetry.lock` pins the main runtime dependencies. The Excel writer is installed separately because it is not yet included in the lock. GUI workflows also require a working **Tkinter/Tcl-Tk** installation and a graphical desktop.

Set `paths.root_dir` in [config.yaml](config.yaml) to your dataset directory, or pass `--root-dir` when running either CLI. Other configuration settings control input columns, time units, frequency ranges, extraction modes and output folders.

## ⚙️ Data Structure Requirements

### 1) Directory Tree

With the default configuration, arrange measurements as follows:

```text
Root_Directory/
├── thickness.txt               # Sample metadata
└── Sample_Name/                # Must match a metadata 'material' entry
    └── Measurement_System/     # For example, Toptica_UoB
        └── pulse/              # Configurable through paths.pulse_dir_names
            ├── sample_data.txt # Time-domain sample/reference columns
            └── reference.txt   # Optional separate reference measurement
```

The signal columns and time unit are set by `file_handling.sample_column`, `file_handling.reference_column` and `file_handling.time_unit`. See [folder_struct](folder_struct) for example inputs.

### 2) Metadata File (`thickness.txt` or `thickness_info.txt`)

Create a **tab-separated** metadata file in the root directory. The default filename is `thickness.txt`; to use `thickness_info.txt`, set `paths.thickness_file` to that name in the configuration.

The default header must match `file_handling.required_cols`:

```text
material	thickness_mm	sigma_um	eps
```

Example with initial permittivity estimates:

```text
material	thickness_mm	sigma_um	eps
ROGERS_TMM3	3.1516	0.382325	3.27
ROGERS_RT6002	3.0707	0.613363	2.94
ROGERS_RT5880LZ	3.1334	0.586083	2.00
```

These permittivity values illustrate the file format; supply estimates appropriate to your samples, or leave those cells blank.

- `material`: sample folder name.
- `thickness_mm`: measured sample thickness in millimetres.
- `sigma_um`: RMS surface height in micrometres. Set it to **0** to disable the surface-roughness contribution.
- `eps`: optional initial estimate of real relative permittivity. **It may be left blank**, while retaining the column header; the preprocessing default is then `1.0`, unless overridden in the configuration. If your file uses `epsR`, change the fourth entry of `file_handling.required_cols` from `eps` to `epsR`.

See [folder_struct/thickness.txt](folder_struct/thickness.txt) for an example with blank permittivity cells.

## 🛠 Execution Modes

Run commands from the repository root with the virtual environment activated. Replace `/path/to/Root_Directory` with your dataset path.

**1. Preprocessing and FFT:**

```bash
python src/tds_preprocessing_CLI.py --config config.yaml --root-dir /path/to/Root_Directory
```

**2. Parameter extraction on one CPU core:**

```bash
python src/tds_extraction_CLI.py --config config.yaml --root-dir /path/to/Root_Directory --backend serial
```

**3. Parameter extraction with multiprocessing:**

```bash
python src/tds_extraction_CLI.py --config config.yaml --root-dir /path/to/Root_Directory --backend multiprocessing
```

Choose either extraction backend after preprocessing. Omit `--root-dir` to use the configured `paths.root_dir`. Use `--help` with either script to inspect its options.

### Optional VS Code Launch Configuration

With the Python debugger extension installed, add this configuration to `.vscode/launch.json`. It uses the root directory from `config.yaml`; select the `.venv` interpreter in VS Code.

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "THz TDS Processing",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/tds_preprocessing_CLI.py",
      "cwd": "${workspaceFolder}",
      "args": ["--config", "config.yaml"]
    },
    {
      "name": "THz TDS Extraction (Serial)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/tds_extraction_CLI.py",
      "cwd": "${workspaceFolder}",
      "args": ["--config", "config.yaml", "--backend", "serial"]
    },
    {
      "name": "THz TDS Extraction (Multiprocessing)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/tds_extraction_CLI.py",
      "cwd": "${workspaceFolder}",
      "args": ["--config", "config.yaml", "--backend", "multiprocessing"]
    }
  ]
}
```

## 📜 Disclosure & Historical Note

> [!CAUTION]
> **Model correction and CST validation**
>
> The updated formulation corrects excessive attenuation and errors associated with the definition of reference planes in the earlier surface-scattering model. These corrections have been checked against CST simulations.
>
> In the evaluated CST cases within the applicable Rayleigh surface-roughness limits, the revised model provides more reliable dielectric-parameter recovery than the conventional smooth-surface extraction, particularly for loss tangent. When surface-scattering attenuation is absent from the conventional model, fitting can incorrectly attribute it to intrinsic material absorption. This biases the extracted loss tangent; real permittivity is comparatively less affected in these evaluations.
>
> **Publication 1** below presents the current derivation, detailed validation, stress tests and uncertainty analysis. It is **Under Review** in the *Reviews of Electromagnetics*, EuCAP 2026 Special Issue.

This repository contains the latest, fully improved version of the THz extraction algorithm. To maintain academic and technical transparency:

- **Legacy Version:** The initial prototype of this software can be found at the [original prototype repository](https://github.com/Matthew-DBrown/THz-TDS-Material-Parameter-Extraction-Algorithm).
- **Evolution:** This implementation is a complete rewrite and improvement. While based on the same physical principles, the codebase has been fundamentally restructured.

### CST Validation Example: Al₂O₃ (TM)

The comparison below shows an Al₂O₃ slab under TM illumination, with a thickness of 2 mm and RMS roughness σₕ = 50 µm. The green dotted curves in the material-parameter panels represent the permittivity and loss tangent assigned in CST.

- **Left — conventional model (σₕ = 0):** Omitting surface-scattering attenuation causes the extraction to attribute that attenuation to intrinsic absorption, making the material appear substantially more lossy than specified in CST.
- **Centre — previous GO model (σₕ = 50 µm):** Excessive scattering attenuation in the earlier formulation drives the extracted loss tangent towards its near-zero lower bound, preventing recovery of the assigned intrinsic loss. The real-permittivity estimate still appears reasonable.
- **Right — corrected scattering-aware model (σₕ = 50 µm):** The revised formulation substantially improves recovery of the CST-assigned loss tangent while retaining a comparably accurate real-permittivity estimate. This example illustrates the improvement in recovering intrinsic material loss after correcting the scattering model.

<details>
<summary><strong>View the Al₂O₃ TM comparison (vector SVG)</strong></summary>

[![Al₂O₃ CST TM comparison: real permittivity, loss tangent, transfer magnitude and phase for the conventional, previous GO and corrected models.](docs/figures/al2o3_cst_tm_model_comparison.svg)](docs/figures/al2o3_cst_tm_model_comparison.svg)

[Open the full-size vector SVG](docs/figures/al2o3_cst_tm_model_comparison.svg).

</details>

## 📚 Citations & References

If you use this algorithm or its GO-based surface-scattering model, cite the relevant studies below. Use **Publication 1 for the current model derivation**.

> [!CAUTION]
> **Publication 1 — CURRENT MODEL DERIVATION · UNDER REVIEW**
>
> Alperen Sari, Yeganeh Farahi, Xen Sergi, Costas Constantinou and Miguel Navarro-Cía, “Validation and Uncertainties of a Geometrical Optics-Based Surface Scattering-Aware Dielectric Parameter Extraction Algorithm,” *Reviews of Electromagnetics*, EuCAP 2026 Special Issue, 2026. **Under Review.**

**Publication 2 — Published:** A. Sari et al., “Interlaboratory mmWave and THz Quasi-Optical Characterization of Commercial Conventional and 3-D Printable Substrates,” *IEEE Transactions on Terahertz Science and Technology*, vol. 16, no. 6, pp. 737–749, 2026. [DOI: 10.1109/TTHZ.2026.3674230](https://doi.org/10.1109/TTHZ.2026.3674230).

**Publication 3 — Published:** A. Sari, Y. Farahi, C. Constantinou and M. Navarro-Cía, “Uncertainty Analysis of a Surface Scattering-Aware Material Extraction Algorithm for THz-TDS Systems,” *2026 20th European Conference on Antennas and Propagation (EuCAP)*, Dublin, Ireland, 2026. [DOI: 10.23919/EuCAP68105.2026.11612344](https://doi.org/10.23919/EuCAP68105.2026.11612344).

**BibTeX:**

```bibtex
@article{sari2026roe,
  title   = {Validation and Uncertainties of a Geometrical Optics-Based Surface Scattering-Aware Dielectric Parameter Extraction Algorithm},
  author  = {Sari, Alperen and Farahi, Yeganeh and Sergi, Xen and Constantinou, Costas and Navarro-C{\'i}a, Miguel},
  journal = {Reviews of Electromagnetics},
  year    = {2026},
  note    = {Under Review; EuCAP 2026 Special Issue}
}

@article{sar2026thz,
  title   = {Interlaboratory {mmWave} and {THz} Quasi-Optical Characterization of Commercial Conventional and 3-D Printable Substrates},
  author  = {Sari, Alperen and others},
  journal = {IEEE Transactions on Terahertz Science and Technology},
  year    = {2026},
  volume  = {16},
  number  = {6},
  pages   = {737--749},
  doi     = {10.1109/TTHZ.2026.3674230}
}

@inproceedings{sari2026eucap,
  title     = {Uncertainty Analysis of a Surface Scattering-Aware Material Extraction Algorithm for {THz-TDS} Systems},
  author    = {Sari, Alperen and Farahi, Yeganeh and Constantinou, Costas and Navarro-C{\'i}a, Miguel},
  booktitle = {2026 20th European Conference on Antennas and Propagation (EuCAP)},
  address   = {Dublin, Ireland},
  year      = {2026},
  doi       = {10.23919/EuCAP68105.2026.11612344}
}
```

## 📂 Open Data & Reproducibility

The processed dataset obtained using this software suite, which supports the findings and characterization results presented in **[Sari et al., 2026]**, is publicly available.

You can access the full dataset via the link below:

👉 **Dataset Access:** [Open dataset](https://bham-my.sharepoint.com/personal/axs1927_student_bham_ac_uk/_layouts/15/guestaccess.aspx?share=IgDtw4CsmWmST4lbBxHRoLcLAUuLgpolsk_CvE-aKGRa0kM&e=K0natZ).

---

## 👤 Author

**Alperen Sari - PHDing at the University of Birmingham**

GitHub: [@alperensar](https://github.com/alperensar)
