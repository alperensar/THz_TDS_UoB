# THz-TDS Dielectric Parameter Extraction Algorithm

This software is an in-house developed algorithm in the University of Birmingham designed to extract complex dielectric parameters from **Terahertz Time-Domain Spectroscopy (THz-TDS)** measurements. It features a dual-stage workflow, advanced optimization routines, and a unique surface roughness-induced scatterring compensation model based on Geometric Optics.

## 🚀 Key Features

- **Two-Stage Extraction:** Seamless conversion from measured time-domain signals (FFT) to frequency-domain parameter extraction.
- **Advanced Optimization:** Support for **Nelder-Mead** simplex and **Least-Squares** based optimization processes.
- **Surface Roughness Modeling:** Incorporates physical scattering effects using **Geometric Optics-based modeling**, enabling high-accuracy extraction for samples with rough surfaces.
- **Performance:** High-speed processing with **Multiprocessing (Multi-CPU)** support for large datasets.
- **Interactive Workflow:** User-friendly GUI for parameter adjustments and visual confirmation before saving results.

---


## 📦 Installation & Setup

### Option A — Editable install (pip)

```bash
# Clone the repository
git clone https://github.com/alperensar/THz_UoB.git
cd THz_UoB

# Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate         # macOS/Linux
# .\.venv\Scripts\Activate.ps1   # Windows (PowerShell)

# Install in editable mode
pip install -e .
```

### Option B — Reproducible install from `pyproject.toml` + lock file

If your repo contains `pyproject.toml` plus a lock file (e.g., `poetry.lock` or `uv.lock`), you can install **exactly** the pinned dependencies:

**Poetry (`poetry.lock`):**
```bash
pip install poetry
poetry install
```

**uv (`uv.lock`):**
```bash
pip install uv
uv sync
```

> ⚠️ **Critical Configuration:** Before execution, you must define your working root directory in `src/tds_extraction_CLI.py` within the `main` function:
>
> `root_dir: Optional[Path] = typer.Option("/Users/alp/Desktop/Toptica_UoB_ROOT", ...)`

---

## ⚙️ Data Structure Requirements

The algorithm expects a strict directory hierarchy and a metadata file for sample thicknesses.

### 1) Directory Tree

```text
Root_Directory/
├── thickness_info.txt           # Metadata file (Required)
├── Sample_Name/                 # Level 1: Folder name must match metadata 'material'
│   └── Measurement_System/      # Level 2: e.g., Toptica_TDS
│       └── pulse/               # Required folder name
│           ├── sample_data.txt  # Time-domain signals (Reference & Sample)
│           └── reference.txt    # (Optional) Separate reference measurement
```

### 2) Metadata File (`thickness_info.txt`)

Create a **tab-separated** file in your Root Directory. Set `sigma_um` to **0** to disable the surface roughness model. You can check the expected folder structure with thickness.txt in folder_struct folder.

**Header must be exactly:**
```text
material	thickness_mm	sigma_um	epsR
```

**Example content:**
```text
ROGERS_TMM3	3.1516	0.382325
ROGERS_RT6002	3.0707	0.613363
ROGERS_RT5880LZ	3.1334	0.586083
```

---

## 🛠 Execution Modes

The software provides three main modes. You can run these via CLI using the `--config config.yaml` argument or use the pre-defined VSCode Launch Profiles.

1. **THz TDS Processing (FFT Stage):** Performs time-to-frequency domain conversion.
2. **THz TDS Extraction (Serial):** Runs the extraction on a single CPU core.
3. **THz TDS Extraction (Multiprocessing):** Utilizes multi-CPU support for faster processing.

### VSCode Launch Configurations (`.vscode/launch.json`)

```json
{
  "version": "0.1.0",
  "configurations": [
    {
      "name": "THz TDS Processing",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/tds_preprocessing_CLI.py",
      "args": ["--config", "config.yaml"]
    },
    {
      "name": "THz TDS Extraction (Serial)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/tds_extraction_CLI.py",
      "args": ["--config", "config.yaml", "--backend", "serial"]
    },
    {
      "name": "THz TDS Extraction (Multiprocessing)",
      "type": "debugpy",
      "request": "launch",
      "program": "${workspaceFolder}/src/tds_extraction_CLI.py",
      "args": ["--config", "config.yaml", "--backend", "multiprocessing"]
    }
  ]
}
```

---

## 📜 Disclosure & Historical Note

This repository contains the latest, fully improved version of the THz extraction algorithm. To maintain academic and technical transparency:

- **Legacy Version:** The initial prototype of this software can be found at: **(https://github.com/Matthew-DBrown/THz-TDS-Material-Parameter-Extraction-Algorithm)**
- **Evolution:** This implementation is a complete rewrite amd improved. While based on the same physical principles, the codebase has been fundamentally restructured.

---

## 📚 Citations & References

If you utilize this algorithm or the Geometric Optics scattering model in your research, please cite:

- Publication 1: A. Sari and others, "Interlaboratory mmWave and THz Quasi-Optical
Characterization of Commercial Conventional and
3-D Printable Substrates," IEEE Transactions on Terahertz Science and Technology
, 2026 (Under Review).
- Publication 2: A. Sari, Y. Farahi, C. Constantinou, M. Navarro Cía, "Uncertainty Analysis of a Surface Scattering-Aware
Material Extraction Algorithm for THz-TDS
Systems," EuCAP, Dublin, Ireland, 2026.

**BibTeX snippet:**
```bibtex
@article{sar2026thz,
  title   = {Interlaboratory mmWave and THz Quasi-Optical Characterization of Commercial Conventional and 3-D Printable Substrates},
  author  = {Sari, Alperen and others.},
  journal = {IEEE Transactions on Terahertz Science and Technology},
  year    = {2026}
}

@article{sar2025EUCAP,
  title   = {Uncertainty Analysis of a Surface Scattering-Aware Material Extraction Algorithm for THz-TDS Systems},
  author  = {Sari, Alperen and Farahi, Yeganeh and Constantinou, Costas and Navarro Cía, Miguel.},
  journal = {EuCAP},
  year    = {2026 (Accepted)}
}
```

---

## 📂 Open Data & Reproducibility

The processed dataset obtained using this software suite, which supports the findings and characterization results presented in **[Sari et al., 2026]**, is publicly available.

You can access the full dataset via the link below:

👉 **Dataset Access:** [https://bham-my.sharepoint.com/personal/axs1927_student_bham_ac_uk/_layouts/15/guestaccess.aspx?share=IgDtw4CsmWmST4lbBxHRoLcLAUuLgpolsk_CvE-aKGRa0kM&e=K0natZ]

--

## 👤 Author

**Alperen Sari - PHDing at the University of Birmingham**  
GitHub: **@alperensar**
