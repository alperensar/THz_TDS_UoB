# tds_gui_selector.py
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# tds_gui_selector:
#   - THzInteractivePlotter: it enables selection of the signal parameters using GUI
#
# author@: Alperen Sari - PhDing at the University of Birmingham
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# TO DO: Apply whole process to multidata on GUI
# TO DO: New GUI libraries to run whole code from GUI
# TO DO: Improve GUI and make EXE
# TO DO: Slidebar dragging problem
from __future__ import annotations
import gc
import time
import json
import traceback
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.constants import c
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.widgets import Button, TextBox, RadioButtons, Slider, RangeSlider

import utils

THZ_TO_FREQ = 1E12
ERROR_EPSILON = 1e-20
TIME_SCALE_PS = 1e-12
STATE_FILENAME = ".tds_gui_defaults.json"


class THzInteractivePlotter:
    """Handles all GUI figures, widgets, saving PDF and user interaction."""

    def __init__(self, processor):
        self.proc = processor
        self._load_state_if_exists()
        self._ui_lock = False 

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.canvas.manager.set_window_title(f"THz Analysis - {getattr(self.proc, 'sample_name', 'Signal')}")

        # MAXIMIZE WINDOW ON STARTUP
        try:
            manager = plt.get_current_fig_manager()
            if hasattr(manager, 'window'):
                if hasattr(manager.window, 'state'):            # TkAgg
                    manager.window.state('zoomed')
                elif hasattr(manager.window, 'showMaximized'):  # Qt5Agg
                    manager.window.showMaximized()
            else:
                manager.full_screen_toggle()
        except:
            pass

        # Initialize Layout & Widgets
        self._setup_layout()
        self._setup_widgets()
        self._draw_figures()



    def _state_path(self) -> Path:
        try:
            return Path(getattr(self.proc, "sample_path", "")).parent / STATE_FILENAME
        except Exception:
            return Path.cwd() / STATE_FILENAME

    def _dump_state(self) -> dict:
        # proc üzerinde GUI ile değişen alanları sakla
        return {
            "noise_floor_db": float(getattr(self.proc, "noise_floor_db", -70.0)),
            "max_freq_before_noise": float(getattr(self.proc, "max_freq_before_noise", 3.0)),  # THz
            "f_min_extract": float(getattr(self.proc, "f_min_extract", 0.0)),   # Hz
            "f_max_extract": float(getattr(self.proc, "f_max_extract", 0.0)),   # Hz
            "unwrap_fit_min": float(getattr(self.proc, "unwrap_fit_min", 0.0)), # Hz
            "unwrap_fit_max": float(getattr(self.proc, "unwrap_fit_max", 0.0)), # Hz
            "fp_term_count": int(getattr(self.proc, "fp_term_count", 0)),
            "use_backup_unwrap": bool(getattr(self.proc, "use_backup_unwrap", False)),
            "version": 1,
        }

    def _apply_state(self, state: dict) -> None:
        # proc alanlarını güncelle
        if "noise_floor_db" in state:
            self.proc.noise_floor_db = float(state["noise_floor_db"])
        if "max_freq_before_noise" in state:
            self.proc.max_freq_before_noise = float(state["max_freq_before_noise"])

        for k in ("f_min_extract", "f_max_extract", "unwrap_fit_min", "unwrap_fit_max"):
            if k in state and state[k] is not None:
                setattr(self.proc, k, float(state[k]))

        if "fp_term_count" in state:
            self.proc.fp_term_count = int(state["fp_term_count"])
        if "use_backup_unwrap" in state:
            self.proc.use_backup_unwrap = bool(state["use_backup_unwrap"])

        # türetilmiş değerleri tazele (GUI’nin içindekiyle aynı mantık)
        try:
            utils.update_indices(self.proc)
            utils.phase_extrapolation(self.proc)
            utils.recalculate_absorption(self.proc)
        except Exception:
            pass

    def _load_state_if_exists(self) -> None:
        p = self._state_path()
        if not p.exists():
            return
        try:
            state = json.loads(p.read_text(encoding="utf-8"))
            self._apply_state(state)
            print(f"↩️ Loaded GUI defaults from: {p}")
        except Exception as e:
            print(f"⚠️ Could not load GUI defaults ({p.name}): {e}")

    def _save_state(self) -> None:
        p = self._state_path()
        try:
            p.write_text(json.dumps(self._dump_state(), indent=2), encoding="utf-8")
            print(f"💾 Saved GUI defaults to: {p}")
        except Exception as e:
            print(f"⚠️ Could not save GUI defaults ({p.name}): {e}")

    # GUI MAINS
    def _setup_layout(self):
        """Creates the 3x2 Grid Layout identical to the Export format."""
        plt.rcParams['font.family']     = 'Times New Roman'
        plt.rcParams['font.size']       = 12
        plt.rcParams['axes.titlesize']  = 12
        plt.rcParams['axes.labelsize']  = 12
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12

        plt.subplots_adjust(left=0.06, right=0.88, top=0.95, bottom=0.20, hspace=0.45, wspace=0.2)
        
        gs = gridspec.GridSpec(3, 2)

        # Assign Axes
        self.ax_phase = self.fig.add_subplot(gs[0, 0])   # (a) Phase
        self.ax_unw   = self.fig.add_subplot(gs[1, 0])   # (b) Unwrapped
        self.ax_fft   = self.fig.add_subplot(gs[2, 0])   # (c) FFT (Ref & Sample)
        self.ax_time  = self.fig.add_subplot(gs[0, 1])   # (d) Time-domain
        self.ax_n     = self.fig.add_subplot(gs[1, 1])   # (e) Refractive Index
        self.ax_alpha = self.fig.add_subplot(gs[2, 1])   # (f) Absorption

        try:
            base_dir = Path(getattr(self.proc, "sample_path", ""))
            self.fig.text(0.01, 0.01, str(base_dir),ha="left", va="bottom", transform=self.fig.transFigure,bbox=dict(facecolor='red', alpha=0.3, edgecolor='none', pad=3))
            base_dir = base_dir.parent
        except Exception:
            pass

    def _setup_widgets(self):
        """Sliders, textboxes, butons and range-sliders."""
        self._ui_lock = False

        label_y = 0.155
        box_width = 0.15
        box_height = 0.03
        box_level_y = 0.105
        slidebar_level_y = 0.06
        bbox_color = dict(facecolor='cyan', alpha=0.3, edgecolor='blue', boxstyle='round')

        extract_min_thz = self.proc.f_min_extract  / THZ_TO_FREQ
        extract_max_thz = self.proc.f_max_extract  / THZ_TO_FREQ
        unwrap_min_thz  = self.proc.unwrap_fit_min / THZ_TO_FREQ
        unwrap_max_thz  = self.proc.unwrap_fit_max / THZ_TO_FREQ
        freq_lim = np.nanmax(self.proc.fft_freqs)  / THZ_TO_FREQ

        # Vertical Noise sliderbar with label
        ax_slide_noise = plt.axes([0.9, 0.25, 0.02, 0.60])
        self.slider_noise = Slider(ax_slide_noise, 'Noise Floor\n(dB)', -160.0, -10.0, valinit=self.proc.noise_floor_db, orientation='vertical',)
        self.slider_noise.label.set_bbox(bbox_color)
        self.slider_noise.on_changed(self._on_noise_changed)

        # Noise box with labels w/o label
        ax_txt_noise = plt.axes([0.892, 0.18, 0.04, 0.035])
        self.box_noise = TextBox(ax_txt_noise, '', initial=f"{self.proc.noise_floor_db:.2f}",)
        self.box_noise.on_submit(self._on_noise_changed)

        # Vertical Cutoff Freq sliderbar with label
        ax_slide_cut = plt.axes([0.96, 0.25, 0.02, 0.60])
        self.slider_limit = Slider(ax_slide_cut,'Cutoff Freq.\n(THz)', 0, freq_lim, valinit=self.proc.max_freq_before_noise, orientation='vertical')
        self.slider_limit.label.set_bbox(bbox_color)
        self.slider_limit.on_changed(self._on_cutoff_changed)

        # Cutoff Freq box with labels w/o label
        ax_txt_cut = plt.axes([0.952, 0.18, 0.04, 0.035])
        self.box_cut = TextBox(ax_txt_cut,'', initial=f"{self.proc.max_freq_before_noise:.2f}")
        self.box_cut.on_submit(self._on_cutoff_changed)

        # Unwrap / Analysis / FP Boxes and Slidebars
        # Horizontal Unwrap sliderbar w/o label
        ax_r_unw = plt.axes([slidebar_level_y, slidebar_level_y, box_width, box_height])
        self.rslider_unw = RangeSlider(ax_r_unw, '', 0.0, freq_lim*0.6, valinit=(unwrap_min_thz, unwrap_max_thz),)
        self.rslider_unw.on_changed(self._on_unwrap_range_changed)

        # Unwrap Range box with label
        ax_txt_unw = plt.axes([slidebar_level_y, box_level_y, box_width, box_height])
        self.box_unw = TextBox(ax_txt_unw, '', initial=f"{unwrap_min_thz:.2f}-{unwrap_max_thz:.2f}",)
        self.box_unw.on_submit(self._on_unwrap_range_changed)
        self.fig.text(slidebar_level_y + box_width/2, label_y, "Unwrap Range", ha="center", va="center", transform=self.fig.transFigure, bbox=bbox_color,)

        # Horizontal Extracted Freqs sliderbar w/o label
        ax_r_anl = plt.axes([0.30, slidebar_level_y, box_width, box_height])
        self.rslider_anl = RangeSlider(ax_r_anl, '', 0.0, freq_lim*0.7, valinit=(extract_min_thz, extract_max_thz),)
        self.rslider_anl.on_changed(self._on_analysis_range_changed)

        # Extracted Freqs range box with label
        ax_txt_anl = plt.axes([0.30, box_level_y, box_width, box_height])
        self.box_anl = TextBox(ax_txt_anl, '', initial=f"{extract_min_thz:.2f}-{extract_max_thz:.2f}",)
        self.box_anl.on_submit(self._on_analysis_range_changed)
        self.fig.text(0.30 + box_width/2, label_y, "Analysis Range", ha="center", va="center", transform=self.fig.transFigure, bbox=bbox_color,)

        # FP Count box with label
        ax_box_fp = plt.axes([0.5, box_level_y, 0.04, box_height])
        self.box_fp = TextBox(ax_box_fp,'', initial=str(self.proc.fp_term_count),)
        self.box_fp.on_submit(self._update_fp_text)
        self.fig.text( 0.5 + 0.04/2, label_y, "FP Count", ha="center", va="center", transform=self.fig.transFigure, bbox=bbox_color,)

        # Unwrapping method buttons
        ax_radio = plt.axes([0.60, 0.02, 0.12, 0.12], frameon=False)
        self.radio = RadioButtons(ax_radio, ('Unwrap Func. 1', 'Unwrap Func. 2'), active=1 if self.proc.use_backup_unwrap else 0)
        self.radio.on_clicked(self._change_unwrap_method)

        # PDF export buttons
        ax_export = plt.axes([0.75, 0.04, 0.08, 0.06])
        self.btn_export = Button(ax_export, 'EXPORT\nPDF', color='lightblue', hovercolor='0.9')
        self.btn_export.on_clicked(self._export_publication_report)

        # Save GUI settings
        ax_save = plt.axes([0.84, 0.04, 0.08, 0.06])
        self.btn_save = Button(ax_save, 'SAVE\nSETTINGS', color='lightgreen', hovercolor='0.9')
        self.btn_save.on_clicked(self._save_and_close)

    def _draw_figures(self):
        """Clears GUI axes and calls the helper to redraw."""
        try:
            gc.collect()
            axes_list = [self.ax_phase, self.ax_fft, self.ax_time, self.ax_unw, self.ax_n, self.ax_alpha]
            for ax in axes_list:
                ax.clear()
                ax.grid(True, linestyle='--', alpha=0.2)

            self.helper_plot_data(
                self.ax_phase,   # Phase
                self.ax_fft,     # FFT
                self.ax_time,    # Time-domain
                self.ax_unw,     # Unwrapped
                self.ax_n,       # Refractive index
                self.ax_alpha,   # Absorption
                is_export=False,)
            
            self.fig.canvas.draw_idle()
            
            try:
                self.fig.canvas.flush_events()
            except NotImplementedError:
                pass

        except Exception as e:
            print("DRAW ERROR:", e)
            traceback.print_exc()
    

    # GUI HELPERS
    def _on_noise_changed(self, val):
        """Handles both slider and textbox updates for noise floor."""
        try:
            if self._ui_lock:
                return

            try:
                val = float(val)
            except ValueError:
                self._ui_lock = True
                self.box_noise.set_val(f"{self.proc.noise_floor_db:.2f}")
                self._ui_lock = False
                return

            self.noise_level = val
            self.proc.noise_floor_db = val 
            utils.recalculate_absorption(self.proc)

            self._ui_lock = True
            self.slider_noise.set_val(val)
            self.box_noise.set_val(f"{val:.2f}")
            self._ui_lock = False

            self._draw_figures()

        except Exception as e:
            print("DR Level ERROR:", e)
            traceback.print_exc()

    def _on_cutoff_changed(self, val):
        """Handles both slider and textbox updates for FFT cutoff frequency."""
        try:
            if self._ui_lock:
                return

            try:
                val = float(val)
            except ValueError:
                self._ui_lock = True
                self.box_cut.set_val(f"{self.proc.max_freq_before_noise:.2f}")
                self._ui_lock = False
                return

            self.cutoff_freq = val
            self.proc.max_freq_before_noise = val
            
            self._ui_lock = True
            self.slider_limit.set_val(val)
            self.box_cut.set_val(f"{val:.2f}")
            self._ui_lock = False

            self._draw_figures()

        except Exception as e:
            print("Max. Freq. Limit ERROR:", e)
            traceback.print_exc()

    def _on_unwrap_range_changed(self, vals):
        """
        Unified handler for both the unwrap range slider and the textbox.
        Accepts either:
            - vals = (fmin_thz, fmax_thz)   from RangeSlider
            - vals = "min-max" string       from TextBox
        """
        if self._ui_lock:
            return

        try:
            import numpy as np
            is_slider_event = isinstance(vals, (tuple, list, np.ndarray))

            if is_slider_event:
                fmin_thz, fmax_thz = float(vals[0]), float(vals[1])
            else:
                parsed = self._parse_range_text(vals)
                if not parsed:
                    fmin = self.proc.unwrap_fit_min / THZ_TO_FREQ
                    fmax = self.proc.unwrap_fit_max / THZ_TO_FREQ
                    self._ui_lock = True
                    try:
                        self.box_unw.set_val(f"{fmin:.2f}-{fmax:.2f}")
                    finally:
                        self._ui_lock = False
                    return
                fmin_thz, fmax_thz = parsed

            fmin_thz, fmax_thz = sorted([fmin_thz, fmax_thz])

            self.proc.unwrap_fit_min = fmin_thz * THZ_TO_FREQ
            self.proc.unwrap_fit_max = fmax_thz * THZ_TO_FREQ

            utils.update_indices(self.proc)
            utils.phase_extrapolation(self.proc)
            utils.recalculate_absorption(self.proc)

            self._ui_lock = True
            try:
                self.box_unw.set_val(f"{fmin_thz:.2f}-{fmax_thz:.2f}")
                if not is_slider_event:
                    self.rslider_unw.set_val((fmin_thz, fmax_thz))
            finally:
                self._ui_lock = False

            self._draw_figures()

        except Exception as e:
            print("Unwrap Range Selection ERROR:", e)
            traceback.print_exc()

    def _on_analysis_range_changed(self, vals):
        """
        Unified handler for both the analysis frequency range slider and textbox.
        Accepts:
            - vals = (fmin_thz, fmax_thz) from RangeSlider
            - vals = "min-max" string     from TextBox
        """
        if self._ui_lock:
            return

        try:
            import numpy as np
            is_slider_event = isinstance(vals, (tuple, list, np.ndarray))

            if is_slider_event:
                fmin_thz, fmax_thz = float(vals[0]), float(vals[1])
            else:
                parsed = self._parse_range_text(vals)
                if not parsed:
                    fmin = self.proc.f_min_extract / THZ_TO_FREQ
                    fmax = self.proc.f_max_extract / THZ_TO_FREQ
                    self._ui_lock = True
                    try:
                        self.box_anl.set_val(f"{fmin:.2f}-{fmax:.2f}")
                    finally:
                        self._ui_lock = False
                    return
                fmin_thz, fmax_thz = parsed

            fmin_thz, fmax_thz = sorted([fmin_thz, fmax_thz])

            self.proc.f_min_extract = fmin_thz * THZ_TO_FREQ
            self.proc.f_max_extract = fmax_thz * THZ_TO_FREQ

            utils.update_indices(self.proc)
            utils.recalculate_absorption(self.proc)

            self._ui_lock = True
            try:
                self.box_anl.set_val(f"{fmin_thz:.2f}-{fmax_thz:.2f}")
                if not is_slider_event:
                    self.rslider_anl.set_val((fmin_thz, fmax_thz))
            finally:
                self._ui_lock = False

            self._draw_figures()

        except Exception as e:
            print("Analysis Range Selection ERROR:", e)
            traceback.print_exc()

    def _update_fp_text(self, text):
        try:
            self.proc.fp_term_count = int(text)
            self._draw_figures()
        except Exception as e:
            print("FP Number Selection ERROR:", e)
            traceback.print_exc()

    def _change_unwrap_method(self, label):
        if  self._ui_lock:
            return
        
        new_val_is_backup = (label == 'Unwrap Func. 2')
        if self.proc.use_backup_unwrap == new_val_is_backup:
            return
        
        self._ui_lock = True
        try:
            self.proc.use_backup_unwrap = new_val_is_backup
            utils.recalculate_absorption(self.proc)
            
            self._draw_figures()
            
        except Exception as e:
            print("Unwrap Method Selection ERROR:", e)
            traceback.print_exc()
        
        finally:
            self._ui_lock = False
   
    def _export_publication_report(self, event=None):
        """ Exports the PDF report."""
        self._flash_button(self.btn_export)
        print("📊 Exporting report...")
        p = self.proc
        
        try:
            base_dir = Path(p.sample_path).parent
        except Exception:
            base_dir = Path.cwd()

        filename = f"{getattr(p, 'sample_name', 'Signal')}_report.pdf"
        out_path = base_dir / filename
        
        print(f"📍 Target File: {out_path.resolve()}")

        try:
            report_fig = Figure(figsize=(16, 12))
            FigureCanvasAgg(report_fig)

            gs = gridspec.GridSpec(3, 2, figure=report_fig, hspace=0.3, wspace=0.20)

            ax_phase   = report_fig.add_subplot(gs[0, 0])
            ax_unwrap  = report_fig.add_subplot(gs[1, 0])
            ax_fft     = report_fig.add_subplot(gs[2, 0])
            ax_time    = report_fig.add_subplot(gs[0, 1])
            ax_index   = report_fig.add_subplot(gs[1, 1])
            ax_alpha   = report_fig.add_subplot(gs[2, 1])

            self.helper_plot_data(
                ax_phase, ax_fft, ax_time, ax_unw=ax_unwrap, 
                ax_n=ax_index, ax_alpha=ax_alpha, is_export=True)
            
            try:
                report_fig.savefig(out_path, bbox_inches='tight', dpi=300)
                print(f"✅ SUCCESS: Report saved to: {out_path.name}")
            
            except PermissionError:
                print("\n" + "!"*40)
                print(f"❌ ERROR: Cannot write to '{filename}'.")
                print("⚠️  The file is likely OPEN in another program.")
                print("   Please close the PDF and try again.")
                print("!"*40 + "\n")
            
            except OSError as e:
                print(f"❌ ERROR: System permission issue: {e}")

        except Exception as e:
            print("❌ UNEXPECTED EXPORT ERROR:", e)
            traceback.print_exc()

    def _save_and_close(self, event):
        try:
            self._flash_button(self.btn_save)
            self._save_state()
            print("\n💾 Saving settings...")
            self.fig.canvas.flush_events()
            self.fig.canvas.start_event_loop(0.001) 

            try:
                self.fig.canvas.manager.window.close()
            except:
                try:
                    self.fig.canvas.manager.destroy()
                except:
                    plt.close(self.fig)

        except Exception as e:
            print("CLOSING ERROR:", e)
            traceback.print_exc()
    
    def _flash_button(self, button):
        """Flashing button """        
        if self.fig is None or self.fig.canvas is None:
            return

        try:
            try:
                original_facecolor = button.ax.get_facecolor()
            except AttributeError:
                return 

            button.ax.set_facecolor('gold')
            
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events() 
            except (SystemError, ValueError, AttributeError):
                pass
            
            time.sleep(0.15)
            
            try:
                button.ax.set_facecolor(original_facecolor)
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except (SystemError, ValueError, AttributeError):
                pass
            
        except Exception as e:
            print(f"Flash button ignored: {e}")
        except Exception as e:
            pass

    def _parse_range_text(self, text):
        try:
            clean = text.replace('-', ' ').replace(',', ' ')
            p = clean.split()
            if len(p) >= 2:
                a, b = float(p[0]), float(p[1])
                return (min(a, b), max(a, b))
        except ValueError:
            pass
        return None

    def _set_unwrap_range(self, fmin_thz, fmax_thz):
        try:
            fmin_thz, fmax_thz = sorted([fmin_thz, fmax_thz])

            self.proc.unwrap_fit_min = fmin_thz * THZ_TO_FREQ
            self.proc.unwrap_fit_max = fmax_thz * THZ_TO_FREQ

            utils.update_indices(self.proc)
            utils.phase_extrapolation(self.proc)
            utils.recalculate_absorption(self.proc)

            self._ui_lock = True
            try:
                self.box_unw.set_val(f"{fmin_thz:.2f}-{fmax_thz:.2f}")
                self.rslider_unw.set_val((fmin_thz, fmax_thz))
            finally:
                self._ui_lock = False

            self._draw_figures()

        except Exception as e:
            print("DRAW ERROR:", e)
            traceback.print_exc()
    
    def _set_analysis_range(self, fmin_thz, fmax_thz):
        fmin_thz, fmax_thz = sorted([fmin_thz, fmax_thz])

        self.proc.f_min_extract = fmin_thz * THZ_TO_FREQ
        self.proc.f_max_extract = fmax_thz * THZ_TO_FREQ
        utils.update_indices(self.proc)
        utils.recalculate_absorption(self.proc)

        self._ui_lock = True
        try:
            self.box_anl.set_val(f"{fmin_thz:.2f}-{fmax_thz:.2f}")
            self.rslider_anl.set_val((fmin_thz, fmax_thz))
        finally:
            self._ui_lock = False

        self._draw_figures()

    # GRAPHS
    def helper_plot_data(self, ax_phase, ax_fft, ax_time, ax_unw, ax_n, ax_alpha, is_export: bool = False):
        """
        Render all 6 panels (phase, unwrapped phase, FFT, time-domain, refractive index, absorption)
        in a single function so that both the GUI and export paths share identical plotting logic.
        """
        p = self.proc
        t_ps = p.time_data / TIME_SCALE_PS

        freq_Hz = p.fft_freqs
        freq_THz = freq_Hz / THZ_TO_FREQ
        full_min_THz = np.nanmin(freq_THz)
        full_max_THz = np.nanmax(freq_THz)

        if freq_Hz.size == 0 or np.all(~np.isfinite(freq_Hz)):
            return
        
        # Analysis and unwrap frequency ranges (if available)
        noise_floor_db = p.noise_floor_db or -70.0
        f_cut_THz = p.max_freq_before_noise or 3.0
        has_analysis = hasattr(p, "f_min_extract") and hasattr(p, "f_max_extract")
        if has_analysis:
            band_min_THz = p.f_min_extract / THZ_TO_FREQ
            band_max_THz = p.f_max_extract / THZ_TO_FREQ

        has_unwrap = hasattr(p, "unwrap_fit_min") and hasattr(p, "unwrap_fit_max")
        if has_unwrap:
            unwrap_min_THz = p.unwrap_fit_min / THZ_TO_FREQ
            unwrap_max_THz = p.unwrap_fit_max / THZ_TO_FREQ

        # Legend style: smaller font when exporting to PDF
        legend_kwargs = {"loc": "best"}
        xlim_kwargs = {"left": full_min_THz, "right": full_max_THz}
        grid_kwargs = {"visible": True, "linestyle": '--', "alpha": 0.4}
        if is_export:
            legend_kwargs["fontsize"] = 12

        # Shared helper to draw analysis/unwrap spans and the cutoff line.
        def _apply_frequency_spans(ax):
            if has_analysis:
                ax.axvspan(band_min_THz, band_max_THz, color='lightblue', alpha=0.3)
            if has_unwrap:
                ax.axvspan(unwrap_min_THz, unwrap_max_THz, color='green', alpha=0.15)
            ax.axvline(f_cut_THz, color='gray', linestyle='--', alpha=0.5)

        # (a) Phase
        ax_phase.cla()
        ax_phase.plot(freq_THz, p.measured_phase, color='b',  linewidth=1.0, label=r'Measured $\angle H$')
        _apply_frequency_spans(ax_phase)

        ax_phase.set_ylabel("Phase (rad)")
        ax_phase.set_xlabel("Frequency (THz)")
        ax_phase.set_title("(a) $H_{measured}$ phase")
        ax_phase.set_xlim(**xlim_kwargs)
        ax_phase.grid(**grid_kwargs)
        ax_phase.legend(**legend_kwargs)

        # (b) Unwrapped Phase
        ax_unw.cla()
        use_backup = p.use_backup_unwrap or False

        # Linear fit (expected phase)
        popt = None
        if use_backup and hasattr(p, "popt_backup"):
            popt = p.popt_backup
        elif (not use_backup) and hasattr(p, "popt"):
            popt = p.popt

        if popt is not None:
            ax_unw.plot(freq_THz, popt[0] * freq_Hz, color='r', linestyle='--', label='Expected linear phase',)

        # Unwrap function 1 (backup)
        if hasattr(p, "zero_unwrapped_phase_backup"):
            lw_b = 2.0 if use_backup else 1.0
            ax_unw.plot(freq_THz, p.zero_unwrapped_phase_backup, color='b', linewidth=lw_b, label='Unwrap (backup)',)

        # Unwrap function 2 (primary)
        if hasattr(p, "zero_unwrapped_phase"):
            lw_g = 2.0 if not use_backup else 1.0
            ax_unw.plot(freq_THz, p.zero_unwrapped_phase, color='g', linewidth=lw_g, label='Unwrap (primary)',)

        _apply_frequency_spans(ax_unw)

        ax_unw.set_ylabel("Phase (rad)")
        ax_unw.set_xlabel("Frequency (THz)")
        ax_unw.set_title("(b) $H_{measured}$ unwrapped phase")
        ax_unw.set_xlim(**xlim_kwargs)
        ax_unw.grid(**grid_kwargs)
        ax_unw.legend(**legend_kwargs)

        # (c) FFT (Reference & Sample) + Noise Floor
        ax_fft.cla()
        ref_db = 20 * np.log10(np.abs(p.fft_reference) + ERROR_EPSILON)
        sam_db = 20 * np.log10(np.abs(p.fft_sample)    + ERROR_EPSILON)

        ax_fft.plot(freq_THz, ref_db, 'r', label='Reference FFT')
        ax_fft.plot(freq_THz, sam_db, 'b', label='Sample FFT')

        # Noise floor (horizontal line)
        ref_db = 20 * np.log10(np.abs(p.fft_reference) + ERROR_EPSILON)
        ref_peak_db = np.nanmax(ref_db)  # veya sadece ROI/band içinde max (aşağıda)

        noise_floor_line_db = ref_peak_db + noise_floor_db  # noise_floor_db negatif offset
        ax_fft.axhline(noise_floor_line_db, color='purple', linestyle=':',label=f'Noise floor (rel {noise_floor_db:.0f} dB)')

        # ax_fft.axhline(noise_floor_db, color='purple', linestyle=':', label='Noise floor')

        _apply_frequency_spans(ax_fft)

        ax_fft.set_ylabel("Magnitude (dB)")
        ax_fft.set_xlabel("Frequency (THz)")
        ax_fft.set_title("(c) Frequency-domain measurement signals")
        ax_fft.set_xlim(**xlim_kwargs)
        ax_fft.grid(**grid_kwargs)
        ax_fft.legend(**legend_kwargs)

        min_db = np.nanmin([np.nanmin(ref_db), np.nanmin(sam_db)])
        ax_fft.set_ylim(bottom=min_db - 10)

        # (d) Time Domain
        ax_time.cla()
        y_sample = None

        if hasattr(p, "reference") and hasattr(p.reference, "y_data"):
            ax_time.plot(t_ps, np.asarray(p.reference.y_data, dtype=float), 'r', alpha=0.5, label='Reference',)

        if hasattr(p, "sample") and hasattr(p.sample, "y_data"):
            y_sample = np.asarray(p.sample.y_data, dtype=float)
            ax_time.plot(t_ps, y_sample, 'b', label='Sample')

        # Fabry–Perot echoes: taken from processor as fp_echo_times in seconds
        if (hasattr(p, "fp_echo_times") and p.fp_echo_times is not None and getattr(p, "fp_term_count", 0) > 0):
            echo_ns = np.asarray(p.fp_echo_times, dtype=float) / TIME_SCALE_PS  # s → ns
            for t_line in echo_ns:
                ax_time.axvline(t_line, color='green', linestyle=':', alpha=0.7)

        ax_time.set_ylabel("Amplitude")
        ax_time.set_xlabel("Time (ps)")
        ax_time.set_title(f"(d) Time-domain signals (FP terms: {getattr(p, 'fp_term_count', 0)})")
        ax_time.autoscale(enable=True, tight=True)
        ax_time.grid(**grid_kwargs)
        ax_time.legend(**legend_kwargs)

        # (e) Refractive Index n(f)
        ax_n.cla()
        n = getattr(p, "current_n", None)
        valid_mask = getattr(p, "valid_mask", np.ones_like(freq_Hz, dtype=bool))
        valid_mask = np.asarray(valid_mask, dtype=bool)

        if n is not None and np.size(n) == freq_Hz.size:
            n_arr = np.asarray(n, dtype=float)
            ax_n.plot(freq_THz[valid_mask], n_arr[valid_mask], 'red', label='n')
            ax_n.fill_between(freq_THz, 0, 1, where=~valid_mask, color='red', alpha=0.1,
                              transform=ax_n.get_xaxis_transform(), label='Excluded (Noise)')

        _apply_frequency_spans(ax_n)

        ax_n.set_title("(e) Refractive index")
        ax_n.set_ylabel("n")
        ax_n.set_xlabel("Frequency (THz)")
        ax_n.set_xlim(**xlim_kwargs)
        ax_n.grid(**grid_kwargs)
        ax_n.legend(**legend_kwargs)

        if n is not None and np.any(valid_mask):
            n_valid = np.asarray(n, dtype=float)[valid_mask]
            n_valid = n_valid[np.isfinite(n_valid)]
            
            if n_valid.size > 0:
                y_min = np.min(n_valid)
                y_max = np.max(n_valid)
                margin = (y_max - y_min) * 0.1

                if margin == 0: 
                    margin = 0.1
                
                ax_n.set_ylim(y_min - margin, y_max + margin)

        # (f) Absorption α and α_max
        ax_alpha.cla()
        alpha     = getattr(p, "current_alpha", None)
        alpha_max = getattr(p, "current_alpha_max", None)

        if alpha is not None and np.size(alpha) == freq_Hz.size:
            ax_alpha.plot(freq_THz[valid_mask], np.asarray(alpha, dtype=float)[valid_mask], 'r', linewidth=1.4, label=r'$\alpha$',)

        if alpha_max is not None and np.size(alpha_max) == freq_Hz.size:
            ax_alpha.plot(freq_THz[valid_mask], np.asarray(alpha_max, dtype=float)[valid_mask],
                color='b', linewidth=1.4, label=rf'$\alpha_{{max}}$ (noise floor {noise_floor_db:.0f} dB)',)
        
        _apply_frequency_spans(ax_alpha)
     
        ax_alpha.set_title("(f) $\\alpha$ and $\\alpha_{max}$")
        ax_alpha.set_xlabel("Frequency (THz)")
        ax_alpha.set_ylabel("Absorption (cm$^{-1}$)")
        ax_alpha.autoscale(enable=True, axis='y', tight=True)
        ax_alpha.set_xlim(**xlim_kwargs)
        ax_alpha.grid(**grid_kwargs)
        ax_alpha.legend(**legend_kwargs)

if __name__ == "__main__":
    pass
