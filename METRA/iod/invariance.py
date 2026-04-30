import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, RadioButtons, Slider


def darken_color(hex_color, amount=0.3):
    import matplotlib.colors as mc

    c = np.array(mc.to_rgb(hex_color))
    return tuple(c * (1 - amount))


class InteractiveRelativeAccuracyEditor:
    def __init__(self):
        self.x_original = np.array([
            5, 7, 8, 10, 12, 15, 18, 22, 26, 32,
            38, 45, 50, 55, 65, 75, 85, 95, 100
        ])

        self.ys_original = {
    "continual_backpropagation": [
        0.0, 0.0, 0.5, 1.0, 2.5, 6.0, 10.0, 15.0, 20.0,
        25.0, 30.0, 34.0, 36.0, 38.0, 40.0, 41.5, 42.5, 43.5, 44.5
    ],
    "shrink_and_perturb": [
        0.0, 0.0, 0.3, 0.8, 2.0, 5.0, 9.0, 14.0, 19.0,
        24.0, 29.0, 33.0, 35.0, 37.0, 39.0, 40.5, 41.5, 42.5, 43.5
    ],
    "head_resetting": [
        0.0, 0.0, 0.1, 0.3, 1.0, 3.0, 6.0, 9.0, 10.5,
        9.5, 8.5, 9.0, 9.5, 8.8, 9.0, 9.2, 9.0, 9.1, 9.0
    ],
    "base_deep_learning_system": [
        0.0, 0.0, 0.8, 2.0, 5.0, 10.0, 16.0, 22.0, 28.0,
        34.0, 39.0, 44.0, 47.0, 50.0, 52.5, 54.0, 55.5, 56.5, 57.5
    ],
}

        self.colors = {
            "continual_backpropagation": "#1f77b4",
            "shrink_and_perturb": "#ff7f0e",
            "head_resetting": "#2ca02c",
            "base_deep_learning_system": "#d62728",
        }

        self.x_dense = np.linspace(self.x_original.min(), self.x_original.max(), 400)

        self.noise_amount = 0.0
        self.noise_seed = 13
        self.rng = np.random.default_rng(self.noise_seed)
        self.noise_by_alg = {}

        self.lines = []
        self.line_info = {}
        self.original_y = {}
        self.current_y = {}
        self.original_std = {}
        self.current_std = {}
        self.std_patches = {}

        self.mean_cps = {}
        self.std_cps = {}

        self.mean_cp_artists = []
        self.std_cp_artists = []

        self.active = None
        self.undo_stack = []
        self.redo_stack = []

        self.save_dir = Path("Outputs") / "RelativeAccuracyPlots"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.fig, self.ax = plt.subplots(figsize=(10.5, 6.2), dpi=150)
        plt.subplots_adjust(left=0.10, right=0.75, bottom=0.25, top=0.92)

        self._set_style()
        self._make_widgets()
        self._connect_events()
        self.rebuild_plot()

    def _set_style(self):
        plt.rcParams.update({
            "font.family": "DejaVu Sans",
            "font.size": 18,
            "axes.labelsize": 24,
            "xtick.labelsize": 20,
            "ytick.labelsize": 20,
            "axes.linewidth": 1,
        })

    def _make_widgets(self):
        def make_slider(rect, label, vmin, vmax, vinit, step=None):
            ax_s = self.fig.add_axes(rect)
            slider = Slider(ax_s, label, vmin, vmax, valinit=vinit, valstep=step)
            slider.on_changed(self.on_slider_change)
            return slider

        self.s_noise = make_slider(
            [0.13, 0.05, 0.54, 0.025],
            "Noise",
            0.0,
            1.0,
            self.noise_amount,
            0.01,
        )

        ax_reset = self.fig.add_axes([0.79, 0.80, 0.16, 0.05])
        self.b_reset = Button(ax_reset, "Reset edits")
        self.b_reset.on_clicked(self.reset_edits)

        ax_undo = self.fig.add_axes([0.79, 0.72, 0.075, 0.05])
        self.b_undo = Button(ax_undo, "Undo")
        self.b_undo.on_clicked(self.undo)

        ax_redo = self.fig.add_axes([0.875, 0.72, 0.075, 0.05])
        self.b_redo = Button(ax_redo, "Redo")
        self.b_redo.on_clicked(self.redo)

        ax_save = self.fig.add_axes([0.79, 0.64, 0.16, 0.05])
        self.b_save = Button(ax_save, "Save plot")
        self.b_save.on_clicked(self.save_plot)

        ax_reedit = self.fig.add_axes([0.79, 0.56, 0.16, 0.05])
        self.b_reedit = Button(ax_reedit, "Re-edit CSV")
        self.b_reedit.on_clicked(self.re_edit_from_output)

        ax_mode = self.fig.add_axes([0.79, 0.40, 0.16, 0.12])
        self.radio = RadioButtons(ax_mode, ("Mean point", "STD point"))

        ax_help = self.fig.add_axes([0.77, 0.08, 0.22, 0.28])
        ax_help.axis("off")
        ax_help.text(
            0,
            1,
            "Mouse editing:\n"
            "• Double click = add mean point\n"
            "• Shift + double click = add STD point\n"
            "• Drag circle = move curve\n"
            "• Drag square = move shading\n"
            "• Endpoints are draggable\n"
            "• Right click point = delete\n\n"
            "Buttons:\n"
            "• Save plot = clean PNG/PDF/SVG + CSV\n"
            "• Re-edit CSV = reload saved data",
            va="top",
            fontsize=9,
        )

    def _connect_events(self):
        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("button_release_event", self.on_release)

    def make_smooth_noise(self, alg):
        if alg not in self.noise_by_alg:
            raw_noise = self.rng.normal(0, 1, len(self.x_dense))
            kernel = np.ones(15) / 15
            smooth_noise = np.convolve(raw_noise, kernel, mode="same")
            self.noise_by_alg[alg] = smooth_noise

        return self.noise_amount * self.noise_by_alg[alg]

    def make_base_curve(self, alg):
        y = np.array(self.ys_original[alg], dtype=float)
        valid = ~np.isnan(y)
        base = np.interp(self.x_dense, self.x_original[valid], y[valid])
        return base + self.make_smooth_noise(alg)

    def make_base_std(self):
        t = np.linspace(0, 1, len(self.x_dense))
        return 0.10 + 1.8 * (t ** 3)

    def on_slider_change(self, _):
        self.noise_amount = float(self.s_noise.val)
        self.rebuild_plot()

    def rebuild_plot(self):
        old_mean_cps = {}
        old_std_cps = {}

        if self.lines:
            for line in self.lines:
                name = self.line_info[line]
                old_mean_cps[name] = copy.deepcopy(self.mean_cps[line])
                old_std_cps[name] = copy.deepcopy(self.std_cps[line])

        self.ax.clear()
        self.clear_control_artists()

        self.lines = []
        self.line_info = {}
        self.original_y = {}
        self.current_y = {}
        self.original_std = {}
        self.current_std = {}
        self.std_patches = {}
        self.mean_cps = {}
        self.std_cps = {}

        for alg in self.ys_original.keys():
            y = self.make_base_curve(alg)
            std = self.make_base_std()
            color = self.colors[alg]

            line, = self.ax.plot(
                self.x_dense,
                y,
                color=color,
                linewidth=1.2,
                label=alg,
                zorder=5,
            )

            self.lines.append(line)
            self.line_info[line] = alg
            self.original_y[line] = y.copy()
            self.current_y[line] = y.copy()
            self.original_std[line] = std.copy()
            self.current_std[line] = std.copy()

            self.mean_cps[line] = old_mean_cps.get(
                alg,
                [(self.x_dense[0], 0.0), (self.x_dense[-1], 0.0)],
            )

            self.std_cps[line] = old_std_cps.get(
                alg,
                [(self.x_dense[0], 0.0), (self.x_dense[-1], 0.0)],
            )

            self.std_patches[line] = None
            self.apply_edits(line)

        self.format_axes()
        self.redraw_control_points()
        self.fig.canvas.draw_idle()

    def format_axes(self):
        self.ax.set_xlim(0, 105)
        self.ax.set_ylim(0, 75)

        self.ax.set_xticks([5, 50, 100])
        self.ax.set_yticks([0, 10, 20, 30, 40, 50,60,70])

        self.ax.set_xlabel("Number of Classes")
        self.ax.set_ylabel("")

        self.ax.set_title(
            "Percentage of Dormant Neurons",
            fontsize=24,
            pad=10,
        )

        self.ax.yaxis.grid(True, alpha=0.35)
        self.ax.xaxis.grid(False)

        legend = self.ax.get_legend()
        if legend is not None:
            legend.remove()

            self.ax.yaxis.grid(True, alpha=0.35)
            self.ax.xaxis.grid(False)

            legend = self.ax.get_legend()
            if legend is not None:
                legend.remove()

    def get_state(self):
        return {
            "mean": {
                self.line_info[line]: copy.deepcopy(self.mean_cps[line])
                for line in self.lines
            },
            "std": {
                self.line_info[line]: copy.deepcopy(self.std_cps[line])
                for line in self.lines
            },
        }

    def save_state(self):
        self.undo_stack.append(self.get_state())
        self.redo_stack.clear()

    def restore_state(self, state):
        for line in self.lines:
            name = self.line_info[line]

            if name in state["mean"]:
                self.mean_cps[line] = copy.deepcopy(state["mean"][name])

            if name in state["std"]:
                self.std_cps[line] = copy.deepcopy(state["std"][name])

            self.apply_edits(line)

        self.redraw_control_points()

    def undo(self, event=None):
        if not self.undo_stack:
            return

        self.redo_stack.append(self.get_state())
        previous = self.undo_stack.pop()
        self.restore_state(previous)

    def redo(self, event=None):
        if not self.redo_stack:
            return

        self.undo_stack.append(self.get_state())
        next_state = self.redo_stack.pop()
        self.restore_state(next_state)

    def reset_edits(self, event=None):
        self.save_state()

        for line in self.lines:
            self.mean_cps[line] = [
                (self.x_dense[0], 0.0),
                (self.x_dense[-1], 0.0),
            ]

            self.std_cps[line] = [
                (self.x_dense[0], 0.0),
                (self.x_dense[-1], 0.0),
            ]

            self.apply_edits(line)

        self.redraw_control_points()

    def overlay(self, controls):
        controls = sorted(controls, key=lambda t: t[0])
        kx = [px for px, _ in controls]
        ky = [py for _, py in controls]
        return np.interp(self.x_dense, np.array(kx), np.array(ky))

    def apply_edits(self, line):
        y = self.original_y[line] + self.overlay(self.mean_cps[line])

        std = self.original_std[line] + self.overlay(self.std_cps[line])
        std = np.maximum(0.02, std)

        line.set_ydata(y)

        self.current_y[line] = y.copy()
        self.current_std[line] = std.copy()

        self.update_std_patch(line)
        self.fig.canvas.draw_idle()

    def update_std_patch(self, line):
        if self.std_patches.get(line) is not None:
            try:
                self.std_patches[line].remove()
            except Exception:
                pass

        x = np.asarray(line.get_xdata(), dtype=float)
        y = np.asarray(line.get_ydata(), dtype=float)
        std = self.current_std[line]

        self.std_patches[line] = self.ax.fill_between(
            x,
            y - std,
            y + std,
            color=line.get_color(),
            alpha=0.08,
            linewidth=0,
            zorder=1,
        )

        line.set_zorder(5)
        self.fig.canvas.draw_idle()

    def nearest_line(self, x_click, y_click):
        best_line = None
        best_dist = np.inf

        for line in self.lines:
            x = np.asarray(line.get_xdata())
            y = np.asarray(line.get_ydata())

            d = np.min((x - x_click) ** 2 + (y - y_click) ** 2)

            if d < best_dist:
                best_dist = d
                best_line = line

        return best_line

    def clear_control_artists(self):
        for artist in self.mean_cp_artists + self.std_cp_artists:
            try:
                artist.remove()
            except Exception:
                pass

        self.mean_cp_artists = []
        self.std_cp_artists = []

    def redraw_control_points(self):
        self.clear_control_artists()

        for line in self.lines:
            self.draw_control_points(line, target="mean")
            self.draw_control_points(line, target="std")

        self.fig.canvas.draw_idle()

    def draw_control_points(self, line, target):
        controls = self.mean_cps[line] if target == "mean" else self.std_cps[line]

        if not controls:
            return

        marker = "o" if target == "mean" else "s"
        size = 52 if target == "mean" else 42
        artists = self.mean_cp_artists if target == "mean" else self.std_cp_artists

        xs = []
        ys = []

        for px, py in controls:
            xs.append(px)

            if target == "mean":
                y_base = np.interp(px, self.x_dense, self.original_y[line])
                ys.append(y_base + py)
            else:
                y_curve = np.interp(px, self.x_dense, self.current_y[line])
                std_base = np.interp(px, self.x_dense, self.original_std[line])
                ys.append(y_curve + std_base + py)

        sc = self.ax.scatter(
            xs,
            ys,
            s=size,
            marker=marker,
            color=line.get_color(),
            edgecolors="black" if target == "mean" else "white",
            zorder=8 if target == "mean" else 7,
        )

        artists.append(sc)

    def nearest_control(self, x_screen, y_screen, max_px=14):
        best = None
        best_dist = np.inf
        trans = self.ax.transData

        for target in ("mean", "std"):
            controls_dict = self.mean_cps if target == "mean" else self.std_cps

            for line in self.lines:
                controls = controls_dict[line]

                for idx, (px, py) in enumerate(controls):
                    if target == "mean":
                        y_base = np.interp(px, self.x_dense, self.original_y[line])
                        y_draw = y_base + py
                    else:
                        y_curve = np.interp(px, self.x_dense, self.current_y[line])
                        std_base = np.interp(px, self.x_dense, self.original_std[line])
                        y_draw = y_curve + std_base + py

                    sx, sy = trans.transform((px, y_draw))
                    dist = np.hypot(sx - x_screen, sy - y_screen)

                    if dist < best_dist:
                        best_dist = dist
                        best = (line, idx, target)

        return best if best_dist <= max_px else None

    def add_control_point(self, line, xdata, ydata, target):
        self.save_state()

        xdata = np.clip(xdata, self.x_dense[0], self.x_dense[-1])

        if target == "mean":
            y_base = np.interp(xdata, self.x_dense, self.original_y[line])
            delta = ydata - y_base
        else:
            y_curve = np.interp(xdata, self.x_dense, self.current_y[line])
            std_base = np.interp(xdata, self.x_dense, self.original_std[line])
            delta = ydata - y_curve - std_base
            delta = max(-std_base + 0.02, delta)

        controls = self.mean_cps if target == "mean" else self.std_cps
        controls[line].append((xdata, delta))
        controls[line].sort(key=lambda t: t[0])

        self.apply_edits(line)
        self.redraw_control_points()

        new_idx = controls[line].index((xdata, delta))
        self.active = (line, new_idx, target)

    def on_press(self, event):
        if event.inaxes != self.ax:
            return

        if event.button == 3:
            hit = self.nearest_control(event.x, event.y)

            if hit:
                line, idx, target = hit
                controls = self.mean_cps if target == "mean" else self.std_cps

                if idx == 0 or idx == len(controls[line]) - 1:
                    return

                self.save_state()
                del controls[line][idx]

                self.apply_edits(line)
                self.redraw_control_points()

            return

        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        hit = self.nearest_control(event.x, event.y)

        if hit:
            self.save_state()
            self.active = hit
            return

        if event.dblclick:
            line = self.nearest_line(event.xdata, event.ydata)
            selected = self.radio.value_selected
            target = "std" if (event.key == "shift" or selected == "STD point") else "mean"
            self.add_control_point(line, event.xdata, event.ydata, target)

    def on_motion(self, event):
        if not self.active or event.inaxes != self.ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        line, idx, target = self.active
        controls = self.mean_cps if target == "mean" else self.std_cps

        if idx == 0:
            xdata = self.x_dense[0]
        elif idx == len(controls[line]) - 1:
            xdata = self.x_dense[-1]
        else:
            left_bound = controls[line][idx - 1][0] + 1e-6
            right_bound = controls[line][idx + 1][0] - 1e-6
            xdata = np.clip(event.xdata, left_bound, right_bound)

        if target == "mean":
            y_base = np.interp(xdata, self.x_dense, self.original_y[line])
            delta = event.ydata - y_base
        else:
            y_curve = np.interp(xdata, self.x_dense, self.current_y[line])
            std_base = np.interp(xdata, self.x_dense, self.original_std[line])
            delta = event.ydata - y_curve - std_base
            delta = max(-std_base + 0.02, delta)

        controls[line][idx] = (xdata, delta)

        self.apply_edits(line)
        self.redraw_control_points()

    def on_release(self, event):
        self.active = None

    def save_plot(self, event=None):
        self.save_dir.mkdir(parents=True, exist_ok=True)

        hidden_artists = []

        for artist in self.mean_cp_artists + self.std_cp_artists:
            hidden_artists.append(artist)
            artist.set_visible(False)

        for ax in self.fig.axes:
            if ax is not self.ax:
                hidden_artists.append(ax)
                ax.set_visible(False)

        legend = self.ax.get_legend()
        if legend is not None:
            legend.set_visible(False)

        png_path = self.save_dir / "dormant_edited.png"
        pdf_path = self.save_dir / "dormant_edited.pdf"
        svg_path = self.save_dir / "dormant_edited.svg"
        csv_path = self.save_dir / "all_curves_combined.csv"

        self.fig.savefig(png_path, dpi=300, bbox_inches="tight")
        self.fig.savefig(pdf_path, bbox_inches="tight")
        self.fig.savefig(svg_path, bbox_inches="tight")

        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("curve_name,number_of_classes,rank,std\n")

            for line in self.lines:
                name = self.line_info[line]
                x = np.asarray(line.get_xdata(), dtype=float)
                y = np.asarray(line.get_ydata(), dtype=float)
                std = np.asarray(self.current_std[line], dtype=float)

                for xi, yi, si in zip(x, y, std):
                    f.write(f"{name},{xi},{yi},{si}\n")

        if legend is not None:
            legend.set_visible(True)

        for artist in hidden_artists:
            artist.set_visible(True)

        self.fig.canvas.draw_idle()

        print("Saved clean plot and CSV files to:")
        print(self.save_dir.resolve())

    def re_edit_from_output(self, event=None):
        csv_path = self.save_dir / "all_curves_combined.csv"

        if not csv_path.exists():
            print(f"No saved file found at: {csv_path.resolve()}")
            return

        data = np.genfromtxt(
            csv_path,
            delimiter=",",
            names=True,
            dtype=None,
            encoding="utf-8",
        )

        if data.size == 0:
            print("Saved CSV is empty.")
            return

        self.save_state()

        saved = {}

        for row in np.atleast_1d(data):
            name = row["curve_name"]

            if name not in saved:
                saved[name] = {
                    "x": [],
                    "y": [],
                    "std": [],
                }

            saved[name]["x"].append(float(row["number_of_classes"]))
            saved[name]["y"].append(float(row["rank"]))
            saved[name]["std"].append(float(row["std"]))

        for line in self.lines:
            name = self.line_info[line]

            if name not in saved:
                print(f"Missing saved curve: {name}")
                continue

            x_saved = np.asarray(saved[name]["x"], dtype=float)
            y_saved = np.asarray(saved[name]["y"], dtype=float)
            s_saved = np.asarray(saved[name]["std"], dtype=float)

            line.set_xdata(x_saved)
            line.set_ydata(y_saved)

            self.original_y[line] = y_saved.copy()
            self.current_y[line] = y_saved.copy()
            self.original_std[line] = s_saved.copy()
            self.current_std[line] = s_saved.copy()

            self.mean_cps[line] = [
                (x_saved[0], 0.0),
                (x_saved[-1], 0.0),
            ]

            self.std_cps[line] = [
                (x_saved[0], 0.0),
                (x_saved[-1], 0.0),
            ]

            self.update_std_patch(line)

        self.redraw_control_points()
        self.fig.canvas.draw_idle()

        print("Loaded saved CSV for re-editing:")
        print(csv_path.resolve())


def main():
    editor = InteractiveRelativeAccuracyEditor()
    plt.show()


if __name__ == "__main__":
    main()