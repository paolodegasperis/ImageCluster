from __future__ import annotations

import os
import platform
import subprocess
import threading
import tkinter as tk
import webbrowser
from pathlib import Path
from tkinter import messagebox, ttk

import launcher
from backend.dependency_check import DEPENDENCIES, OPTIONAL_DEPENDENCIES
from backend.runtime_dirs import ensure_runtime_dirs
from project_paths import get_project_root

ROOT = get_project_root()
WINDOWS_NO_CONSOLE = 0x08000000

BG = "#f3efe6"
PANEL = "#ffffff"
TEXT = "#122033"
MUTED = "#5d6876"
NAVY = "#112240"
NAVY_2 = "#18365f"
ACCENT = "#f0ad2c"
ACCENT_2 = "#4fb8b0"
LINE = "#d9d0c0"
FONT = "Segoe UI" if platform.system() == "Windows" else "Helvetica"
LOGO_PATH = ROOT / "icone" / "ImageClusetIcon.png"


def project_python() -> Path | None:
    py = launcher.venv_python()
    return py if py.exists() else None


def _module_installed(py: Path, import_name: str) -> bool:
    return launcher.module_available(py, import_name)


def installed_state(torch_variant: str) -> dict:
    py = project_python()
    deps: list[dict] = []
    optional: list[dict] = []
    if py is not None:
        deps = [
            {
                "name": dep.name,
                "installed": _module_installed(py, dep.import_name),
                "required": dep.required,
                "note": dep.note,
            }
            for dep in DEPENDENCIES
        ]
        optional = [
            {
                "name": dep.name,
                "installed": _module_installed(py, dep.import_name),
                "required": dep.required,
                "note": dep.note,
            }
            for dep in OPTIONAL_DEPENDENCIES
        ]

    missing = [item["name"] for item in deps if item["required"] and not item["installed"]]
    ok = bool(py is not None and not missing)

    return {
        "python": py is not None,
        "venv": launcher.VENV.exists(),
        "deps_ok": ok,
        "requirements_current": launcher.requirements_current(torch_variant),
        "missing": missing,
        "optional": optional,
        "python_path": str(py) if py else "",
    }


class InstallerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("ImagePlot-CLIP")
        self.minsize(1020, 700)
        self.configure(bg=BG)
        self._worker: threading.Thread | None = None
        self._selected_variant = tk.StringVar(value=self.default_variant())
        self._status_text = tk.StringVar(value="Ready")
        self._headline_text = tk.StringVar(value="Install or launch ImagePlot-CLIP")
        self._logo_photo = self._load_logo()
        self._build_ui()
        if self._logo_photo is not None:
            try:
                self.iconphoto(True, self._logo_photo)
            except tk.TclError:
                pass
        self.after(200, self.refresh_status)

    def default_variant(self) -> str:
        system = platform.system()
        if system == "Windows":
            return "cpu"
        if system == "Darwin":
            return "macos"
        return "cpu"

    def _load_logo(self) -> tk.PhotoImage | None:
        if LOGO_PATH.exists():
            try:
                return tk.PhotoImage(file=str(LOGO_PATH))
            except tk.TclError:
                return None
        return None

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("Root.TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("Body.TLabel", font=(FONT, 10), background=PANEL, foreground="#334155")
        style.configure("PanelTitle.TLabel", font=(FONT, 12, "bold"), background=PANEL, foreground=TEXT)
        style.configure("Subtle.TLabel", font=(FONT, 9), background=PANEL, foreground=MUTED)
        style.configure("HeroChip.TLabel", font=(FONT, 9, "bold"), background=NAVY_2, foreground="#ffffff", padding=(12, 6))
        style.configure("HeroNote.TLabel", font=(FONT, 10), background=NAVY, foreground="#dbe6f5")
        style.configure("HeroTitle.TLabel", font=(FONT, 24, "bold"), background=NAVY, foreground="#ffffff")
        style.configure("HeroStatus.TLabel", font=(FONT, 10, "bold"), background=NAVY, foreground="#f5d99b")
        style.configure("Secondary.TButton", font=(FONT, 10), padding=(14, 9))
        style.configure(
            "TProgressbar",
            troughcolor="#e5dbca",
            bordercolor="#e5dbca",
            background=ACCENT_2,
            lightcolor=ACCENT_2,
            darkcolor=ACCENT_2,
        )

        root = ttk.Frame(self, style="Root.TFrame", padding=22)
        root.pack(fill="both", expand=True)

        hero = tk.Frame(root, bg=NAVY, highlightbackground="#254066", highlightthickness=1)
        hero.pack(fill="x")
        hero.grid_columnconfigure(1, weight=1)

        logo_frame = tk.Frame(hero, bg=NAVY)
        logo_frame.grid(row=0, column=0, rowspan=2, padx=(20, 12), pady=18, sticky="w")
        if self._logo_photo is not None:
            tk.Label(logo_frame, image=self._logo_photo, bg=NAVY).pack()
        else:
            tk.Canvas(logo_frame, width=112, height=112, bg=NAVY, highlightthickness=0).pack()

        title_frame = tk.Frame(hero, bg=NAVY)
        title_frame.grid(row=0, column=1, padx=(0, 12), pady=(18, 4), sticky="w")
        tk.Label(title_frame, text="ImagePlot-CLIP", font=(FONT, 24, "bold"), fg="#ffffff", bg=NAVY).pack(anchor="w")
        tk.Label(
            title_frame,
            text="A guided setup window for install, repair and launch",
            font=(FONT, 10),
            fg="#d6deea",
            bg=NAVY,
        ).pack(anchor="w", pady=(4, 0))
        tk.Label(
            title_frame,
            text="Designed so first-time users never need to touch terminals or scripts.",
            font=(FONT, 10),
            fg="#c7d6ea",
            bg=NAVY,
        ).pack(anchor="w", pady=(8, 0))

        badge_frame = tk.Frame(hero, bg=NAVY)
        badge_frame.grid(row=0, column=2, padx=18, pady=18, sticky="ne")
        tk.Label(badge_frame, text="ONE-CLICK SETUP", bg=NAVY_2, fg="#ffffff", font=(FONT, 9, "bold"), padx=12, pady=6).pack(anchor="e")
        tk.Label(badge_frame, textvariable=self._status_text, bg=NAVY, fg="#f7e7bf", font=(FONT, 10, "bold")).pack(anchor="e", pady=(10, 0))
        tk.Label(badge_frame, textvariable=self._headline_text, bg=NAVY, fg="#edf1f7", font=(FONT, 9), justify="right", wraplength=220).pack(anchor="e", pady=(4, 0))

        body = ttk.Frame(root, style="Root.TFrame")
        body.pack(fill="both", expand=True, pady=(18, 0))

        left = ttk.Frame(body, style="Panel.TFrame", padding=18)
        left.pack(side="left", fill="both", expand=True, padx=(0, 10))
        right = ttk.Frame(body, style="Panel.TFrame", padding=18)
        right.pack(side="right", fill="both", expand=True, padx=(10, 0))

        ttk.Label(left, text="State", style="PanelTitle.TLabel").pack(anchor="w")
        self._state_items: dict[str, ttk.Label] = {}
        for key, label in [
            ("python", "Python"),
            ("venv", "Project environment"),
            ("deps", "Required libraries"),
            ("variant", "Selected mode"),
        ]:
            row = ttk.Frame(left, style="Panel.TFrame")
            row.pack(fill="x", pady=(10, 0))
            ttk.Label(row, text=label, style="Body.TLabel").pack(side="left")
            value = ttk.Label(row, text="...", style="Body.TLabel")
            value.pack(side="right")
            self._state_items[key] = value

        ttk.Separator(left).pack(fill="x", pady=18)
        ttk.Label(left, text="Actions", style="PanelTitle.TLabel").pack(anchor="w")

        button_row = ttk.Frame(left, style="Panel.TFrame")
        button_row.pack(fill="x", pady=(12, 0))
        self.install_button = tk.Button(
            button_row,
            text="Installa / Ripara",
            command=self.install_or_repair,
            bg=ACCENT,
            fg="#111827",
            activebackground="#e99d12",
            activeforeground="#111827",
            relief="flat",
            bd=0,
            padx=18,
            pady=12,
            font=(FONT, 10, "bold"),
        )
        self.install_button.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.launch_button = tk.Button(
            button_row,
            text="Avvia app",
            command=self.launch_app,
            bg=NAVY_2,
            fg="#ffffff",
            activebackground="#0f2642",
            activeforeground="#ffffff",
            relief="flat",
            bd=0,
            padx=18,
            pady=12,
            font=(FONT, 10, "bold"),
        )
        self.launch_button.pack(side="left", fill="x", expand=True, padx=(8, 0))

        secondary = ttk.Frame(left, style="Panel.TFrame")
        secondary.pack(fill="x", pady=(10, 0))
        ttk.Button(secondary, text="Verifica ora", style="Secondary.TButton", command=self.refresh_status).pack(side="left")
        ttk.Button(secondary, text="Apri guida", style="Secondary.TButton", command=self.open_guide).pack(side="left", padx=(8, 0))
        ttk.Button(secondary, text="Esci", style="Secondary.TButton", command=self.destroy).pack(side="right")

        self.progress = ttk.Progressbar(left, mode="indeterminate")
        self.progress.pack(fill="x", pady=(16, 0))

        ttk.Separator(left).pack(fill="x", pady=18)
        ttk.Label(left, text="Mode", style="PanelTitle.TLabel").pack(anchor="w")
        mode_frame = ttk.Frame(left, style="Panel.TFrame")
        mode_frame.pack(fill="x", pady=(12, 0))
        self._mode_hint = ttk.Label(mode_frame, text="", style="Body.TLabel")
        if platform.system() == "Windows":
            for mode, label in [("cpu", "CPU"), ("cuda", "CUDA")]:
                ttk.Radiobutton(mode_frame, text=label, value=mode, variable=self._selected_variant).pack(side="left", padx=(0, 10))
            self._mode_hint.config(text="CPU is the safest first choice.")
        elif platform.system() == "Darwin":
            ttk.Radiobutton(mode_frame, text="macOS", value="macos", variable=self._selected_variant).pack(side="left")
            self._mode_hint.config(text="macOS uses the standard PyTorch wheel and may use Apple acceleration when available.")
        else:
            ttk.Radiobutton(mode_frame, text="CPU", value="cpu", variable=self._selected_variant).pack(side="left")
            self._mode_hint.config(text="CPU is the default mode on this platform.")
        self._mode_hint.pack(anchor="w", pady=(8, 0))

        ttk.Label(right, text="Details", style="PanelTitle.TLabel").pack(anchor="w")
        self._detail_box = tk.Text(
            right,
            height=19,
            wrap="word",
            relief="flat",
            bg="#fbf7ef",
            fg=TEXT,
            highlightthickness=1,
            highlightbackground=LINE,
            font=(FONT, 10),
        )
        self._detail_box.pack(fill="both", expand=True, pady=(12, 0))
        self._detail_box.insert("1.0", "The first launch can install the Python packages required by ImagePlot-CLIP.\n")
        self._detail_box.configure(state="disabled")

        footer = ttk.Frame(root, style="Root.TFrame")
        footer.pack(fill="x", pady=(14, 0))
        ttk.Label(footer, text="ImagePlot-CLIP setup and launch assistant", style="Subtle.TLabel").pack(side="left")
        ttk.Label(footer, text="Designed for first-time users", style="Subtle.TLabel").pack(side="right")

    def open_guide(self) -> None:
        webbrowser.open((ROOT / "README.md").resolve().as_uri())

    def _set_detail(self, text: str) -> None:
        self._detail_box.configure(state="normal")
        self._detail_box.delete("1.0", "end")
        self._detail_box.insert("1.0", text)
        self._detail_box.configure(state="disabled")

    def _set_busy(self, message: str) -> None:
        self._status_text.set(message)
        self.install_button.configure(state="disabled")
        self.launch_button.configure(state="disabled")
        self.progress.start(12)

    def _set_ready(self, message: str) -> None:
        self._status_text.set(message)
        self.install_button.configure(state="normal")
        self.launch_button.configure(state="normal")
        self.progress.stop()

    def refresh_status(self) -> None:
        state = installed_state(self._selected_variant.get())
        python_label = "Ready" if state["python"] else "Not found"
        venv_label = "Ready" if state["venv"] else "Missing"
        deps_label = "Ready" if state["deps_ok"] else "Missing"
        variant_label = self._selected_variant.get().upper()
        self._state_items["python"].configure(text=python_label)
        self._state_items["venv"].configure(text=venv_label)
        self._state_items["deps"].configure(text=deps_label)
        self._state_items["variant"].configure(text=variant_label)

        detail_lines = [
            f"Project root: {ROOT}",
            f"Python environment: {'present' if state['python'] else 'not created yet'}",
            f"Virtual environment folder: {'found' if state['venv'] else 'not found'}",
            f"Required libraries: {'all installed' if state['deps_ok'] else 'some missing'}",
        ]
        if state["missing"]:
            detail_lines.append("")
            detail_lines.append("Missing libraries:")
            detail_lines.extend(f"- {name}" for name in state["missing"])
        if state["optional"]:
            detail_lines.append("")
            detail_lines.append("Optional extras:")
            for item in state["optional"]:
                detail_lines.append(f"- {item['name']}: {'installed' if item['installed'] else 'not installed'}")
        self._set_detail("\n".join(detail_lines))

        if state["python"] and state["venv"] and state["deps_ok"] and state["requirements_current"]:
            self._set_ready("Ready to launch")
            self._headline_text.set("Everything is ready. You can launch the app now.")
        else:
            self._set_ready("Needs setup")
            self._headline_text.set("Install or repair the local environment, then launch the app.")

    def _run_background(self, title: str, worker) -> None:
        if self._worker and self._worker.is_alive():
            return

        def job() -> None:
            try:
                worker()
                self.after(0, self.refresh_status)
            except Exception as exc:
                self.after(0, lambda: messagebox.showerror(title, str(exc), parent=self))
                self.after(0, self.refresh_status)

        self._set_busy(title)
        self._worker = threading.Thread(target=job, daemon=True)
        self._worker.start()

    def install_or_repair(self) -> None:
        variant = self._selected_variant.get()

        def worker() -> None:
            launcher.check_python_version()
            ensure_runtime_dirs()
            py = launcher.ensure_venv()
            launcher.install_pytorch(py, variant, force=True)
            launcher.install_core_requirements(py)
            launcher.verify_required_modules(py)
            launcher.write_stamp(variant)

        self._run_background("Installing or repairing the environment...", worker)

    def launch_app(self) -> None:
        variant = self._selected_variant.get()

        def worker() -> None:
            launcher.check_python_version()
            ensure_runtime_dirs()
            py = launcher.ensure_venv()
            if not launcher.requirements_current(variant):
                launcher.install_pytorch(py, variant)
                launcher.install_core_requirements(py)
                launcher.verify_required_modules(py)
                launcher.write_stamp(variant)

            kwargs = {"cwd": str(ROOT), "env": os.environ.copy()}
            if platform.system() == "Windows":
                kwargs["creationflags"] = WINDOWS_NO_CONSOLE
            subprocess.Popen([str(py), "run.py"], **kwargs)
            self.after(0, self.destroy)

        self._run_background("Starting the app...", worker)


def main() -> int:
    app = InstallerApp()
    app.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
