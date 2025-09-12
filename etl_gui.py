# -*- coding: utf-8 -*-
import sys, threading, queue, subprocess, importlib.util
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

class GuiLogger:
    def __init__(self, q): self.q=q
    def write(self,s): 
        if s: self.q.put(s)
    def flush(self): pass

def load_run_etl_from_file(py_path: Path):
    """Try to load run_etl(cat_dir, municipio, out_dir) from the given file.
       Returns (callable_or_None, module_or_None, error_message_or_None)"""
    try:
        spec = importlib.util.spec_from_file_location(py_path.stem, str(py_path))
        if spec is None or spec.loader is None:
            return (None, None, f"No pude cargar el módulo desde {py_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        fn = getattr(mod, "run_etl", None)
        if callable(fn):
            return (fn, mod, None)
        else:
            return (None, mod, "No encontré una función run_etl(cat_dir, municipio, out_dir).")
    except Exception as e:
        return (None, None, f"Error importando {py_path.name}: {e}")

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ETL CAT GUI")
        self.geometry("820x520")
        self.q = queue.Queue()
        self._worker = None
        self._orig_out = sys.stdout
        self._orig_err = sys.stderr

        frm = ttk.Frame(self, padding=12); frm.pack(fill=tk.X)

        # Script .py selector
        self.script_var = tk.StringVar()
        self._row(frm, "Script (.py):", self.script_var, browse="file")

        # CAT dir
        self.cat_var = tk.StringVar()
        self._row(frm, "Carpeta CAT:", self.cat_var, browse="dir")

        # Municipio
        self.muni_var = tk.StringVar(value="Barcelona")
        self._row(frm, "Municipio:", self.muni_var)

        # Output dir
        self.out_var = tk.StringVar()
        self._row(frm, "Carpeta salida:", self.out_var, browse="dir")

        # Buttons
        btns = ttk.Frame(self, padding=(12,0)); btns.pack(fill=tk.X)
        self.run_btn = ttk.Button(btns, text="Ejecutar", command=self.run_clicked)
        self.run_btn.pack(side=tk.RIGHT)
        ttk.Button(btns, text="Limpiar", command=lambda: self.console.delete("1.0", tk.END)).pack(side=tk.RIGHT, padx=(0,8))

        # Console
        consf = ttk.Frame(self, padding=12); consf.pack(fill=tk.BOTH, expand=True)
        self.console = tk.Text(consf, wrap="word")
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.insert(tk.END, "Listo.\n")

        self.after(60, self._drain)

    def _row(self, parent, label, var, browse=None):
        row = ttk.Frame(parent); row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text=label, width=16).pack(side=tk.LEFT)
        ttk.Entry(row, textvariable=var).pack(side=tk.LEFT, fill=tk.X, expand=True)
        if browse == "dir":
            ttk.Button(row, text="Examinar…", command=lambda: self._pick_dir(var)).pack(side=tk.LEFT, padx=6)
        elif browse == "file":
            ttk.Button(row, text="Examinar…", command=lambda: self._pick_file(var)).pack(side=tk.LEFT, padx=6)

    def _pick_dir(self, var):
        p = filedialog.askdirectory()
        if p: var.set(p)

    def _pick_file(self, var):
        p = filedialog.askopenfilename(filetypes=[("Python files","*.py")])
        if p: var.set(p)

    def run_clicked(self):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("En curso", "Ya hay un proceso ejecutándose.")
            return
        script = self.script_var.get().strip()
        cat    = self.cat_var.get().strip()
        muni   = self.muni_var.get().strip()
        outdir = self.out_var.get().strip()
        if not script or not cat or not muni or not outdir:
            messagebox.showerror("Faltan datos", "Rellena todos los campos.")
            return
        pscript = Path(script)
        if not pscript.exists():
            messagebox.showerror("Script inválido", "El archivo .py no existe.")
            return

        self.console.insert(tk.END, f"\n▶ Ejecutando\n  - Script: {script}\n  - CAT: {cat}\n  - Municipio: {muni}\n  - Salida: {outdir}\n\n")
        self.console.see(tk.END)
        self.run_btn.config(state=tk.DISABLED)

        # redirect prints
        sys.stdout = GuiLogger(self.q)
        sys.stderr = GuiLogger(self.q)

        def worker():
            exit_code = 0
            try:
                run_fn, _mod, err = load_run_etl_from_file(pscript)
                if run_fn:
                    # Call Python function directly
                    run_fn(cat, muni, outdir)
                else:
                    # Fall back to "python script.py <cat> <muni> <outdir>"
                    if err:
                        print(f"ℹ️ {err}\nIntento ejecutarlo como script con argumentos.")
                    cmd = [sys.executable, str(pscript), cat, muni, outdir]
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in proc.stdout:
                        print(line, end="")
                    exit_code = proc.wait()
            except SystemExit as e:
                exit_code = int(e.code) if isinstance(e.code, int) else 1
            except Exception as ex:
                print(f"❌ ERROR: {ex}")
                exit_code = 1
            finally:
                sys.stdout = self._orig_out
                sys.stderr = self._orig_err
                self.q.put(f"\nProceso finalizado con código {exit_code}\n")
                self.q.put("__DONE__")

        self._worker = threading.Thread(target=worker, daemon=True)
        self._worker.start()

    def _drain(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg == "__DONE__":
                    self.run_btn.config(state=tk.NORMAL)
                else:
                    self.console.insert(tk.END, msg)
                    self.console.see(tk.END)
        except queue.Empty:
            pass
        self.after(60, self._drain)

if __name__ == "__main__":
    App().mainloop()
