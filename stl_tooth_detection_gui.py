"""GUI 3D (PyVista) pour placer des détections dentaires sur scans STL IOS.

Objectifs:
- visualisation 3D plus précise et fluide que Matplotlib (rendu VTK/PyVista),
- initialisation automatique des dents (heuristique améliorée),
- correction manuelle par point-picking,
- export JSON (maxillaire + mandibulaire).

Installation:
    pip install numpy trimesh pyvista
"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tkinter as tk
from tkinter import filedialog, ttk


def ensure_dependencies() -> None:
    required = ("numpy", "trimesh", "pyvista")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    if missing:
        packages = " ".join(missing)
        print(
            "Dépendances manquantes pour lancer la GUI.\n"
            f"Installez-les avec: pip install {packages}\n"
            "Puis relancez: python stl_tooth_detection_gui.py"
        )
        raise SystemExit(1)


ensure_dependencies()

import numpy as np
import pyvista as pv
import trimesh


@dataclass
class JawMesh:
    name: str
    path: Path
    vertices: np.ndarray
    faces: np.ndarray


def build_pyvista_mesh(vertices: np.ndarray, faces: np.ndarray) -> pv.PolyData:
    # PyVista attend un tableau [3, i, j, k, 3, i, j, k, ...]
    f = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
    return pv.PolyData(vertices, f)


def compute_local_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vecs = vecs[:, order]
    axis_arch = vecs[:, 0]
    axis_depth = vecs[:, 1]
    axis_height = vecs[:, 2]
    return centered, axis_arch, axis_depth, axis_height


def moving_average(x: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return x
    pad = k // 2
    xpad = np.pad(x, (pad, pad), mode="edge")
    kernel = np.ones(k) / k
    return np.convolve(xpad, kernel, mode="valid")


def peak_indices(signal: np.ndarray, n_peaks: int) -> np.ndarray:
    if signal.size < 3:
        return np.array([], dtype=int)
    mids = np.arange(1, len(signal) - 1)
    mask = (signal[mids] >= signal[mids - 1]) & (signal[mids] >= signal[mids + 1])
    cand = mids[mask]
    if len(cand) == 0:
        return np.array([], dtype=int)
    scores = signal[cand]
    order = np.argsort(scores)[::-1]
    cand = cand[order]

    selected: list[int] = []
    min_sep = max(1, len(signal) // (n_peaks * 2))
    for idx in cand:
        if all(abs(idx - s) >= min_sep for s in selected):
            selected.append(int(idx))
        if len(selected) >= n_peaks:
            break
    return np.array(sorted(selected), dtype=int)


def detect_teeth_landmarks(vertices: np.ndarray, jaw: str, n_teeth: int = 16) -> np.ndarray:
    """Initialisation heuristique des landmarks dentaires sur STL IOS.

    Stratégie mise à jour pour éviter les points sur la face interne (palatin/lingual):
    - repère local PCA,
    - filtre occlusal plus strict,
    - segmentation selon quantiles sur l'axe arcade,
    - dans chaque segment: priorité à la zone occlusale + externe (radiale/buccale).
    """
    if len(vertices) < n_teeth:
        raise ValueError("Le maillage contient trop peu de points pour détecter les dents.")

    centered, axis_arch, axis_depth, axis_height = compute_local_frame(vertices)
    arch = centered @ axis_arch
    depth = centered @ axis_depth
    height = centered @ axis_height
    radial = np.linalg.norm(centered[:, :2], axis=1)

    # Filtre occlusal plus strict pour éviter palais/plancher.
    if jaw == "maxillaire":
        occ = height >= np.quantile(height, 0.72)
        h_dir = 1.0
    else:
        occ = height <= np.quantile(height, 0.28)
        h_dir = -1.0

    if occ.sum() < n_teeth * 50:
        # fallback progressif si scan incomplet.
        if jaw == "maxillaire":
            occ = height >= np.quantile(height, 0.62)
        else:
            occ = height <= np.quantile(height, 0.38)
    if occ.sum() < n_teeth * 30:
        occ = np.ones(len(vertices), dtype=bool)

    occ_arch = arch[occ]
    # Quantiles => plus robuste que min/max aux extrêmes aberrants.
    edges = np.quantile(occ_arch, np.linspace(0.0, 1.0, n_teeth + 1))

    landmarks: list[np.ndarray] = []
    for i in range(n_teeth):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (arch >= lo) & (arch <= hi if i == n_teeth - 1 else arch < hi)
        in_bin &= occ
        if in_bin.sum() < 40:
            in_bin = (arch >= lo) & (arch <= hi if i == n_teeth - 1 else arch < hi)

        if in_bin.sum() == 0:
            idx = int(np.argmin(np.abs(arch - (lo + hi) * 0.5)))
            landmarks.append(vertices[idx])
            continue

        pts = vertices[in_bin]
        h = height[in_bin] * h_dir
        r = radial[in_bin]
        d = np.abs(depth[in_bin])

        # Garde la partie la plus occlusale localement, puis prend le plus externe.
        h_cut = np.quantile(h, 0.65)
        top = h >= h_cut
        if top.sum() < 5:
            top = h >= np.quantile(h, 0.5)

        score = 0.55 * h + 0.35 * r + 0.10 * d
        score = np.where(top, score, score - 1e6)

        best = int(np.argmax(score))
        landmarks.append(pts[best])

    det = np.asarray(landmarks)
    # Ordre gauche->droite cohérent pour labels.
    proj = (det - vertices.mean(axis=0, keepdims=True)) @ axis_arch
    return det[np.argsort(proj)]


def fdi_labels(jaw: str, n_teeth: int) -> list[str]:
    if n_teeth != 16:
        return [str(i + 1) for i in range(n_teeth)]
    if jaw == "maxillaire":
        return [str(x) for x in [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]]
    return [str(x) for x in [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]]


class ToothDetectionApp:
    def __init__(self) -> None:
        self.meshes: dict[str, JawMesh | None] = {"maxillaire": None, "mandibulaire": None}
        self.detections: dict[str, np.ndarray | None] = {"maxillaire": None, "mandibulaire": None}
        self.current_jaw = "maxillaire"
        self.n_teeth = 16

        self.plotter = pv.Plotter(title="Tooth STL Detection - PyVista", window_size=(1300, 860))
        self.plotter.enable_anti_aliasing("ssaa")
        self.plotter.add_axes()
        self.plotter.set_background("#0f172a")

        self.mesh_actor: Any = None
        self.points_actor: Any = None
        self.label_actor: Any = None
        self.info_actor: Any = None

        self.root = tk.Tk()
        self.root.title("Contrôle import STL / landmarks")
        self.root.geometry("430x380")
        self.status_var = tk.StringVar(value="Chargez un STL maxillaire ou mandibulaire.")
        self.jaw_var = tk.StringVar(value=self.current_jaw)
        self.teeth_var = tk.IntVar(value=self.n_teeth)

        self._build_control_panel()
        self._add_ui()
        self._refresh_scene()

    def _add_ui(self) -> None:
        self.plotter.add_text(
            "Raccourcis: M=load max | N=load mand | J=switch jaw | D=detect | C=clear | E=export | +/- teeth | clic=ajuster",
            position="upper_left",
            font_size=10,
            color="white",
        )
        self.plotter.add_key_event("m", lambda: self.load_stl("maxillaire"))
        self.plotter.add_key_event("n", lambda: self.load_stl("mandibulaire"))
        self.plotter.add_key_event("j", self.switch_jaw)
        self.plotter.add_key_event("d", self.run_detection)
        self.plotter.add_key_event("c", self.clear_detection)
        self.plotter.add_key_event("e", self.export_json)
        self.plotter.add_key_event("plus", lambda: self.set_n_teeth(self.n_teeth + 1))
        self.plotter.add_key_event("minus", lambda: self.set_n_teeth(self.n_teeth - 1))

        self.plotter.enable_point_picking(
            callback=self._on_pick,
            use_picker=True,
            show_message="Cliquez sur la dent pour corriger le landmark le plus proche.",
            color="yellow",
            point_size=14,
            pickable_window=False,
            left_clicking=True,
        )

    def _build_control_panel(self) -> None:
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frm, text="Arcade active:").pack(anchor="w")
        jaw_box = ttk.Combobox(frm, textvariable=self.jaw_var, values=["maxillaire", "mandibulaire"], state="readonly")
        jaw_box.pack(fill=tk.X, pady=(2, 8))
        jaw_box.bind("<<ComboboxSelected>>", lambda _e: self._set_jaw_from_ui())

        row = ttk.Frame(frm)
        row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(row, text="Charger maxillaire", command=lambda: self.load_stl("maxillaire")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(row, text="Charger mandibulaire", command=lambda: self.load_stl("mandibulaire")).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        ttk.Label(frm, text="Nombre de dents:").pack(anchor="w")
        sp = ttk.Spinbox(frm, from_=8, to=32, textvariable=self.teeth_var, width=8, command=self._set_teeth_from_ui)
        sp.pack(anchor="w", pady=(2, 10))

        row2 = ttk.Frame(frm)
        row2.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(row2, text="Détecter", command=self.run_detection).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))
        ttk.Button(row2, text="Effacer", command=self.clear_detection).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=4)
        ttk.Button(row2, text="Exporter JSON", command=self.export_json).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(4, 0))

        ttk.Label(frm, textvariable=self.status_var, wraplength=390, foreground="#1f2937").pack(anchor="w", pady=(8, 0))

    def _set_jaw_from_ui(self) -> None:
        self.current_jaw = self.jaw_var.get()
        self._refresh_scene(f"Arcade active: {self.current_jaw}")

    def _set_teeth_from_ui(self) -> None:
        self.set_n_teeth(int(self.teeth_var.get()))

    def set_n_teeth(self, value: int) -> None:
        self.n_teeth = int(np.clip(value, 8, 32))
        self.teeth_var.set(self.n_teeth)
        self._refresh_scene(f"Nombre de dents = {self.n_teeth}")

    def switch_jaw(self) -> None:
        self.current_jaw = "mandibulaire" if self.current_jaw == "maxillaire" else "maxillaire"
        self.jaw_var.set(self.current_jaw)
        self._refresh_scene(f"Arcade active: {self.current_jaw}")

    def _pick_file(self, jaw: str) -> str | None:
        path = filedialog.askopenfilename(
            title=f"Sélectionner STL {jaw}",
            filetypes=[("STL files", "*.stl"), ("Tous les fichiers", "*.*")],
        )
        return path or None

    def load_stl(self, jaw: str) -> None:
        path = self._pick_file(jaw)
        if not path:
            return

        mesh = trimesh.load_mesh(path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            self._refresh_scene("Erreur: fichier STL invalide.")
            return

        self.meshes[jaw] = JawMesh(
            name=jaw,
            path=Path(path),
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
        )
        self.current_jaw = jaw
        self.jaw_var.set(jaw)
        self._refresh_scene(f"Chargé {jaw}: {Path(path).name}")

    def run_detection(self) -> None:
        mesh = self.meshes.get(self.current_jaw)
        if mesh is None:
            self._refresh_scene(f"Chargez un STL {self.current_jaw} d'abord.")
            return

        try:
            pts = detect_teeth_landmarks(mesh.vertices, self.current_jaw, self.n_teeth)
        except Exception as exc:  # noqa: BLE001
            self._refresh_scene(f"Erreur détection: {exc}")
            return

        self.detections[self.current_jaw] = pts
        self._refresh_scene(f"Détection: {len(pts)} points ({self.current_jaw}).")

    def clear_detection(self) -> None:
        self.detections[self.current_jaw] = None
        self._refresh_scene(f"Détections effacées ({self.current_jaw}).")

    def _on_pick(self, picked: Any) -> None:
        if picked is None:
            return
        mesh = self.meshes.get(self.current_jaw)
        if mesh is None:
            return

        p = np.asarray(picked)
        if p.shape != (3,):
            return

        if self.detections[self.current_jaw] is None:
            self.detections[self.current_jaw] = np.empty((0, 3), dtype=float)

        det = self.detections[self.current_jaw]
        assert det is not None

        if len(det) == 0:
            det = np.array([p], dtype=float)
        elif len(det) < self.n_teeth:
            det = np.vstack([det, p])
        else:
            idx = np.argmin(np.sum((det - p) ** 2, axis=1))
            det[idx] = p

        # Re-ordonne gauche->droite dans l'axe arcade pour garder les labels cohérents.
        centered, axis_arch, _, _ = compute_local_frame(mesh.vertices)
        proj = (det - mesh.vertices.mean(axis=0, keepdims=True)) @ axis_arch
        order = np.argsort(proj)
        self.detections[self.current_jaw] = det[order]

        self._refresh_scene("Landmark ajusté manuellement (point-picking).")

    def _save_file(self) -> str | None:
        out = filedialog.asksaveasfilename(
            title="Exporter les détections",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        return out or None

    def export_json(self) -> None:
        payload: dict[str, dict[str, Any]] = {}
        for jaw, mesh in self.meshes.items():
            if mesh is None:
                continue
            det = self.detections.get(jaw)
            payload[jaw] = {
                "stl_path": str(mesh.path),
                "labels": fdi_labels(jaw, self.n_teeth),
                "landmarks": [] if det is None else det.tolist(),
            }

        if not payload:
            self._refresh_scene("Aucun maillage chargé: export impossible.")
            return

        out = self._save_file()
        if not out:
            return

        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self._refresh_scene(f"Export JSON: {Path(out).name}")

    def _refresh_scene(self, status: str | None = None) -> None:
        self.plotter.clear_actors()
        self.plotter.add_axes()

        info = [
            f"Arcade active: {self.current_jaw}",
            f"Dents: {self.n_teeth}",
        ]
        if status:
            info.append(status)
            self.status_var.set(status)

        mesh = self.meshes.get(self.current_jaw)
        if mesh is not None:
            poly = build_pyvista_mesh(mesh.vertices, mesh.faces)
            self.mesh_actor = self.plotter.add_mesh(
                poly,
                color="#B7D5E8",
                smooth_shading=True,
                specular=0.25,
                pbr=True,
                metallic=0.05,
                roughness=0.55,
                show_edges=False,
            )

            det = self.detections.get(self.current_jaw)
            if det is not None and len(det) > 0:
                cloud = pv.PolyData(det)
                cloud["labels"] = np.array(fdi_labels(self.current_jaw, len(det)))
                self.points_actor = self.plotter.add_mesh(
                    cloud,
                    render_points_as_spheres=True,
                    point_size=18,
                    color="#FF3B30",
                )
                self.label_actor = self.plotter.add_point_labels(
                    det,
                    fdi_labels(self.current_jaw, len(det)),
                    point_size=0,
                    font_size=12,
                    text_color="white",
                    shape_opacity=0.2,
                )
            info.append(f"Sommets mesh: {len(mesh.vertices)}")
        else:
            info.append("Aucun STL chargé pour cette arcade.")

        self.plotter.add_text("\n".join(info), position="lower_left", font_size=11, color="white", name="status")
        self.plotter.add_text(
            "M/N charger STL | D détecter | C effacer | J changer arcade | E exporter | clic gauche = ajuster",
            position="upper_left",
            font_size=10,
            color="white",
            name="help",
        )
        self.plotter.render()

    def run(self) -> None:
        self.plotter.show(interactive_update=True, auto_close=False)
        while not getattr(self.plotter, "_closed", False):
            try:
                self.root.update_idletasks()
                self.root.update()
            except tk.TclError:
                break
            # Selon la version PyVista, la fenêtre VTK peut disparaître sans app_window.
            if getattr(self.plotter, "ren_win", None) is None:
                break
            self.plotter.update()
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def main() -> None:
    app = ToothDetectionApp()
    app.run()


if __name__ == "__main__":
    main()
