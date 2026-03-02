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
    if len(vertices) < n_teeth:
        raise ValueError("Le maillage contient trop peu de points pour détecter les dents.")

    centered, axis_arch, axis_depth, axis_height = compute_local_frame(vertices)
    arch = centered @ axis_arch
    depth = centered @ axis_depth
    height = centered @ axis_height

    # Garde surtout la zone occlusale pour limiter les faux points sur gencive.
    if jaw == "maxillaire":
        h_cut = np.quantile(height, 0.65)
        occ = height >= h_cut
    else:
        h_cut = np.quantile(height, 0.35)
        occ = height <= h_cut

    if occ.sum() < n_teeth * 20:
        occ = np.ones(len(vertices), dtype=bool)

    occ_arch = arch[occ]

    # Histogramme lissé + pics de densité pour mieux approximer les centres dentaires.
    hist, edges = np.histogram(occ_arch, bins=max(64, n_teeth * 8))
    hist_s = moving_average(hist.astype(float), 7)
    peaks = peak_indices(hist_s, n_teeth)

    if len(peaks) >= max(4, n_teeth // 2):
        centers_1d = 0.5 * (edges[peaks] + edges[peaks + 1])
        centers_1d = np.sort(centers_1d)
        # Si trop de pics, sous-échantillonne régulièrement.
        if len(centers_1d) > n_teeth:
            idx = np.linspace(0, len(centers_1d) - 1, n_teeth).round().astype(int)
            centers_1d = centers_1d[idx]
        # Si pas assez, complète par quantiles.
        elif len(centers_1d) < n_teeth:
            qfill = np.quantile(occ_arch, np.linspace(0.0, 1.0, n_teeth))
            mix = np.unique(np.concatenate([centers_1d, qfill]))
            idx = np.linspace(0, len(mix) - 1, n_teeth).round().astype(int)
            centers_1d = mix[idx]
        bin_centers = centers_1d
    else:
        bin_centers = np.quantile(occ_arch, np.linspace(0.0, 1.0, n_teeth))

    landmarks: list[np.ndarray] = []
    span = max(1e-6, (arch.max() - arch.min()) / (n_teeth * 2.5))

    for c in bin_centers:
        local = np.abs(arch - c) <= span
        if local.sum() < 50:
            local = np.abs(arch - c) <= span * 1.8
        if local.sum() == 0:
            nearest = np.argmin(np.abs(arch - c))
            landmarks.append(vertices[nearest])
            continue

        # Préfère les sommets les plus occlusaux dans la fenêtre locale.
        h_local = height[local]
        d_local = np.abs(depth[local])
        pts_local = vertices[local]

        if jaw == "maxillaire":
            h_rank = h_local
        else:
            h_rank = -h_local

        # Score: occlusal élevé + proche de la crête centrale (|depth| faible)
        score = h_rank - 0.35 * d_local
        best = np.argmax(score)
        landmarks.append(pts_local[best])

    return np.asarray(landmarks)


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

        self.plotter = pv.Plotter(title="Tooth STL Detection - PyVista", window_size=(1450, 900))
        self.plotter.enable_anti_aliasing("ssaa")
        self.plotter.add_axes()
        self.plotter.set_background("#101820")

        self.mesh_actor: Any = None
        self.points_actor: Any = None
        self.label_actor: Any = None
        self.info_actor: Any = None

        self._add_ui()
        self._refresh_scene()

    def _add_ui(self) -> None:
        self.plotter.add_text(
            "Raccourcis: M=load max | N=load mand | J=switch jaw | D=detect | C=clear | E=export | +/- teeth | P=pick",
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
            show_message="Mode précision: cliquez sur la dent pour remplacer le landmark le plus proche.",
            color="yellow",
            point_size=14,
            pickable_window=False,
            left_clicking=True,
        )

    def set_n_teeth(self, value: int) -> None:
        self.n_teeth = int(np.clip(value, 8, 32))
        self._refresh_scene(f"Nombre de dents = {self.n_teeth}")

    def switch_jaw(self) -> None:
        self.current_jaw = "mandibulaire" if self.current_jaw == "maxillaire" else "maxillaire"
        self._refresh_scene(f"Arcade active: {self.current_jaw}")

    def _pick_file(self, jaw: str) -> str | None:
        # Tkinter uniquement pour file dialog natif (pas d'UI principale Tkinter).
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title=f"Sélectionner STL {jaw}",
            filetypes=[("STL files", "*.stl"), ("Tous les fichiers", "*.*")],
        )
        root.destroy()
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
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        out = filedialog.asksaveasfilename(
            title="Exporter les détections",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        root.destroy()
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
        self.plotter.show()


def main() -> None:
    app = ToothDetectionApp()
    app.run()


if __name__ == "__main__":
    main()
