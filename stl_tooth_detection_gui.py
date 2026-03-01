"""GUI pour placer des détections de dents sur des scans STL (maxillaire/mandibulaire).

Dépendances:
    pip install numpy trimesh matplotlib

L'algorithme de détection est heuristique et sert d'initialisation:
- projection du maillage sur le plan occlusal,
- alignement par PCA,
- découpage de l'arcade en N segments,
- sélection d'un point extrême par segment.

Les points détectés peuvent ensuite être exportés en JSON.
"""

from __future__ import annotations

import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import trimesh
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


@dataclass
class JawMesh:
    name: str
    path: Path
    vertices: np.ndarray
    faces: np.ndarray


def align_xy_pca(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne les coordonnées centrées + axes principaux XY."""
    centered = vertices - vertices.mean(axis=0, keepdims=True)
    xy = centered[:, :2]
    cov = np.cov(xy, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    axis_arch = eigvecs[:, 0]
    axis_depth = eigvecs[:, 1]
    return centered, axis_arch, axis_depth


def nearest_vertex(vertices: np.ndarray, point: np.ndarray) -> np.ndarray:
    idx = np.argmin(np.sum((vertices - point) ** 2, axis=1))
    return vertices[idx]


def detect_teeth_landmarks(vertices: np.ndarray, jaw: str, n_teeth: int = 16) -> np.ndarray:
    """Détection heuristique de points de dents.

    jaw: "maxillaire" ou "mandibulaire".
    """
    if vertices.shape[0] < n_teeth:
        raise ValueError("Le maillage contient trop peu de points pour détecter les dents.")

    centered, axis_arch, _ = align_xy_pca(vertices)
    arch_coord = centered[:, :2] @ axis_arch

    q = np.linspace(0.0, 1.0, n_teeth + 1)
    edges = np.quantile(arch_coord, q)

    points = []
    for i in range(n_teeth):
        low, high = edges[i], edges[i + 1]
        if i < n_teeth - 1:
            mask = (arch_coord >= low) & (arch_coord < high)
        else:
            mask = (arch_coord >= low) & (arch_coord <= high)

        local = vertices[mask]
        if len(local) == 0:
            center_guess = vertices[np.argmin(np.abs(arch_coord - (low + high) / 2.0))]
            points.append(center_guess)
            continue

        z = local[:, 2]
        if jaw == "maxillaire":
            z_cut = np.quantile(z, 0.8)
            candidates = local[z >= z_cut]
        else:
            z_cut = np.quantile(z, 0.2)
            candidates = local[z <= z_cut]

        if len(candidates) == 0:
            candidates = local

        centroid = candidates.mean(axis=0)
        points.append(nearest_vertex(local, centroid))

    return np.asarray(points)


def fdi_labels(jaw: str, n_teeth: int) -> list[str]:
    if n_teeth != 16:
        return [str(i + 1) for i in range(n_teeth)]

    if jaw == "maxillaire":
        return [str(x) for x in [18, 17, 16, 15, 14, 13, 12, 11, 21, 22, 23, 24, 25, 26, 27, 28]]
    return [str(x) for x in [48, 47, 46, 45, 44, 43, 42, 41, 31, 32, 33, 34, 35, 36, 37, 38]]


class ToothDetectionGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Détection de dents STL (IOS)")
        self.root.geometry("1200x760")

        self.meshes: dict[str, JawMesh | None] = {"maxillaire": None, "mandibulaire": None}
        self.detections: dict[str, np.ndarray | None] = {"maxillaire": None, "mandibulaire": None}

        self.current_jaw = tk.StringVar(value="maxillaire")
        self.n_teeth = tk.IntVar(value=16)
        self.alpha = tk.DoubleVar(value=0.22)

        self._build_layout()

    def _build_layout(self) -> None:
        controls = ttk.Frame(self.root, padding=10)
        controls.pack(side=tk.LEFT, fill=tk.Y)

        ttk.Label(controls, text="Arcade affichée:").pack(anchor="w")
        ttk.Combobox(
            controls,
            textvariable=self.current_jaw,
            values=["maxillaire", "mandibulaire"],
            state="readonly",
            width=16,
        ).pack(anchor="w", pady=(2, 10))

        ttk.Button(controls, text="Charger STL maxillaire", command=lambda: self.load_stl("maxillaire")).pack(
            fill=tk.X, pady=3
        )
        ttk.Button(controls, text="Charger STL mandibulaire", command=lambda: self.load_stl("mandibulaire")).pack(
            fill=tk.X, pady=3
        )

        ttk.Separator(controls).pack(fill=tk.X, pady=12)

        ttk.Label(controls, text="Nombre de dents:").pack(anchor="w")
        ttk.Spinbox(controls, from_=8, to=32, textvariable=self.n_teeth, width=8).pack(anchor="w", pady=(2, 8))

        ttk.Label(controls, text="Opacité maillage:").pack(anchor="w")
        ttk.Scale(controls, variable=self.alpha, from_=0.05, to=0.9, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=(2, 10))

        ttk.Button(controls, text="Détecter dents (heuristique)", command=self.run_detection).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="Effacer détections", command=self.clear_detection).pack(fill=tk.X, pady=3)
        ttk.Button(controls, text="Exporter JSON", command=self.export_json).pack(fill=tk.X, pady=3)

        ttk.Separator(controls).pack(fill=tk.X, pady=12)

        self.status = tk.StringVar(value="Chargez un STL maxillaire/mandibulaire pour commencer.")
        ttk.Label(controls, textvariable=self.status, wraplength=260, foreground="#2f4f4f").pack(anchor="w")

        fig_frame = ttk.Frame(self.root, padding=8)
        fig_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(9, 7), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.current_jaw.trace_add("write", lambda *_: self.update_plot())
        self.alpha.trace_add("write", lambda *_: self.update_plot())

    def load_stl(self, jaw: str) -> None:
        file_path = filedialog.askopenfilename(
            title=f"Sélectionner STL {jaw}",
            filetypes=[("STL files", "*.stl"), ("Tous les fichiers", "*.*")],
        )
        if not file_path:
            return

        mesh = trimesh.load_mesh(file_path, force="mesh")
        if not isinstance(mesh, trimesh.Trimesh):
            messagebox.showerror("Erreur", "Impossible de lire le maillage STL.")
            return

        self.meshes[jaw] = JawMesh(
            name=jaw,
            path=Path(file_path),
            vertices=np.asarray(mesh.vertices),
            faces=np.asarray(mesh.faces),
        )
        self.status.set(f"STL {jaw} chargé: {Path(file_path).name} ({len(mesh.vertices)} sommets).")
        self.current_jaw.set(jaw)
        self.update_plot()

    def run_detection(self) -> None:
        jaw = self.current_jaw.get()
        mesh = self.meshes.get(jaw)
        if mesh is None:
            messagebox.showwarning("Aucun STL", f"Chargez d'abord un STL {jaw}.")
            return

        try:
            pts = detect_teeth_landmarks(mesh.vertices, jaw=jaw, n_teeth=int(self.n_teeth.get()))
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Erreur détection", str(exc))
            return

        self.detections[jaw] = pts
        self.status.set(f"{len(pts)} points détectés pour l'arcade {jaw}.")
        self.update_plot()

    def clear_detection(self) -> None:
        jaw = self.current_jaw.get()
        self.detections[jaw] = None
        self.status.set(f"Détections effacées pour {jaw}.")
        self.update_plot()

    def export_json(self) -> None:
        payload: dict[str, dict[str, object]] = {}
        for jaw, mesh in self.meshes.items():
            if mesh is None:
                continue
            detections = self.detections.get(jaw)
            payload[jaw] = {
                "stl_path": str(mesh.path),
                "labels": fdi_labels(jaw, int(self.n_teeth.get())),
                "landmarks": [] if detections is None else detections.tolist(),
            }

        if not payload:
            messagebox.showwarning("Export impossible", "Aucun maillage chargé.")
            return

        out = filedialog.asksaveasfilename(
            title="Exporter les détections",
            defaultextension=".json",
            filetypes=[("JSON", "*.json")],
        )
        if not out:
            return

        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        self.status.set(f"Détections exportées dans {Path(out).name}.")

    def update_plot(self) -> None:
        jaw = self.current_jaw.get()
        mesh = self.meshes.get(jaw)

        self.ax.clear()
        self.ax.set_title(f"Visualisation STL - {jaw}")
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        if mesh is None:
            self.ax.text2D(0.05, 0.95, f"Aucun STL {jaw} chargé.", transform=self.ax.transAxes)
            self.canvas.draw_idle()
            return

        tris = mesh.vertices[mesh.faces]
        coll = Poly3DCollection(tris, alpha=float(self.alpha.get()), edgecolor="none", facecolor="#86b3d1")
        self.ax.add_collection3d(coll)

        det = self.detections.get(jaw)
        if det is not None and len(det) > 0:
            self.ax.scatter(det[:, 0], det[:, 1], det[:, 2], c="crimson", s=35, depthshade=True)
            labels = fdi_labels(jaw, len(det))
            for p, label in zip(det, labels):
                self.ax.text(p[0], p[1], p[2], label, color="darkred", fontsize=8)

        mins = mesh.vertices.min(axis=0)
        maxs = mesh.vertices.max(axis=0)
        center = (mins + maxs) / 2.0
        radius = np.max(maxs - mins) / 2.0
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)

        self.canvas.draw_idle()


def main() -> None:
    root = tk.Tk()
    app = ToothDetectionGUI(root)
    app.update_plot()
    root.mainloop()


if __name__ == "__main__":
    main()
