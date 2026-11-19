"""Modul za predobdelavo, filtriranje, segmentacijo in volumetrijo oblakov točk (2. korak).

Nadomestek za ključna orodja v programu CloudCompare (izrez, SOR filter,
DBSCAN izolacija objekta, vokselno/naključno redčenje in 2.5D izračun volumna).
Vse funkcije delujejo s tabelami ``np.ndarray`` oblike ``(N, 3)`` tipa ``np.float64``,
da se ohrani popolna natančnost velikih geodetskih koordinat.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN

DBSCAN_DIRECT_LIMIT: int = 50_000
MAX_VOLUME_GRID_CELLS: int = 25_000_000


def _ensure_points_f64(points: np.ndarray) -> np.ndarray:
    """Preveri obliko ``(N, 3)`` in zagotovi zapis v ``np.float64``."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Pričakovana oblika oblaka točk je (N, 3), dobljena pa {arr.shape}."
        )
    return arr


def _voxelize_centroids(
    points: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vektorsko razdeli točke v voksle in izračuna težišče ter število točk na voksel.

    Za ohranjanje numerične natančnosti velikih geodetskih koordinat najprej odšteje
    izhodišče (minimum), sešteje zamaknjene koordinate ter nato izhodišče prišteje nazaj.

    Vrne:
        ``(centroids, inverse, counts)``:
        - ``centroids``: ``(M, 3)`` ``np.float64`` težišča posameznih vokslov,
        - ``inverse``: ``(N,)`` indeks voksla za vsako izvorno točko,
        - ``counts``: ``(M,)`` število izvornih točk v vsakem vokslu.
    """
    origin = points.min(axis=0)
    shifted = points - origin
    voxel_indices = np.floor(shifted / voxel_size).astype(np.int64)

    _, inverse, counts = np.unique(
        voxel_indices,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )

    n_voxels = counts.shape[0]
    sum_x = np.bincount(inverse, weights=shifted[:, 0], minlength=n_voxels)
    sum_y = np.bincount(inverse, weights=shifted[:, 1], minlength=n_voxels)
    sum_z = np.bincount(inverse, weights=shifted[:, 2], minlength=n_voxels)

    centroids = np.column_stack((sum_x, sum_y, sum_z)) / counts[:, np.newaxis]
    centroids += origin
    return np.ascontiguousarray(centroids, dtype=np.float64), inverse, counts


def crop_bounding_box(
    points: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
) -> np.ndarray:
    """Izreže oblak točk znotraj podanih meja ``(x_min, x_max)``, ``(y_min, y_max)``, ``(z_min, z_max)``."""
    pts = _ensure_points_f64(points)
    if pts.shape[0] == 0:
        return np.empty((0, 3), dtype=np.float64)

    x_min, x_max = min(x_range), max(x_range)
    y_min, y_max = min(y_range), max(y_range)
    z_min, z_max = min(z_range), max(z_range)

    mask = (
        (pts[:, 0] >= x_min)
        & (pts[:, 0] <= x_max)
        & (pts[:, 1] >= y_min)
        & (pts[:, 1] <= y_max)
        & (pts[:, 2] >= z_min)
        & (pts[:, 2] <= z_max)
    )
    return np.ascontiguousarray(pts[mask], dtype=np.float64)


def statistical_outlier_removal(
    points: np.ndarray,
    k_neighbors: int = 20,
    std_ratio: float = 2.0,
) -> tuple[np.ndarray, int]:
    """Statistično odstranjevanje osamelcev (SOR filter – ekvivalent CloudCompare SOR).

    Za vsako točko z uporabo ``scipy.spatial.cKDTree`` izračuna povprečno razdaljo do
    njenih ``k_neighbors`` najbližjih sosedov ter odstrani točke, pri katerih povprečna
    razdalja presega ``mean + std_ratio * std``.

    Vrne:
        ``(filtered_points, removed_count)``
    """
    pts = _ensure_points_f64(points)
    n_points = pts.shape[0]
    if n_points <= 1 or k_neighbors < 1:
        return np.ascontiguousarray(pts.copy(), dtype=np.float64), 0

    k_eff = min(int(k_neighbors), n_points - 1)

    # Centriranje pred gradnjo KD-drevesa prepreči numerično izgubo pri velikih koordinatah
    pts_centered = pts - pts.mean(axis=0)
    tree = cKDTree(pts_centered)
    distances, _ = tree.query(pts_centered, k=k_eff + 1, workers=-1)

    # Prvi stolpec (indeks 0) je razdalja točke do same sebe (0.0)
    neighbor_dists = distances[:, 1:]
    mean_dists = neighbor_dists.mean(axis=1)

    global_mean = float(mean_dists.mean())
    global_std = float(mean_dists.std())
    threshold = global_mean + float(std_ratio) * global_std

    mask = mean_dists <= threshold
    filtered = np.ascontiguousarray(pts[mask], dtype=np.float64)
    removed_count = int(n_points - filtered.shape[0])
    return filtered, removed_count


def extract_largest_cluster(
    points: np.ndarray,
    eps: float = 0.05,
    min_samples: int = 15,
) -> np.ndarray:
    """Avtomatska izolacija največjega povezanega objekta z algoritmom DBSCAN.

    Če ima oblak več kot ``DBSCAN_DIRECT_LIMIT`` (50.000) točk, za zaščito pomnilnika
    (RAM) najprej izvede fino vokselno združevanje in na težiščih vokslov izvede
    utežen ``DBSCAN(sample_weight=counts)``, nato pa oznake skupin preslika nazaj na
    vse izvorne točke v polni ločljivosti ``np.float64``.

    Če nobena skupina ne izpolnjuje kriterija ``min_samples``, vrne prazen oblak ``(0, 3)``.
    """
    pts = _ensure_points_f64(points)
    n_points = pts.shape[0]
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float64)
    if eps <= 0:
        raise ValueError("Parameter 'eps' mora biti večji od 0.")
    if min_samples < 1:
        raise ValueError("Parameter 'min_samples' mora biti vsaj 1.")

    if n_points <= DBSCAN_DIRECT_LIMIT:
        pts_centered = pts - pts.mean(axis=0)
        labels = DBSCAN(
            eps=float(eps),
            min_samples=int(min_samples),
            algorithm="kd_tree",
            n_jobs=-1,
        ).fit_predict(pts_centered)

        valid_labels = labels[labels != -1]
        if valid_labels.size == 0:
            return np.empty((0, 3), dtype=np.float64)

        unique_labels, cluster_counts = np.unique(valid_labels, return_counts=True)
        largest_label = unique_labels[int(np.argmax(cluster_counts))]
        return np.ascontiguousarray(pts[labels == largest_label], dtype=np.float64)

    # Prostorsko in pomnilniško učinkovit pristop za oblake > 50.000 točk:
    # Utežen DBSCAN nad težišči finih vokslov (sample_weight = število točk v vokslu).
    voxel_size = max(float(eps) / 3.0, 1e-6)
    centroids, inverse, counts = _voxelize_centroids(pts, voxel_size)

    for _ in range(8):
        if centroids.shape[0] <= DBSCAN_DIRECT_LIMIT:
            break
        scale = max((centroids.shape[0] / 40_000.0) ** (1.0 / 3.0), 1.25)
        voxel_size *= scale
        centroids, inverse, counts = _voxelize_centroids(pts, voxel_size)

    centroids_centered = centroids - centroids.mean(axis=0)
    # Povečaj efektivni radij največ za pol diagonale voksla, če je moral voksel zrasti
    effective_eps = max(float(eps), float(voxel_size) * np.sqrt(3.0))
    voxel_labels = DBSCAN(
        eps=effective_eps,
        min_samples=int(min_samples),
        algorithm="kd_tree",
        n_jobs=-1,
    ).fit_predict(centroids_centered, sample_weight=counts)

    valid_mask = voxel_labels != -1
    if not np.any(valid_mask):
        return np.empty((0, 3), dtype=np.float64)

    # Izberi skupino z največjim skupnim številom izvornih točk
    valid_voxel_labels = voxel_labels[valid_mask]
    valid_voxel_counts = counts[valid_mask]
    unique_labels, inv_labels = np.unique(valid_voxel_labels, return_inverse=True)
    total_points_per_cluster = np.bincount(inv_labels, weights=valid_voxel_counts)
    largest_label = unique_labels[int(np.argmax(total_points_per_cluster))]

    point_mask = voxel_labels[inverse] == largest_label
    return np.ascontiguousarray(pts[point_mask], dtype=np.float64)


def resample_cloud(
    points: np.ndarray,
    method: str = "voxel",
    voxel_size: float = 0.005,
    target_count: int = 5000,
) -> np.ndarray:
    """Redčenje oblaka točk (ekvivalent CloudCompare ``Edit -> Octree -> Resample``).

    Podprti metodi:
    - ``"voxel"``: prostorsko enakomerno redčenje na mrežo kock velikosti ``voxel_size``
      z izračunom težišča v vsakem vokslu (brez Python zank).
    - ``"random"``: naključno redčenje na natanko ``target_count`` točk.
    """
    pts = _ensure_points_f64(points)
    n_points = pts.shape[0]
    if n_points == 0:
        return np.empty((0, 3), dtype=np.float64)

    method_norm = method.strip().lower()
    if method_norm == "voxel":
        if voxel_size <= 0:
            raise ValueError("Velikost voksla (voxel_size) mora biti večja od 0.")
        centroids, _, _ = _voxelize_centroids(pts, float(voxel_size))
        return centroids

    if method_norm == "random":
        if target_count < 1:
            raise ValueError("Ciljno število točk (target_count) mora biti vsaj 1.")
        if target_count >= n_points:
            return np.ascontiguousarray(pts.copy(), dtype=np.float64)
        rng = np.random.default_rng(42)
        indices = np.sort(rng.choice(n_points, size=int(target_count), replace=False))
        return np.ascontiguousarray(pts[indices], dtype=np.float64)

    raise ValueError(
        f"Nepodprta metoda redčenja '{method}'. Uporabite 'voxel' ali 'random'."
    )


def compute_25d_volume(
    points: np.ndarray,
    ground_z: float,
    grid_step: float = 0.01,
) -> tuple[dict[str, Any], np.ndarray]:
    """Izračuna 2.5D volumen oblaka točk glede na referenčno višino ``ground_z``.

    Ekvivalent orodju CloudCompare ``Tools -> Volume -> Compute 2.5D volume``:
    - projicira točke v smeri osi Z na 2D rastrsko mrežo s korakom ``grid_step``,
    - za vsako neprazno celico izračuna povprečno višino ``z_avg``,
    - izračuna skupni neto volumen ``V = sum((z_avg - ground_z) * grid_step^2)``,
      površino tlorisa, dodani volumen (nad ``ground_z``) in odvzeti volumen (pod ``ground_z``).

    Vrne:
        ``(results_dict, z_matrix)``, kjer je ``z_matrix`` 2D matrika povprečnih višin
        oblike ``(ny, nx)`` (prazne celice imajo vrednost ``np.nan``).
    """
    pts = _ensure_points_f64(points)
    if pts.shape[0] == 0:
        raise ValueError("Oblak točk je prazen – izračun volumna ni mogoč.")
    if grid_step <= 0:
        raise ValueError("Korak mreže (grid_step) mora biti večji od 0.")

    step = float(grid_step)
    ref_z = float(ground_z)

    x_min = float(pts[:, 0].min())
    y_min = float(pts[:, 1].min())

    ix = np.floor((pts[:, 0] - x_min) / step + 1e-7).astype(np.int64)
    iy = np.floor((pts[:, 1] - y_min) / step + 1e-7).astype(np.int64)

    nx = int(ix.max()) + 1
    ny = int(iy.max()) + 1
    total_cells = nx * ny

    if total_cells > MAX_VOLUME_GRID_CELLS:
        raise ValueError(
            f"Rastrska mreža bi imela {nx} × {ny} = {total_cells:,} celic, kar presega "
            f"varno pomnilniško omejitev ({MAX_VOLUME_GRID_CELLS:,} celic). "
            f"Povečajte korak mreže (grid_step) ali predhodno izrežite oblak točk."
        )

    flat_idx = iy * nx + ix
    counts = np.bincount(flat_idx, minlength=total_cells)
    z_sums = np.bincount(flat_idx, weights=pts[:, 2], minlength=total_cells)

    z_flat = np.full(total_cells, np.nan, dtype=np.float64)
    occupied = counts > 0
    z_flat[occupied] = z_sums[occupied] / counts[occupied]
    z_matrix = z_flat.reshape((ny, nx))

    cell_area = step * step
    valid_z = z_flat[occupied]
    dz = valid_z - ref_z

    occupied_cells = int(np.count_nonzero(occupied))
    surface_area_m2 = float(occupied_cells * cell_area)
    total_volume_m3 = float(np.sum(dz * cell_area))
    added_volume_m3 = float(np.sum(dz[dz > 0.0] * cell_area))
    removed_volume_m3 = float(np.sum(np.abs(dz[dz < 0.0]) * cell_area))

    x_coords = x_min + (np.arange(nx, dtype=np.float64) + 0.5) * step
    y_coords = y_min + (np.arange(ny, dtype=np.float64) + 0.5) * step

    results: dict[str, Any] = {
        "volume_m3": total_volume_m3,
        "volume_dm3": total_volume_m3 * 1000.0,
        "added_volume_m3": added_volume_m3,
        "added_volume_dm3": added_volume_m3 * 1000.0,
        "removed_volume_m3": removed_volume_m3,
        "removed_volume_dm3": removed_volume_m3 * 1000.0,
        "surface_area_m2": surface_area_m2,
        "occupied_cells": occupied_cells,
        "total_cells": int(total_cells),
        "grid_shape": (ny, nx),
        "grid_step": step,
        "ground_z": ref_z,
        "x_coords": x_coords,
        "y_coords": y_coords,
    }
    return results, z_matrix
