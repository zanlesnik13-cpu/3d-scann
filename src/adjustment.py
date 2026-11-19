"""Modul za geodetsko izravnavo oblaka točk po metodi najmanjših kvadratov (MNK) – 3. korak.

Implementira splošni Gauss-Helmertov model izravnave ``A * v + B * Delta = f``
z zaporednim (akumulativnim) seštevanjem normalnih enačb po blokih točk brez
sestavljanja velikih matrik ``A`` in ``B`` v pomnilniku.

Za numerično stabilnost pri velikih geodetskih koordinatah (npr. D96/TM) se vse
točke pred izravnavo premaknejo v lokalni sistem s težiščem oblaka, po konvergenci
pa se izračunani parametri premaknejo nazaj v izvirni geodetski koordinatni sistem.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np

DEFAULT_MAX_ITER: int = 30
DEFAULT_TOL: float = 1e-9
DEFAULT_CHUNK_SIZE: int = 5_000


@dataclass
class AdjustmentResult:
    """Enotna izhodna struktura rezultatov geodetske izravnave po MNK."""

    model_name: str
    params: dict[str, float]
    param_stds: dict[str, float]
    sigma0: float
    Q_dd: np.ndarray
    iterations: int
    converged: bool
    residuals: np.ndarray
    n_points: int
    n_unknowns: int
    degrees_of_freedom: int
    rmse: float
    min_residual: float
    max_residual: float
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Pretvori rezultat izravnave v standardni Python slovar."""
        return {
            "model_name": self.model_name,
            "params": dict(self.params),
            "param_stds": dict(self.param_stds),
            "sigma0": self.sigma0,
            "Q_dd": self.Q_dd,
            "iterations": self.iterations,
            "converged": self.converged,
            "residuals": self.residuals,
            "n_points": self.n_points,
            "n_unknowns": self.n_unknowns,
            "degrees_of_freedom": self.degrees_of_freedom,
            "rmse": self.rmse,
            "min_residual": self.min_residual,
            "max_residual": self.max_residual,
            "extra": dict(self.extra),
        }


def _prepare_local_points(points: np.ndarray, n_unknowns: int) -> tuple[np.ndarray, np.ndarray]:
    """Preveri vhodni oblak točk in ga premakne v lokalni sistem s težiščem ``centroid``."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(
            f"Pričakovana oblika oblaka točk je (N, 3), dobljena pa {pts.shape}."
        )
    if not np.all(np.isfinite(pts)):
        raise ValueError("Oblak točk vsebuje neveljavne (NaN ali Inf) koordinate.")
    if pts.shape[0] <= n_unknowns:
        raise ValueError(
            f"Premalo točk za izravnavo: n = {pts.shape[0]} mora biti strogo večje "
            f"od števila neznank u = {n_unknowns}."
        )

    centroid = np.mean(pts, axis=0, dtype=np.float64)
    pts_local = np.ascontiguousarray(pts - centroid, dtype=np.float64)
    return pts_local, centroid


def _accumulate_normal_equations(
    pts_local: np.ndarray,
    params: np.ndarray,
    eval_chunk: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Akumulativno sestavi normalno matriko ``N``, vektor ``t`` in ``ftpf`` po blokih.

    Za vsako točko ``i`` velja:
    - ``Pe_i = 1.0 / (A_i @ A_i.T)``
    - ``N += B_i.T @ (Pe_i * B_i)``
    - ``t += B_i.T @ (Pe_i * f_i)``
    - ``ftpf += f_i * Pe_i * f_i``
    """
    n_points = pts_local.shape[0]
    u = params.shape[0]

    N = np.zeros((u, u), dtype=np.float64)
    t = np.zeros((u, 1), dtype=np.float64)
    ftpf = 0.0

    for start in range(0, n_points, chunk_size):
        chunk = pts_local[start : start + chunk_size]
        A_chunk, B_chunk, f_chunk = eval_chunk(chunk, params)

        # A_i @ A_i.T za vsako točko v bloku
        aa = np.einsum("ij,ij->i", A_chunk, A_chunk)
        valid = aa > 1e-24
        if not np.all(valid):
            aa = aa[valid]
            B_chunk = B_chunk[valid]
            f_chunk = f_chunk[valid]
            if aa.size == 0:
                continue

        pe = 1.0 / aa  # Pe_i = (A_i * A_i^T)^(-1)

        # N += sum_i B_i^T * Pe_i * B_i
        weighted_B = B_chunk * pe[:, np.newaxis]
        N += B_chunk.T @ weighted_B

        # t += sum_i B_i^T * Pe_i * f_i
        t[:, 0] += weighted_B.T @ f_chunk

        # ftpf += sum_i f_i * Pe_i * f_i
        ftpf += float(np.dot(f_chunk * pe, f_chunk))

    return N, t, ftpf


def _run_ghm_adjustment(
    pts_local: np.ndarray,
    init_params: np.ndarray,
    eval_chunk: Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray, np.ndarray]],
    *,
    post_update: Callable[[np.ndarray], np.ndarray] | None = None,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, int, bool]:
    """Izvede iterativno izravnavo splošnega modela ``A * v + B * Delta = f``.

    Vrne:
        ``(params, Q_dd, sigma0, sigma_params, iterations, converged)``
    """
    n_points = pts_local.shape[0]
    params = np.asarray(init_params, dtype=np.float64).copy()
    u = params.shape[0]
    dof = n_points - u

    converged = False
    iterations = 0

    for it in range(1, max_iter + 1):
        iterations = it
        N, t, _ = _accumulate_normal_equations(
            pts_local, params, eval_chunk, chunk_size=chunk_size
        )

        try:
            Delta = np.linalg.solve(N, t)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "Normalna matrika N je singularna – preverite geometrijo oblaka "
                "ali začetne približke parametrov."
            ) from exc

        if not np.all(np.isfinite(Delta)):
            raise ValueError(
                "Izravnava je divergirala (neveljavni popravki Delta). "
                "Preverite izbiro modela in začetne približke."
            )

        params = params + Delta.ravel()
        if post_update is not None:
            params = post_update(params)

        if float(np.max(np.abs(Delta))) <= tol:
            converged = True
            break

    # Končno vrednotenje N, t in ftpf na končnih izravnanih parametrih za točno oceno natančnosti
    N_final, t_final, ftpf_final = _accumulate_normal_equations(
        pts_local, params, eval_chunk, chunk_size=chunk_size
    )
    try:
        Q_dd = np.linalg.inv(N_final)
        Delta_final = Q_dd @ t_final
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "Normalna matrika N na končnih parametrih ni obrnljiva – kofaktorske matrike Q_dd ni mogoče izračunati."
        ) from exc

    vtv = ftpf_final - float((Delta_final.T @ t_final).item())
    sigma0_sq = vtv / float(dof)
    sigma0 = float(np.sqrt(max(sigma0_sq, 0.0)))

    diag_q = np.maximum(np.diag(Q_dd), 0.0)
    sigma_params = sigma0 * np.sqrt(diag_q)

    return params, Q_dd, sigma0, sigma_params, iterations, converged


def _parse_initial_params(
    initial_params: Mapping[str, float] | Sequence[float] | None,
    keys: tuple[str, ...],
) -> np.ndarray | None:
    """Pretvori ročno podane začetne približke (slovar ali zaporedje) v ``np.ndarray``."""
    if initial_params is None:
        return None
    if isinstance(initial_params, Mapping):
        missing = [k for k in keys if k not in initial_params]
        if missing:
            raise ValueError(
                f"Manjkajoči začetni približki: {missing}. Pričakovani ključi: {list(keys)}."
            )
        return np.array([float(initial_params[k]) for k in keys], dtype=np.float64)
    arr = np.asarray(initial_params, dtype=np.float64).ravel()
    if arr.shape[0] != len(keys):
        raise ValueError(
            f"Pričakovanih je {len(keys)} začetnih približkov ({keys}), dobljenih pa {arr.shape[0]}."
        )
    return arr


# ---------------------------------------------------------------------------
# 1. KROGLA: F = (x - x0)^2 + (y - y0)^2 + (z - z0)^2 - R^2 = 0
# ---------------------------------------------------------------------------


def _initial_sphere_params(pts_local: np.ndarray, chunk_size: int = DEFAULT_CHUNK_SIZE) -> np.ndarray:
    """Izračuna začetne približke (x0, y0, z0, R) za kroglo z algebraično linearno rešitvijo."""
    ata = np.zeros((4, 4), dtype=np.float64)
    atb = np.zeros(4, dtype=np.float64)

    for start in range(0, pts_local.shape[0], chunk_size):
        chunk = pts_local[start : start + chunk_size]
        m = np.empty((chunk.shape[0], 4), dtype=np.float64)
        m[:, :3] = 2.0 * chunk
        m[:, 3] = 1.0
        b = np.einsum("ij,ij->i", chunk, chunk)
        ata += m.T @ m
        atb += m.T @ b

    try:
        sol = np.linalg.solve(ata, atb)
        x0, y0, z0, c = sol
        r_sq = c + x0 * x0 + y0 * y0 + z0 * z0
        if np.isfinite(r_sq) and r_sq > 1e-12:
            r0 = float(np.sqrt(r_sq))
            return np.array([x0, y0, z0, r0], dtype=np.float64)
    except np.linalg.LinAlgError:
        pass

    # Rezervni približek: težišče (0, 0, 0) in povprečna razdalja
    mean_r = float(np.mean(np.linalg.norm(pts_local, axis=1)))
    return np.array([0.0, 0.0, 0.0, max(mean_r, 1e-3)], dtype=np.float64)


def _eval_sphere_chunk(
    chunk: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, z0, R = params
    dx = chunk[:, 0] - x0
    dy = chunk[:, 1] - y0
    dz = chunk[:, 2] - z0

    F = dx * dx + dy * dy + dz * dz - R * R
    f = -F

    A = np.empty((chunk.shape[0], 3), dtype=np.float64)
    A[:, 0] = 2.0 * dx
    A[:, 1] = 2.0 * dy
    A[:, 2] = 2.0 * dz

    B = np.empty((chunk.shape[0], 4), dtype=np.float64)
    B[:, 0] = -2.0 * dx
    B[:, 1] = -2.0 * dy
    B[:, 2] = -2.0 * dz
    B[:, 3] = -2.0 * R
    return A, B, f


def fit_sphere(
    points: np.ndarray,
    initial_params: Mapping[str, float] | Sequence[float] | None = None,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AdjustmentResult:
    """Izravnava krogle po MNK (4 neznanke: ``x0, y0, z0, R``).

    Model: ``F = (x - x0)^2 + (y - y0)^2 + (z - z0)^2 - R^2 = 0``.
    Če so podani ročni ``initial_params`` (slovar s ključi ``x0, y0, z0, R`` ali zaporedje),
    se pričakujejo v izvirnem geodetskem koordinatnem sistemu.
    """
    pts_local, centroid = _prepare_local_points(points, n_unknowns=4)

    manual = _parse_initial_params(initial_params, ("x0", "y0", "z0", "R"))
    if manual is not None:
        init_local = np.array(
            [
                manual[0] - centroid[0],
                manual[1] - centroid[1],
                manual[2] - centroid[2],
                abs(manual[3]),
            ],
            dtype=np.float64,
        )
    else:
        init_local = _initial_sphere_params(pts_local, chunk_size=chunk_size)

    def _keep_positive_radius(p: np.ndarray) -> np.ndarray:
        p[3] = abs(p[3])
        return p

    p_loc, Q_dd, sigma0, sigma_p, iterations, converged = _run_ghm_adjustment(
        pts_local,
        init_local,
        _eval_sphere_chunk,
        post_update=_keep_positive_radius,
        max_iter=max_iter,
        tol=tol,
        chunk_size=chunk_size,
    )

    x0_g = float(p_loc[0] + centroid[0])
    y0_g = float(p_loc[1] + centroid[1])
    z0_g = float(p_loc[2] + centroid[2])
    R_g = float(abs(p_loc[3]))

    # Pravokotna (predznačena) odstopanja v metrih (izračun v lokalnem sistemu za natančnost)
    diff = pts_local - p_loc[:3]
    residuals = np.linalg.norm(diff, axis=1) - R_g

    n_points = int(pts_local.shape[0])
    return AdjustmentResult(
        model_name="Krogla",
        params={"x0": x0_g, "y0": y0_g, "z0": z0_g, "R": R_g},
        param_stds={
            "x0": float(sigma_p[0]),
            "y0": float(sigma_p[1]),
            "z0": float(sigma_p[2]),
            "R": float(sigma_p[3]),
        },
        sigma0=sigma0,
        Q_dd=Q_dd,
        iterations=iterations,
        converged=converged,
        residuals=residuals,
        n_points=n_points,
        n_unknowns=4,
        degrees_of_freedom=n_points - 4,
        rmse=float(np.sqrt(np.mean(residuals * residuals))),
        min_residual=float(np.min(residuals)),
        max_residual=float(np.max(residuals)),
        extra={"centroid": centroid.copy()},
    )


# ---------------------------------------------------------------------------
# 2. RAVNINA: fiksirana največja komponenta normale na 1.0 (3 neznanke)
# ---------------------------------------------------------------------------


def _initial_plane_normal_svd(
    pts_local: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Izračuna začetno enotsko normalo ravnine preko 3×3 kovariančne matrike (ekvivalent SVD)."""
    cov = np.zeros((3, 3), dtype=np.float64)
    for start in range(0, pts_local.shape[0], chunk_size):
        chunk = pts_local[start : start + chunk_size]
        cov += chunk.T @ chunk

    # Lastni vektor za najmanjšo lastno vrednost simetrične matrike cov
    _, eigvecs = np.linalg.eigh(cov)
    normal = eigvecs[:, 0]
    norm = float(np.linalg.norm(normal))
    if norm <= 1e-15:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return normal / norm


def fit_plane(
    points: np.ndarray,
    initial_params: Mapping[str, float] | Sequence[float] | None = None,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AdjustmentResult:
    """Izravnava ravnine po MNK (3 neznanke, brez singularnosti za navpične stene).

    Iz začetnega SVD razcepa določi največjo komponento normale (X, Y ali Z) in jo
    fiksira na ``1.0`` (npr. ``a*x + b*y + z + d_loc = 0`` za os Z oziroma ustrezna
    permutacija osi za X in Y).

    V ``params`` vrne enotsko normalo ``(nx, ny, nz)``, globalni odmik ``d`` v enačbi
    ``nx*x + ny*y + nz*z + d = 0`` ter lokalni odmik pri težišču ``d_local``.
    Ročni ``initial_params`` so lahko podani kot ``(nx, ny, nz, d)`` ali slovar s
    ključi ``nx, ny, nz, d``.
    """
    pts_local, centroid = _prepare_local_points(points, n_unknowns=3)

    if initial_params is not None:
        manual = _parse_initial_params(initial_params, ("nx", "ny", "nz", "d"))
        assert manual is not None
        n_init = manual[:3]
        n_norm = float(np.linalg.norm(n_init))
        if n_norm <= 1e-12:
            raise ValueError("Začetni vektor normale (nx, ny, nz) ne sme biti ničelni.")
        n_unit_init = n_init / n_norm
        d_global_init = manual[3] / n_norm
        d_local_unit_init = d_global_init + float(np.dot(n_unit_init, centroid))
    else:
        n_unit_init = _initial_plane_normal_svd(pts_local, chunk_size=chunk_size)
        d_local_unit_init = 0.0

    # Fiksiramo največjo komponento normale na 1.0 (prepreči singularnost pri navpičnih stenah)
    fixed_axis = int(np.argmax(np.abs(n_unit_init)))
    free_axes = tuple(ax for ax in (0, 1, 2) if ax != fixed_axis)
    axis_names = ("x", "y", "z")
    slope_names = (f"n{axis_names[free_axes[0]]}_rel", f"n{axis_names[free_axes[1]]}_rel")

    scale_init = n_unit_init[fixed_axis]
    init_local = np.array(
        [
            n_unit_init[free_axes[0]] / scale_init,
            n_unit_init[free_axes[1]] / scale_init,
            d_local_unit_init / scale_init,
        ],
        dtype=np.float64,
    )

    def _eval_plane_chunk(
        chunk: np.ndarray,
        params: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u0, u1, d_raw = params
        w = np.empty(3, dtype=np.float64)
        w[free_axes[0]] = u0
        w[free_axes[1]] = u1
        w[fixed_axis] = 1.0

        F = chunk @ w + d_raw
        f = -F

        A = np.broadcast_to(w, (chunk.shape[0], 3)).copy()

        B = np.empty((chunk.shape[0], 3), dtype=np.float64)
        B[:, 0] = chunk[:, free_axes[0]]
        B[:, 1] = chunk[:, free_axes[1]]
        B[:, 2] = 1.0
        return A, B, f

    p_loc, Q_dd, sigma0, sigma_raw, iterations, converged = _run_ghm_adjustment(
        pts_local,
        init_local,
        _eval_plane_chunk,
        max_iter=max_iter,
        tol=tol,
        chunk_size=chunk_size,
    )

    # Pretvorba v enotsko normalo (nx, ny, nz) in odmik d (v globalnem in lokalnem sistemu)
    w = np.empty(3, dtype=np.float64)
    w[free_axes[0]] = p_loc[0]
    w[free_axes[1]] = p_loc[1]
    w[fixed_axis] = 1.0
    d_raw = float(p_loc[2])

    norm_w = float(np.linalg.norm(w))
    n_unit = w / norm_w
    d_unit_local = d_raw / norm_w
    d_global = float(d_unit_local - np.dot(n_unit, centroid))

    # Prenos natančnosti (zakon prenosa varianc J * Q_dd * J^T) za (nx, ny, nz, d, d_local)
    # Neznanke MNK so (w[free_axes[0]], w[free_axes[1]], d_raw)
    J = np.zeros((5, 3), dtype=np.float64)
    for col_idx, ax in enumerate(free_axes):
        e_ax = np.zeros(3, dtype=np.float64)
        e_ax[ax] = 1.0
        dn_dw = (e_ax - n_unit[ax] * n_unit) / norm_w
        dd_loc_dw = -d_raw * n_unit[ax] / (norm_w * norm_w)
        dd_glob_dw = dd_loc_dw - float(np.dot(centroid, dn_dw))

        J[0:3, col_idx] = dn_dw
        J[3, col_idx] = dd_glob_dw
        J[4, col_idx] = dd_loc_dw

    # Odvod po d_raw (tretja neznanka)
    J[3, 2] = 1.0 / norm_w
    J[4, 2] = 1.0 / norm_w

    Q_derived = J @ Q_dd @ J.T
    sigma_derived = sigma0 * np.sqrt(np.maximum(np.diag(Q_derived), 0.0))

    # Pravokotna odstopanja v metrih (v lokalnem sistemu brez numerične izgube)
    residuals = pts_local @ n_unit + d_unit_local

    n_points = int(pts_local.shape[0])
    return AdjustmentResult(
        model_name="Ravnina",
        params={
            "nx": float(n_unit[0]),
            "ny": float(n_unit[1]),
            "nz": float(n_unit[2]),
            "d": d_global,
            "d_local": float(d_unit_local),
        },
        param_stds={
            "nx": float(sigma_derived[0]),
            "ny": float(sigma_derived[1]),
            "nz": float(sigma_derived[2]),
            "d": float(sigma_derived[3]),
            "d_local": float(sigma_derived[4]),
        },
        sigma0=sigma0,
        Q_dd=Q_dd,
        iterations=iterations,
        converged=converged,
        residuals=residuals,
        n_points=n_points,
        n_unknowns=3,
        degrees_of_freedom=n_points - 3,
        rmse=float(np.sqrt(np.mean(residuals * residuals))),
        min_residual=float(np.min(residuals)),
        max_residual=float(np.max(residuals)),
        extra={
            "fixed_axis": axis_names[fixed_axis],
            "raw_param_names": (slope_names[0], slope_names[1], "d_raw"),
            "raw_params": p_loc.copy(),
            "raw_param_stds": sigma_raw.copy(),
            "centroid": centroid.copy(),
        },
    )


# ---------------------------------------------------------------------------
# 3. POKONČNI VALJ: F = (x - x0)^2 + (y - y0)^2 - R^2 = 0
# ---------------------------------------------------------------------------


def _initial_cylinder_vertical_params(
    pts_local: np.ndarray,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Izračuna začetne približke (x0, y0, R) za pokončni valj (algebraična krožnica + rezervno težišče)."""
    ata = np.zeros((3, 3), dtype=np.float64)
    atb = np.zeros(3, dtype=np.float64)

    for start in range(0, pts_local.shape[0], chunk_size):
        xy = pts_local[start : start + chunk_size, :2]
        m = np.empty((xy.shape[0], 3), dtype=np.float64)
        m[:, :2] = 2.0 * xy
        m[:, 2] = 1.0
        b = xy[:, 0] * xy[:, 0] + xy[:, 1] * xy[:, 1]
        ata += m.T @ m
        atb += m.T @ b

    try:
        sol = np.linalg.solve(ata, atb)
        x0, y0, c = sol
        r_sq = c + x0 * x0 + y0 * y0
        if np.isfinite(r_sq) and r_sq > 1e-12:
            r0 = float(np.sqrt(r_sq))
            return np.array([x0, y0, r0], dtype=np.float64)
    except np.linalg.LinAlgError:
        pass

    mean_r = float(np.mean(np.hypot(pts_local[:, 0], pts_local[:, 1])))
    return np.array([0.0, 0.0, max(mean_r, 1e-3)], dtype=np.float64)


def _eval_cylinder_vertical_chunk(
    chunk: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, R = params
    dx = chunk[:, 0] - x0
    dy = chunk[:, 1] - y0

    F = dx * dx + dy * dy - R * R
    f = -F

    A = np.zeros((chunk.shape[0], 3), dtype=np.float64)
    A[:, 0] = 2.0 * dx
    A[:, 1] = 2.0 * dy
    # dF/dz = 0

    B = np.empty((chunk.shape[0], 3), dtype=np.float64)
    B[:, 0] = -2.0 * dx
    B[:, 1] = -2.0 * dy
    B[:, 2] = -2.0 * R
    return A, B, f


def fit_cylinder_vertical(
    points: np.ndarray,
    initial_params: Mapping[str, float] | Sequence[float] | None = None,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AdjustmentResult:
    """Izravnava pokončnega valja po MNK (3 neznanke: ``x0, y0, R``).

    Model: ``F = (x - x0)^2 + (y - y0)^2 - R^2 = 0`` (os valja vzporedna osi Z).
    """
    pts_local, centroid = _prepare_local_points(points, n_unknowns=3)

    manual = _parse_initial_params(initial_params, ("x0", "y0", "R"))
    if manual is not None:
        init_local = np.array(
            [manual[0] - centroid[0], manual[1] - centroid[1], abs(manual[2])],
            dtype=np.float64,
        )
    else:
        init_local = _initial_cylinder_vertical_params(pts_local, chunk_size=chunk_size)

    def _keep_positive_radius(p: np.ndarray) -> np.ndarray:
        p[2] = abs(p[2])
        return p

    p_loc, Q_dd, sigma0, sigma_p, iterations, converged = _run_ghm_adjustment(
        pts_local,
        init_local,
        _eval_cylinder_vertical_chunk,
        post_update=_keep_positive_radius,
        max_iter=max_iter,
        tol=tol,
        chunk_size=chunk_size,
    )

    x0_g = float(p_loc[0] + centroid[0])
    y0_g = float(p_loc[1] + centroid[1])
    R_g = float(abs(p_loc[2]))

    residuals = np.hypot(pts_local[:, 0] - p_loc[0], pts_local[:, 1] - p_loc[1]) - R_g

    n_points = int(pts_local.shape[0])
    return AdjustmentResult(
        model_name="Pokončni valj",
        params={"x0": x0_g, "y0": y0_g, "R": R_g},
        param_stds={
            "x0": float(sigma_p[0]),
            "y0": float(sigma_p[1]),
            "R": float(sigma_p[2]),
        },
        sigma0=sigma0,
        Q_dd=Q_dd,
        iterations=iterations,
        converged=converged,
        residuals=residuals,
        n_points=n_points,
        n_unknowns=3,
        degrees_of_freedom=n_points - 3,
        rmse=float(np.sqrt(np.mean(residuals * residuals))),
        min_residual=float(np.min(residuals)),
        max_residual=float(np.max(residuals)),
        extra={"centroid": centroid.copy()},
    )


# ---------------------------------------------------------------------------
# 4. POKONČNI STOŽEC: F = (x - x0)^2 + (y - y0)^2 - k^2 * (z - z_vrh)^2 = 0
# ---------------------------------------------------------------------------


def _initial_cone_vertical_params(pts_local: np.ndarray) -> np.ndarray:
    """Izračuna začetne približke (x0, y0, z_vrh, k) za pokončni stožec.

    ``(x0, y0)`` vzame iz težišča ``(0, 0)``, ``z_vrh`` in ``k = tan(alpha)`` pa
    iz linearne regresije radija v odvisnosti od ``z`` (``r(z) = m * z + b``).
    """
    x0 = 0.0
    y0 = 0.0
    r = np.hypot(pts_local[:, 0] - x0, pts_local[:, 1] - y0)
    z = pts_local[:, 2]

    z_mean = float(np.mean(z))
    r_mean = float(np.mean(r))
    dz = z - z_mean
    var_z = float(np.dot(dz, dz))

    if var_z > 1e-12:
        slope = float(np.dot(dz, r - r_mean) / var_z)
        intercept = r_mean - slope * z_mean
        if abs(slope) > 1e-4:
            k0 = abs(slope)
            z_vrh0 = -intercept / slope
            return np.array([x0, y0, z_vrh0, k0], dtype=np.float64)

    # Rezervni približek, če je naklon premajhen
    z_max = float(np.max(z))
    return np.array([x0, y0, z_max + max(r_mean, 0.1), 0.5], dtype=np.float64)


def _eval_cone_vertical_chunk(
    chunk: np.ndarray,
    params: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x0, y0, z_vrh, k = params
    dx = chunk[:, 0] - x0
    dy = chunk[:, 1] - y0
    dz = chunk[:, 2] - z_vrh
    k2 = k * k

    F = dx * dx + dy * dy - k2 * dz * dz
    f = -F

    A = np.empty((chunk.shape[0], 3), dtype=np.float64)
    A[:, 0] = 2.0 * dx
    A[:, 1] = 2.0 * dy
    A[:, 2] = -2.0 * k2 * dz

    B = np.empty((chunk.shape[0], 4), dtype=np.float64)
    B[:, 0] = -2.0 * dx
    B[:, 1] = -2.0 * dy
    B[:, 2] = 2.0 * k2 * dz
    B[:, 3] = -2.0 * k * dz * dz
    return A, B, f


def fit_cone_vertical(
    points: np.ndarray,
    initial_params: Mapping[str, float] | Sequence[float] | None = None,
    *,
    max_iter: int = DEFAULT_MAX_ITER,
    tol: float = DEFAULT_TOL,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AdjustmentResult:
    """Izravnava pokončnega stožca po MNK (4 neznanke: ``x0, y0, z_vrh, k``).

    Model: ``F = (x - x0)^2 + (y - y0)^2 - k^2 * (z - z_vrh)^2 = 0``,
    kjer je ``k = tan(alpha)`` (``alpha`` je polovični kot vrha stožca).
    """
    pts_local, centroid = _prepare_local_points(points, n_unknowns=4)

    manual = _parse_initial_params(initial_params, ("x0", "y0", "z_vrh", "k"))
    if manual is not None:
        init_local = np.array(
            [
                manual[0] - centroid[0],
                manual[1] - centroid[1],
                manual[2] - centroid[2],
                abs(manual[3]),
            ],
            dtype=np.float64,
        )
    else:
        init_local = _initial_cone_vertical_params(pts_local)

    def _keep_positive_k(p: np.ndarray) -> np.ndarray:
        p[3] = abs(p[3])
        return p

    p_loc, Q_dd, sigma0, sigma_p, iterations, converged = _run_ghm_adjustment(
        pts_local,
        init_local,
        _eval_cone_vertical_chunk,
        post_update=_keep_positive_k,
        max_iter=max_iter,
        tol=tol,
        chunk_size=chunk_size,
    )

    x0_g = float(p_loc[0] + centroid[0])
    y0_g = float(p_loc[1] + centroid[1])
    z_vrh_g = float(p_loc[2] + centroid[2])
    k_g = float(abs(p_loc[3]))

    # Točna pravokotna razdalja od stožca: (r - k * |z - z_vrh|) / sqrt(1 + k^2)
    r_horiz = np.hypot(pts_local[:, 0] - p_loc[0], pts_local[:, 1] - p_loc[1])
    h_axial = np.abs(pts_local[:, 2] - p_loc[2])
    residuals = (r_horiz - k_g * h_axial) / np.sqrt(1.0 + k_g * k_g)

    alpha_rad = float(np.arctan(k_g))
    sigma_alpha_rad = float(sigma_p[3] / (1.0 + k_g * k_g))

    n_points = int(pts_local.shape[0])
    return AdjustmentResult(
        model_name="Pokončni stožec",
        params={"x0": x0_g, "y0": y0_g, "z_vrh": z_vrh_g, "k": k_g},
        param_stds={
            "x0": float(sigma_p[0]),
            "y0": float(sigma_p[1]),
            "z_vrh": float(sigma_p[2]),
            "k": float(sigma_p[3]),
        },
        sigma0=sigma0,
        Q_dd=Q_dd,
        iterations=iterations,
        converged=converged,
        residuals=residuals,
        n_points=n_points,
        n_unknowns=4,
        degrees_of_freedom=n_points - 4,
        rmse=float(np.sqrt(np.mean(residuals * residuals))),
        min_residual=float(np.min(residuals)),
        max_residual=float(np.max(residuals)),
        extra={
            "alpha_deg": float(np.degrees(alpha_rad)),
            "sigma_alpha_deg": float(np.degrees(sigma_alpha_rad)),
            "centroid": centroid.copy(),
        },
    )
