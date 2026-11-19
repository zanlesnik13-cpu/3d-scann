"""Glavna Streamlit aplikacija za obdelavo oblakov točk (GMS Vaja 2).

Namenjena obdelavi oblakov točk, posnetih z elektronskim tahimetrom Leica MS60
in izvoženih iz programske opreme Leica Infinity.
"""

from __future__ import annotations

from datetime import datetime
import hashlib
import io
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.adjustment import (
    AdjustmentResult,
    fit_cone_vertical,
    fit_cylinder_vertical,
    fit_plane,
    fit_sphere,
)
from src.io_handler import (
    list_local_point_clouds,
    load_point_cloud,
    load_point_cloud_from_bytes,
    point_cloud_to_txt_bytes,
)
from src.processing import (
    DBSCAN_DIRECT_LIMIT,
    compute_25d_volume,
    crop_bounding_box,
    extract_largest_cluster,
    resample_cloud,
    statistical_outlier_removal,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_PLOT_POINTS = 30_000
MAX_HEATMAP_DIM = 1_200
LARGE_COORD_THRESHOLD = 10_000.0

st.set_page_config(
    page_title="GMS Vaja 2 - Obdelava oblaka točk",
    layout="wide",
)


@st.cache_data(show_spinner="Nalaganje oblaka točk iz mape data/ ...")
def load_local_cloud_cached(
    filepath_str: str,
    mtime_ns: int,
    size_bytes: int,
) -> np.ndarray:
    """Predpomnjeno branje lokalne datoteke oblaka točk."""
    _ = (mtime_ns, size_bytes)
    return load_point_cloud(filepath_str)


@st.cache_data(show_spinner="Nalaganje naložene datoteke oblaka točk ...")
def load_uploaded_cloud_cached(file_bytes: bytes, filename: str) -> np.ndarray:
    """Predpomnjeno branje oblaka točk iz naložene datoteke."""
    return load_point_cloud_from_bytes(file_bytes, filename)


@st.cache_data(show_spinner="Izvajanje predobdelave oblaka točk ...")
def run_preprocessing_pipeline_cached(
    source_id: str,
    _raw_points: np.ndarray,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    use_sor: bool,
    sor_k: int,
    sor_std: float,
    use_dbscan: bool,
    dbscan_eps: float,
    dbscan_min_samples: int,
    resample_mode: str,
    voxel_size: float,
    target_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Predpomnjeno izvajanje celotne verige predobdelave oblaka točk."""
    _ = source_id
    stats: dict[str, Any] = {
        "raw_count": int(_raw_points.shape[0]),
        "after_crop": int(_raw_points.shape[0]),
        "sor_removed": 0,
        "after_sor": int(_raw_points.shape[0]),
        "after_dbscan": int(_raw_points.shape[0]),
        "dbscan_used_voxel_mode": False,
        "after_resample": int(_raw_points.shape[0]),
    }

    current = crop_bounding_box(_raw_points, x_range, y_range, z_range)
    stats["after_crop"] = int(current.shape[0])

    if use_sor and current.shape[0] > 1:
        current, sor_removed = statistical_outlier_removal(
            current,
            k_neighbors=sor_k,
            std_ratio=sor_std,
        )
        stats["sor_removed"] = sor_removed
    stats["after_sor"] = int(current.shape[0])

    if use_dbscan and current.shape[0] > 0:
        if current.shape[0] > DBSCAN_DIRECT_LIMIT:
            stats["dbscan_used_voxel_mode"] = True
        current = extract_largest_cluster(
            current,
            eps=dbscan_eps,
            min_samples=dbscan_min_samples,
        )
    stats["after_dbscan"] = int(current.shape[0])

    if current.shape[0] > 0:
        if resample_mode == "Voxel":
            current = resample_cloud(
                current,
                method="voxel",
                voxel_size=voxel_size,
            )
        elif resample_mode == "Target count":
            current = resample_cloud(
                current,
                method="random",
                target_count=target_count,
            )
    stats["after_resample"] = int(current.shape[0])

    return current, stats


@st.cache_data(show_spinner="Izračun 2.5D volumna ...")
def compute_25d_volume_cached(
    processed_sig: tuple[Any, ...],
    _processed_points: np.ndarray,
    ground_z: float,
    grid_step: float,
) -> tuple[dict[str, Any], np.ndarray]:
    """Predpomnjen 2.5D izračun volumna nad očiščenim oblakom."""
    _ = processed_sig
    return compute_25d_volume(
        _processed_points,
        ground_z=ground_z,
        grid_step=grid_step,
    )


@st.cache_data(show_spinner="Priprava datoteke za prenos ...")
def export_txt_cached(
    processed_sig: tuple[Any, ...],
    _processed_points: np.ndarray,
) -> bytes:
    """Predpomnjena pretvorba očiščenega oblaka v .txt bajte (6 decimalnih mest)."""
    _ = processed_sig
    return point_cloud_to_txt_bytes(_processed_points)


def format_int_sl(value: int) -> str:
    """Oblikuje celo število s presledki kot ločili tisočic."""
    return f"{value:,}".replace(",", " ")


def _padded_axis_bounds(min_val: float, max_val: float) -> tuple[float, float]:
    """Zaokroži meje osi navzven na 1 mm, da začetni drsniki ne odrežejo robnih točk."""
    lo = math.floor(min_val * 1000.0) / 1000.0
    hi = math.ceil(max_val * 1000.0) / 1000.0
    if hi <= lo:
        hi = lo + 0.001
    return float(lo), float(hi)


def _sync_session_state_for_source(
    source_id: str,
    source_name: str,
    raw_points: np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Sinhronizira in ponastavi ``st.session_state``, kadar uporabnik zamenja vhodno datoteko."""
    min_xyz = raw_points.min(axis=0)
    max_xyz = raw_points.max(axis=0)
    x_bounds = _padded_axis_bounds(float(min_xyz[0]), float(max_xyz[0]))
    y_bounds = _padded_axis_bounds(float(min_xyz[1]), float(max_xyz[1]))
    z_bounds = _padded_axis_bounds(float(min_xyz[2]), float(max_xyz[2]))
    center = 0.5 * (min_xyz + max_xyz)

    if st.session_state.get("active_source_id") != source_id:
        st.session_state["active_source_id"] = source_id
        st.session_state["active_source_name"] = source_name
        st.session_state["raw_points"] = raw_points
        st.session_state["processed_points"] = raw_points

        # Ponastavitev filtrov na privzete vrednosti za novo datoteko
        st.session_state["crop_x"] = x_bounds
        st.session_state["crop_y"] = y_bounds
        st.session_state["crop_z"] = z_bounds
        st.session_state["use_sor"] = False
        st.session_state["sor_k"] = 20
        st.session_state["sor_std"] = 2.0
        st.session_state["use_dbscan"] = False
        st.session_state["dbscan_eps"] = 0.05
        st.session_state["dbscan_min_samples"] = 15
        st.session_state["resample_mode"] = "Brez redčenja"
        st.session_state["voxel_size"] = 0.005
        st.session_state["target_count"] = min(5000, int(raw_points.shape[0]))

        min_z = float(min_xyz[2])
        st.session_state["vol_ground_z"] = min_z
        st.session_state["vol_last_proc_min_z"] = min_z
        st.session_state["vol_grid_step"] = 0.01
        st.session_state["use_local_center"] = bool(
            np.max(np.abs(center)) > LARGE_COORD_THRESHOLD
        )
        st.session_state.pop("adjustment_output", None)
        st.session_state.pop("volume_results", None)
        st.session_state.pop("preprocessing_info", None)
    else:
        st.session_state["raw_points"] = raw_points
        st.session_state["active_source_name"] = source_name

    return x_bounds, y_bounds, z_bounds


def _reset_crop_callback(
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
) -> None:
    """Ponastavi drsnike grobe izbire območja (X, Y, Z) na celoten surovi oblak."""
    st.session_state["crop_x"] = x_bounds
    st.session_state["crop_y"] = y_bounds
    st.session_state["crop_z"] = z_bounds


def _set_ground_z_callback(new_z: float) -> None:
    """Nastavi referenčno višino ravnine za 2.5D volumen na podano vrednost."""
    st.session_state["vol_ground_z"] = float(new_z)
    st.session_state["vol_last_proc_min_z"] = float(new_z)


def build_point_cloud_figure(
    points: np.ndarray,
    center: np.ndarray,
    *,
    use_local_center: bool,
    marker_size: float,
) -> go.Figure:
    """Ustvari interaktivni 3D graf oblaka točk (obarvan po višini Z)."""
    x_true = points[:, 0]
    y_true = points[:, 1]
    z_true = points[:, 2]

    if use_local_center:
        x_plot = x_true - center[0]
        y_plot = y_true - center[1]
        z_plot = z_true - center[2]
        x_title = "ΔX [m]"
        y_title = "ΔY [m]"
        z_title = "ΔZ [m]"
    else:
        x_plot = x_true
        y_plot = y_true
        z_plot = z_true
        x_title = "X [m]"
        y_title = "Y [m]"
        z_title = "Z [m]"

    customdata = np.column_stack((x_true, y_true, z_true))

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x_plot,
                y=y_plot,
                z=z_plot,
                mode="markers",
                customdata=customdata,
                hovertemplate=(
                    "X: %{customdata[0]:.5f} m<br>"
                    "Y: %{customdata[1]:.5f} m<br>"
                    "Z: %{customdata[2]:.5f} m"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=marker_size,
                    color=z_true,
                    colorscale="Viridis",
                    colorbar=dict(title="Z [m]"),
                    opacity=0.85,
                ),
            )
        ]
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=620,
        scene=dict(
            aspectmode="data",
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
        ),
    )
    return fig


def build_volume_heatmap(
    z_matrix: np.ndarray,
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    *,
    ground_z: float,
    relative_height: bool,
) -> tuple[go.Figure, bool]:
    """Ustvari 2D toplotni zemljevid višin s ``plotly.express.imshow``."""
    ny, nx = z_matrix.shape
    stride = max(1, math.ceil(max(ny, nx) / MAX_HEATMAP_DIM))
    was_downsampled = stride > 1

    z_plot = z_matrix[::stride, ::stride]
    x_plot = x_coords[::stride]
    y_plot = y_coords[::stride]

    if relative_height:
        display_data = z_plot - ground_z
        color_label = "ΔZ = Z - ground_z [m]"
        color_scale = "RdBu_r"
        kwargs: dict[str, Any] = {"color_continuous_midpoint": 0.0}
    else:
        display_data = z_plot
        color_label = "Z [m]"
        color_scale = "Viridis"
        kwargs = {}

    fig = px.imshow(
        display_data,
        x=x_plot,
        y=y_plot,
        origin="lower",
        aspect="equal",
        color_continuous_scale=color_scale,
        labels={"x": "X [m]", "y": "Y [m]", "color": color_label},
        **kwargs,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        height=560,
    )
    return fig, was_downsampled


def _render_tab_preprocessing(
    source_id: str,
    source_name: str,
    raw_points: np.ndarray,
    x_bounds: tuple[float, float],
    y_bounds: tuple[float, float],
    z_bounds: tuple[float, float],
    use_local_center: bool,
    marker_size: float,
) -> tuple[np.ndarray, tuple[Any, ...]]:
    """Izriše Zavihek 1: Pregled in predobdelava (CloudCompare)."""
    st.subheader(f"Pregled in predobdelava: `{source_name}`")

    # 1. Grobi izrez (Bounding Box)
    with st.expander("1. Grobi izrez po koordinatah (Bounding Box)", expanded=True):
        c_x, c_y, c_z = st.columns(3)
        x_step = max((x_bounds[1] - x_bounds[0]) / 200.0, 0.001)
        y_step = max((y_bounds[1] - y_bounds[0]) / 200.0, 0.001)
        z_step = max((z_bounds[1] - z_bounds[0]) / 200.0, 0.001)

        with c_x:
            x_range = st.slider(
                "Območje X [m]",
                min_value=x_bounds[0],
                max_value=x_bounds[1],
                step=x_step,
                format="%.3f",
                key="crop_x",
            )
        with c_y:
            y_range = st.slider(
                "Območje Y [m]",
                min_value=y_bounds[0],
                max_value=y_bounds[1],
                step=y_step,
                format="%.3f",
                key="crop_y",
            )
        with c_z:
            z_range = st.slider(
                "Območje Z [m]",
                min_value=z_bounds[0],
                max_value=z_bounds[1],
                step=z_step,
                format="%.3f",
                key="crop_z",
            )

        st.button(
            "Ponastavi izrez",
            on_click=_reset_crop_callback,
            args=(x_bounds, y_bounds, z_bounds),
        )

    # 2. SOR filter, DBSCAN izolacija in redčenje oblaka
    with st.expander("2. Filtriranje šuma, izolacija objekta in redčenje", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            st.markdown("**SOR filter (odstranjevanje osamelcev)**")
            use_sor = st.checkbox(
                "Vklopi SOR filter",
                key="use_sor",
                help="Odstrani točke s povprečno razdaljo do k sosedov nad (mean + std_ratio * std).",
            )
            sor_k = int(
                st.number_input(
                    "Število sosedov (k)",
                    min_value=2,
                    max_value=200,
                    step=1,
                    key="sor_k",
                    disabled=not use_sor,
                )
            )
            sor_std = float(
                st.number_input(
                    "Razmerje standardnega odklona (std_ratio)",
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    format="%.2f",
                    key="sor_std",
                    disabled=not use_sor,
                )
            )

        with f_col2:
            st.markdown("**DBSCAN izolacija največjega objekta**")
            use_dbscan = st.checkbox(
                "Izoliraj največji povezan objekt",
                key="use_dbscan",
                help="Z algoritmom DBSCAN samodejno obdrži največjo povezano skupino točk.",
            )
            dbscan_eps = float(
                st.number_input(
                    "Radij soseščine eps [m]",
                    min_value=0.001,
                    max_value=5.0,
                    step=0.005,
                    format="%.3f",
                    key="dbscan_eps",
                    disabled=not use_dbscan,
                )
            )
            dbscan_min_samples = int(
                st.number_input(
                    "Min. št. točk v soseščini (min_samples)",
                    min_value=2,
                    max_value=500,
                    step=1,
                    key="dbscan_min_samples",
                    disabled=not use_dbscan,
                )
            )

        with f_col3:
            st.markdown("**Redčenje oblaka (Resampling)**")
            resample_mode = st.selectbox(
                "Metoda redčenja",
                options=("Brez redčenja", "Voxel", "Target count"),
                key="resample_mode",
            )
            voxel_size = float(
                st.number_input(
                    "Velikost voksla [m]",
                    min_value=0.0005,
                    max_value=2.0,
                    step=0.001,
                    format="%.4f",
                    key="voxel_size",
                    disabled=resample_mode != "Voxel",
                )
            )
            max_target = max(int(raw_points.shape[0]), 1)
            if st.session_state.get("target_count", 1) > max_target:
                st.session_state["target_count"] = max_target
            target_count = int(
                st.number_input(
                    "Ciljno število točk",
                    min_value=1,
                    max_value=max_target,
                    step=500,
                    key="target_count",
                    disabled=resample_mode != "Target count",
                )
            )

    # Izvedi celoten cevovod obdelave (predpomnjeno)
    processed_points, stats = run_preprocessing_pipeline_cached(
        source_id=source_id,
        _raw_points=raw_points,
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        use_sor=use_sor,
        sor_k=sor_k,
        sor_std=sor_std,
        use_dbscan=use_dbscan,
        dbscan_eps=dbscan_eps,
        dbscan_min_samples=dbscan_min_samples,
        resample_mode=resample_mode,
        voxel_size=voxel_size,
        target_count=target_count,
    )

    # Shranimo očiščen oblak in povzetek filtrov v session_state za naslednje zavihke in poročilo
    st.session_state["processed_points"] = processed_points
    st.session_state["preprocessing_info"] = {
        "raw_count": stats["raw_count"],
        "processed_count": stats["after_resample"],
        "x_range": x_range,
        "y_range": y_range,
        "z_range": z_range,
        "crop_removed": stats["raw_count"] - stats["after_crop"],
        "use_sor": use_sor,
        "sor_k": sor_k,
        "sor_std": sor_std,
        "sor_removed": stats["sor_removed"],
        "use_dbscan": use_dbscan,
        "dbscan_eps": dbscan_eps,
        "dbscan_min_samples": dbscan_min_samples,
        "dbscan_removed": stats["after_sor"] - stats["after_dbscan"],
        "resample_mode": resample_mode,
        "voxel_size": voxel_size,
        "target_count": target_count,
    }

    processed_sig = (
        source_id,
        x_range,
        y_range,
        z_range,
        use_sor,
        sor_k,
        sor_std,
        use_dbscan,
        dbscan_eps,
        dbscan_min_samples,
        resample_mode,
        voxel_size,
        target_count,
    )

    if stats["dbscan_used_voxel_mode"]:
        st.info(
            f"Vhodni oblak za DBSCAN je vseboval več kot {format_int_sl(DBSCAN_DIRECT_LIMIT)} točk, "
            "zato je bil samodejno uporabljen prostorsko učinkovit utežen vokselni DBSCAN "
            "za varno porabo pomnilnika (RAM) na Streamlit Cloud."
        )

    raw_count = stats["raw_count"]
    proc_count = stats["after_resample"]
    diff_count = proc_count - raw_count
    diff_pct = (100.0 * diff_count / raw_count) if raw_count > 0 else 0.0

    # Primerjava števila točk pred in po obdelavi + geometrijske metrike
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    m_col1.metric("Točke pred obdelavo", format_int_sl(raw_count))
    m_col2.metric(
        "Točke po obdelavi",
        format_int_sl(proc_count),
        delta=f"{format_int_sl(diff_count)} ({diff_pct:+.1f} %)",
        delta_color="off",
    )

    if proc_count == 0:
        m_col3.metric("Dimenzija dX", "–")
        m_col4.metric("Dimenzija dY", "–")
        m_col5.metric("Dimenzija dZ", "–")
        st.warning(
            "Po trenutnih nastavitvah filtrov ni ostala nobena točka. "
            "Razširite meje izreza ali povečajte parameter `eps` pri DBSCAN."
        )
        return processed_points, processed_sig

    min_xyz = processed_points.min(axis=0)
    max_xyz = processed_points.max(axis=0)
    dims = max_xyz - min_xyz
    center = 0.5 * (min_xyz + max_xyz)

    m_col3.metric("Dimenzija dX", f"{dims[0]:.4f} m")
    m_col4.metric("Dimenzija dY", f"{dims[1]:.4f} m")
    m_col5.metric("Dimenzija dZ", f"{dims[2]:.4f} m")

    c_col1, c_col2, c_col3, c_col4 = st.columns([1, 1, 1, 1])
    c_col1.metric("Središče X (Box center)", f"{center[0]:.4f} m")
    c_col2.metric("Središče Y (Box center)", f"{center[1]:.4f} m")
    c_col3.metric("Središče Z (Box center)", f"{center[2]:.4f} m")

    # Gumb za prenos očiščenega oblaka v .txt (6 decimalnih mest)
    export_name = f"{Path(source_name).stem}_ociscen.txt"
    txt_bytes = export_txt_cached(processed_sig, processed_points)
    with c_col4:
        st.write("")
        st.download_button(
            label="Prenesi očiščen oblak (.txt)",
            data=txt_bytes,
            file_name=export_name,
            mime="text/plain",
            width="stretch",
        )

    # Povzetek korakov in podrobnosti omejevalnega kvadra
    with st.expander("Podrobnosti obdelave in omejevalnega kvadra (float64)", expanded=False):
        step_summary = pd.DataFrame(
            {
                "Korak": [
                    "Surovi oblak",
                    "Po grobem izrezu (X, Y, Z)",
                    "Po SOR filtru",
                    "Po DBSCAN izolaciji",
                    "Po redčenju (končni oblak)",
                ],
                "Število točk": [
                    format_int_sl(stats["raw_count"]),
                    format_int_sl(stats["after_crop"]),
                    format_int_sl(stats["after_sor"]),
                    format_int_sl(stats["after_dbscan"]),
                    format_int_sl(stats["after_resample"]),
                ],
                "Status": [
                    "Vhod",
                    f"Odstranjenih: {format_int_sl(stats['raw_count'] - stats['after_crop'])}",
                    (
                        f"Aktiven (odstranjenih: {format_int_sl(stats['sor_removed'])})"
                        if use_sor
                        else "Izklopljen"
                    ),
                    (
                        f"Aktiven (odstranjenih: {format_int_sl(stats['after_sor'] - stats['after_dbscan'])})"
                        if use_dbscan
                        else "Izklopljen"
                    ),
                    resample_mode,
                ],
            }
        )
        st.dataframe(step_summary, width="stretch", hide_index=True)

        bbox_df = pd.DataFrame(
            {
                "Os": ["X", "Y", "Z"],
                "Min [m]": [f"{min_xyz[0]:.5f}", f"{min_xyz[1]:.5f}", f"{min_xyz[2]:.5f}"],
                "Max [m]": [f"{max_xyz[0]:.5f}", f"{max_xyz[1]:.5f}", f"{max_xyz[2]:.5f}"],
                "Središče [m]": [f"{center[0]:.5f}", f"{center[1]:.5f}", f"{center[2]:.5f}"],
                "Dimenzija [m]": [f"{dims[0]:.5f}", f"{dims[1]:.5f}", f"{dims[2]:.5f}"],
            }
        )
        st.dataframe(bbox_df, width="stretch", hide_index=True)
        st.caption(
            f"Shranjen tip koordinat v pomnilniku: `{processed_points.dtype}` "
            f"(oblika: `{processed_points.shape}`)."
        )

    # Interaktivni 3D izris očiščenega oblaka
    st.subheader("Interaktivni 3D prikaz očiščenega oblaka")
    if proc_count > MAX_PLOT_POINTS:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(proc_count, size=MAX_PLOT_POINTS, replace=False)
        plot_points = processed_points[sample_indices]
        st.info(
            f"Za odziven 3D izris v brskalniku je prikazan naključni podvzorec "
            f"**{format_int_sl(MAX_PLOT_POINTS)}** od skupno **{format_int_sl(proc_count)}** točk. "
            f"Za vse nadaljnje izračune se v ozadju ohranja celoten očiščen oblak (`float64`)."
        )
    else:
        plot_points = processed_points

    fig = build_point_cloud_figure(
        plot_points,
        center,
        use_local_center=use_local_center,
        marker_size=marker_size,
    )
    st.plotly_chart(fig, width="stretch")
    return processed_points, processed_sig


def _render_tab_volume(
    processed_points: np.ndarray,
    processed_sig: tuple[Any, ...],
) -> None:
    """Izriše Zavihek 2: Izračun volumna (2.5D)."""
    st.subheader("Izračun 2.5D volumna")

    if processed_points.shape[0] == 0:
        st.warning(
            "Očiščen oblak točk je prazen. V zavihku **Pregled in predobdelava** "
            "prilagodite filtre, da ostanejo veljavne točke."
        )
        return

    current_min_z = float(processed_points[:, 2].min())
    current_max_z = float(processed_points[:, 2].max())

    # Če uporabnik ni ročno spremenil ground_z, ga samodejno sinhroniziramo z novim min(Z)
    last_auto_z = st.session_state.get("vol_last_proc_min_z")
    if (
        "vol_ground_z" not in st.session_state
        or (
            last_auto_z is not None
            and abs(float(st.session_state["vol_ground_z"]) - float(last_auto_z)) < 1e-6
            and abs(current_min_z - float(last_auto_z)) >= 1e-6
        )
    ):
        st.session_state["vol_ground_z"] = current_min_z
        st.session_state["vol_last_proc_min_z"] = current_min_z

    p_col1, p_col2, p_col3, p_col4 = st.columns([2, 2, 1.5, 1.5])
    with p_col1:
        ground_z = float(
            st.number_input(
                "Referenčna višina ravnine ground_z [m]",
                step=0.005,
                format="%.4f",
                key="vol_ground_z",
                help=(
                    f"Privzeto min(Z) očiščenega oblaka ({current_min_z:.4f} m). "
                    f"Razpon Z očiščenega oblaka: [{current_min_z:.4f}, {current_max_z:.4f}] m."
                ),
            )
        )
    with p_col2:
        grid_step = float(
            st.number_input(
                "Korak rastrske mreže grid_step [m]",
                min_value=0.001,
                max_value=5.0,
                step=0.005,
                format="%.4f",
                key="vol_grid_step",
                help="Velikost kvadratne celice v XY ravnini (privzeto 0.01 m = 1 cm).",
            )
        )
    with p_col3:
        st.write("")
        st.button(
            "Nastavi na min(Z)",
            on_click=_set_ground_z_callback,
            args=(current_min_z,),
            width="stretch",
        )
    with p_col4:
        heatmap_mode = st.selectbox(
            "Prikaz toplotnega zemljevida",
            options=("Absolutna višina Z [m]", "Relativna višina ΔZ [m]"),
        )

    try:
        results, z_matrix = compute_25d_volume_cached(
            processed_sig=processed_sig,
            _processed_points=processed_points,
            ground_z=ground_z,
            grid_step=grid_step,
        )
    except ValueError as exc:
        st.session_state["volume_results"] = None
        st.warning(str(exc))
        return

    # Shranimo rezultat volumna za samodejno vključitev v končno poročilo
    st.session_state["volume_results"] = {
        k: v for k, v in results.items() if k not in ("x_coords", "y_coords")
    }

    # Glavne metrike volumna in površine
    v_col1, v_col2, v_col3, v_col4 = st.columns(4)
    v_col1.metric("Skupni volumen [m³]", f"{results['volume_m3']:.6f} m³")
    v_col2.metric("Skupni volumen [dm³ / L]", f"{results['volume_dm3']:.3f} dm³")
    v_col3.metric("Površina tlorisa [m²]", f"{results['surface_area_m2']:.4f} m²")
    v_col4.metric(
        "Neprazne celice",
        f"{format_int_sl(results['occupied_cells'])} / {format_int_sl(results['total_cells'])}",
    )

    d_col1, d_col2, d_col3 = st.columns(3)
    d_col1.metric(
        "Dodani volumen (nad ground_z)",
        f"{results['added_volume_m3']:.6f} m³",
        delta=f"{results['added_volume_dm3']:.3f} dm³",
        delta_color="off",
    )
    d_col2.metric(
        "Odvzeti volumen (pod ground_z)",
        f"{results['removed_volume_m3']:.6f} m³",
        delta=f"{results['removed_volume_dm3']:.3f} dm³",
        delta_color="off",
    )
    ny, nx = results["grid_shape"]
    d_col3.metric("Velikost rastrske mreže", f"{nx} × {ny} celic")

    st.subheader("2D toplotni zemljevid višin (Heatmap)")
    fig_heat, was_downsampled = build_volume_heatmap(
        z_matrix,
        results["x_coords"],
        results["y_coords"],
        ground_z=ground_z,
        relative_height=(heatmap_mode == "Relativna višina ΔZ [m]"),
    )
    if was_downsampled:
        st.info(
            "Toplotni zemljevid je za odziven prikaz v brskalniku samodejno pomanjšan; "
            "izračunani volumen temelji na celotni rastrski mreži brez zmanjševanja ločljivosti."
        )
    st.plotly_chart(fig_heat, width="stretch")


# ---------------------------------------------------------------------------
# Zavihek 3: Pomožne funkcije za MNK izravnavo, 3D ploskve, histogram in poročilo
# ---------------------------------------------------------------------------

ADJUSTMENT_MODELS = {
    "Krogla": fit_sphere,
    "Ravnina": fit_plane,
    "Pokončni valj": fit_cylinder_vertical,
    "Pokončni stožec": fit_cone_vertical,
}

_PARAM_META: dict[str, tuple[str, str, bool]] = {
    "x0": ("Središče / os X₀", "m", True),
    "y0": ("Središče / os Y₀", "m", True),
    "z0": ("Središče Z₀", "m", True),
    "R": ("Polmer R", "m", True),
    "nx": ("Normala nₓ", "-", False),
    "ny": ("Normala nᵧ", "-", False),
    "nz": ("Normala n_z", "-", False),
    "d": ("Globalni odmik d", "m", True),
    "d_local": ("Lokalni odmik d_loc (pri težišču)", "m", True),
    "z_vrh": ("Višina vrha Z_vrh", "m", True),
    "k": ("Naklon stožca k = tan(α)", "-", False),
}


def _get_qdd_labels(result: AdjustmentResult) -> list[str]:
    """Vrne imena vrstic/stolpcev kofaktorske matrike Q_dd za izbrani model."""
    if result.model_name == "Krogla":
        return ["x0", "y0", "z0", "R"]
    if result.model_name == "Ravnina":
        raw_names = result.extra.get("raw_param_names")
        if raw_names:
            return list(raw_names)
        return ["u1", "u2", "d_raw"]
    if result.model_name == "Pokončni valj":
        return ["x0", "y0", "R"]
    if result.model_name == "Pokončni stožec":
        return ["x0", "y0", "z_vrh", "k"]
    return [f"u{i + 1}" for i in range(result.n_unknowns)]


def _build_params_dataframe(result: AdjustmentResult) -> pd.DataFrame:
    """Sestavi pregledno tabelo izravnanih parametrov z ocenami natančnosti."""
    rows: list[dict[str, str]] = []
    for key, val in result.params.items():
        std_val = result.param_stds.get(key, float("nan"))
        label, unit, is_meter = _PARAM_META.get(key, (key, "-", False))
        rows.append(
            {
                "Parameter": f"{label} ({key})",
                "Izravnana vrednost": f"{val:.6f}",
                "Enota": unit,
                "± σ [m / enota]": f"{std_val:.6f}",
                "± σ [mm]": f"{std_val * 1000.0:.4f} mm" if is_meter else "–",
            }
        )

    if result.model_name == "Pokončni stožec" and "alpha_deg" in result.extra:
        rows.append(
            {
                "Parameter": "Polovični kot vrha α",
                "Izravnana vrednost": f"{result.extra['alpha_deg']:.4f}",
                "Enota": "°",
                "± σ [m / enota]": f"{result.extra['sigma_alpha_deg']:.4f}",
                "± σ [mm]": "–",
            }
        )

    return pd.DataFrame(rows)


def _build_fitted_surface_mesh(
    result: AdjustmentResult,
    points: np.ndarray,
    shift: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generira 3D mrežo (Xs, Ys, Zs) izravnanega geometrijskega telesa za ``go.Surface``."""
    p = result.params
    model = result.model_name

    if model == "Krogla":
        u = np.linspace(0.0, 2.0 * np.pi, 60)
        v = np.linspace(0.0, np.pi, 40)
        uu, vv = np.meshgrid(u, v)
        r = p["R"]
        xs = (p["x0"] - shift[0]) + r * np.cos(uu) * np.sin(vv)
        ys = (p["y0"] - shift[1]) + r * np.sin(uu) * np.sin(vv)
        zs = (p["z0"] - shift[2]) + r * np.cos(vv)
        return xs, ys, zs

    if model == "Ravnina":
        n_vec = np.array([p["nx"], p["ny"], p["nz"]], dtype=np.float64)
        n_vec /= np.linalg.norm(n_vec)
        # Izberi referenčni vektor, ki ni vzporeden z normalo
        ref = np.array([1.0, 0.0, 0.0]) if abs(n_vec[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        e1 = np.cross(n_vec, ref)
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n_vec, e1)

        centroid = result.extra.get("centroid", np.mean(points, axis=0))
        d_local = p.get("d_local", float(np.dot(n_vec, centroid) + p["d"]))
        # Projekcija težišča na izravnano ravnino v lokalnih (zamaknjenih) koordinatah
        p0_global = centroid - d_local * n_vec
        p0_plot = p0_global - shift

        rel_pts = points - p0_global
        s_proj = rel_pts @ e1
        t_proj = rel_pts @ e2

        s_pad = max(0.05 * float(s_proj.max() - s_proj.min()), 1e-4)
        t_pad = max(0.05 * float(t_proj.max() - t_proj.min()), 1e-4)
        s_grid = np.linspace(float(s_proj.min()) - s_pad, float(s_proj.max()) + s_pad, 20)
        t_grid = np.linspace(float(t_proj.min()) - t_pad, float(t_proj.max()) + t_pad, 20)
        ss, tt = np.meshgrid(s_grid, t_grid)

        xs = p0_plot[0] + ss * e1[0] + tt * e2[0]
        ys = p0_plot[1] + ss * e1[1] + tt * e2[1]
        zs = p0_plot[2] + ss * e1[2] + tt * e2[2]
        return xs, ys, zs

    if model == "Pokončni valj":
        theta = np.linspace(0.0, 2.0 * np.pi, 64)
        z_min = float(points[:, 2].min())
        z_max = float(points[:, 2].max())
        if z_max <= z_min:
            z_max = z_min + 0.1
        z_vals = np.linspace(z_min, z_max, 25)
        tt, zz = np.meshgrid(theta, z_vals)
        r = p["R"]
        xs = (p["x0"] - shift[0]) + r * np.cos(tt)
        ys = (p["y0"] - shift[1]) + r * np.sin(tt)
        zs = zz - shift[2]
        return xs, ys, zs

    if model == "Pokončni stožec":
        theta = np.linspace(0.0, 2.0 * np.pi, 64)
        z_min = float(points[:, 2].min())
        z_max = float(points[:, 2].max())
        if z_max <= z_min:
            z_max = z_min + 0.1
        z_vals = np.linspace(z_min, z_max, 30)
        tt, zz = np.meshgrid(theta, z_vals)
        rr = p["k"] * np.abs(zz - p["z_vrh"])
        xs = (p["x0"] - shift[0]) + rr * np.cos(tt)
        ys = (p["y0"] - shift[1]) + rr * np.sin(tt)
        zs = zz - shift[2]
        return xs, ys, zs

    raise ValueError(f"Neznan model za 3D izris ploskve: {model}")


def build_adjustment_3d_figure(
    points: np.ndarray,
    result: AdjustmentResult,
    *,
    use_local_center: bool,
    marker_size: float,
) -> tuple[go.Figure, bool]:
    """3D graf oblaka točk (obarvan po rezidualih v mm, RdBu_r) + polprosojna ploskev telesa."""
    n_points = points.shape[0]
    was_subsampled = n_points > MAX_PLOT_POINTS

    if was_subsampled:
        rng = np.random.default_rng(42)
        idx = rng.choice(n_points, size=MAX_PLOT_POINTS, replace=False)
        pts_plot = points[idx]
        res_plot_mm = result.residuals[idx] * 1000.0
    else:
        pts_plot = points
        res_plot_mm = result.residuals * 1000.0

    center = 0.5 * (points.min(axis=0) + points.max(axis=0))
    shift = center if use_local_center else np.zeros(3, dtype=np.float64)

    x_title = "ΔX [m]" if use_local_center else "X [m]"
    y_title = "ΔY [m]" if use_local_center else "Y [m]"
    z_title = "ΔZ [m]" if use_local_center else "Z [m]"

    # Simetrična meja barvne lestvice okoli 0 (RdBu_r)
    clim = float(np.percentile(np.abs(res_plot_mm), 99.5))
    if clim <= 1e-9:
        clim = max(float(np.max(np.abs(res_plot_mm))), 1e-3)

    customdata = np.column_stack((pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], res_plot_mm))

    scatter_trace = go.Scatter3d(
        x=pts_plot[:, 0] - shift[0],
        y=pts_plot[:, 1] - shift[1],
        z=pts_plot[:, 2] - shift[2],
        mode="markers",
        name="Oblak točk (odstopanja)",
        customdata=customdata,
        hovertemplate=(
            "X: %{customdata[0]:.5f} m<br>"
            "Y: %{customdata[1]:.5f} m<br>"
            "Z: %{customdata[2]:.5f} m<br>"
            "Odstopanje: %{customdata[3]:+.3f} mm"
            "<extra></extra>"
        ),
        marker=dict(
            size=marker_size,
            color=res_plot_mm,
            colorscale="RdBu_r",
            cmin=-clim,
            cmax=clim,
            cmid=0.0,
            colorbar=dict(title="Odstopanje [mm]"),
            opacity=0.9,
        ),
    )

    xs, ys, zs = _build_fitted_surface_mesh(result, points, shift)
    surface_trace = go.Surface(
        x=xs,
        y=ys,
        z=zs,
        opacity=0.35,
        showscale=False,
        colorscale=[[0.0, "rgb(31, 119, 180)"], [1.0, "rgb(31, 119, 180)"]],
        name=f"Izravnana ploskev ({result.model_name})",
        hoverinfo="skip",
    )

    fig = go.Figure(data=[scatter_trace, surface_trace])
    fig.update_layout(
        margin=dict(l=0, r=0, t=10, b=0),
        height=650,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
        scene=dict(
            aspectmode="data",
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
        ),
    )
    return fig, was_subsampled


def build_residuals_histogram(result: AdjustmentResult) -> go.Figure:
    """Ustvari histogram pravokotnih odstopanj (rezidualov) v milimetrih okoli 0."""
    res_mm = result.residuals * 1000.0
    sigma0_mm = result.sigma0 * 1000.0

    fig = px.histogram(
        x=res_mm,
        nbins=80,
        labels={"x": "Pravokotno odstopanje [mm]", "count": "Število točk"},
        color_discrete_sequence=["#2b6cb0"],
    )
    fig.update_layout(
        yaxis_title="Število točk",
        xaxis_title="Pravokotno odstopanje [mm]",
        margin=dict(l=0, r=0, t=25, b=0),
        height=380,
        bargap=0.05,
    )
    fig.add_vline(x=0.0, line_width=2, line_color="black")
    if sigma0_mm > 0:
        fig.add_vline(
            x=sigma0_mm,
            line_width=1.5,
            line_dash="dash",
            line_color="#d97706",
            annotation_text="+σ₀",
            annotation_position="top right",
        )
        fig.add_vline(
            x=-sigma0_mm,
            line_width=1.5,
            line_dash="dash",
            line_color="#d97706",
            annotation_text="-σ₀",
            annotation_position="top left",
        )
        fig.add_vline(
            x=3.0 * sigma0_mm,
            line_width=1.2,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text="+3σ₀",
            annotation_position="top right",
        )
        fig.add_vline(
            x=-3.0 * sigma0_mm,
            line_width=1.2,
            line_dash="dot",
            line_color="#dc2626",
            annotation_text="-3σ₀",
            annotation_position="top left",
        )
    return fig


def _generate_report_txt(
    *,
    source_name: str,
    prep_info: dict[str, Any] | None,
    volume_results: dict[str, Any] | None,
    adj_output: dict[str, Any],
) -> bytes:
    """Generira lepo oblikovano slovensko tekstovno poročilo o obdelavi in izravnavi."""
    result: AdjustmentResult = adj_output["result"]
    lines: list[str] = []
    sep_major = "=" * 78
    sep_minor = "-" * 78

    lines.append(sep_major)
    lines.append("GMS VAJA 2 – POROČILO O OBDELAVI IN GEODETSKI IZRAVNAVI OBLAKA TOČK")
    lines.append(sep_major)
    lines.append(f"Datum in čas izvoza:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Vhodna datoteka:           {source_name}")

    if prep_info:
        lines.append(f"Število surovih točk:      {format_int_sl(prep_info['raw_count'])}")
        lines.append(f"Število očiščenih točk:    {format_int_sl(prep_info['processed_count'])}")
    else:
        lines.append(f"Število točk v izravnavi:  {format_int_sl(result.n_points)}")

    lines.append("")
    lines.append("1. UPORABLJENI FILTRI PREDOBDELAVE")
    lines.append(sep_minor)
    if prep_info:
        if prep_info.get("cropped_externally"):
            lines.append(
                "  Grobi izrez (Bounding Box): Izveden predhodno v CloudCompare (uvožen obrezan element)"
            )
        elif "x_range" in prep_info:
            xr = prep_info["x_range"]
            yr = prep_info["y_range"]
            zr = prep_info["z_range"]
            lines.append(
                f"  Grobi izrez (Bounding Box): "
                f"X=[{xr[0]:.3f}, {xr[1]:.3f}] m, "
                f"Y=[{yr[0]:.3f}, {yr[1]:.3f}] m, "
                f"Z=[{zr[0]:.3f}, {zr[1]:.3f}] m "
                f"(odstranjenih: {format_int_sl(prep_info['crop_removed'])})"
            )
        if prep_info.get("use_sor"):
            lines.append(
                f"  SOR filter:                 AKTIVEN "
                f"(k={prep_info['sor_k']}, std_ratio={prep_info['sor_std']:.2f}, "
                f"odstranjenih: {format_int_sl(prep_info['sor_removed'])})"
            )
        else:
            lines.append("  SOR filter:                 Izklopljen")

        if prep_info.get("use_dbscan"):
            lines.append(
                f"  DBSCAN izolacija objekta:   AKTIVNA "
                f"(eps={prep_info['dbscan_eps']:.3f} m, "
                f"min_samples={prep_info['dbscan_min_samples']}, "
                f"odstranjenih: {format_int_sl(prep_info['dbscan_removed'])})"
            )
        else:
            lines.append("  DBSCAN izolacija objekta:   Izklopljena")

        rmode = prep_info.get("resample_mode", "Brez redčenja")
        if rmode == "Voxel":
            lines.append(f"  Redčenje oblaka:            Voxel (voxel_size={prep_info['voxel_size']:.4f} m)")
        elif rmode == "Target count":
            lines.append(
                f"  Redčenje oblaka:            Naključno (target_count={format_int_sl(prep_info['target_count'])})"
            )
        else:
            lines.append("  Redčenje oblaka:            Brez redčenja")
    else:
        lines.append("  Predobdelava ni bila uporabljena (uporabljen surovi oblak).")

    lines.append("")
    lines.append("2. IZRAČUN VOLUMNA (2.5D)")
    lines.append(sep_minor)
    if volume_results:
        ny, nx = volume_results["grid_shape"]
        lines.append(f"  Referenčna višina ground_z: {volume_results['ground_z']:.4f} m")
        lines.append(f"  Korak rastrske mreže:       {volume_results['grid_step']:.4f} m ({nx} × {ny} celic)")
        lines.append(
            f"  Skupni neto volumen:        {volume_results['volume_m3']:.6f} m³ "
            f"({volume_results['volume_dm3']:.3f} dm³ / L)"
        )
        lines.append(
            f"  Dodani volumen (> ground):  {volume_results['added_volume_m3']:.6f} m³ "
            f"({volume_results['added_volume_dm3']:.3f} dm³)"
        )
        lines.append(
            f"  Odvzeti volumen (< ground): {volume_results['removed_volume_m3']:.6f} m³ "
            f"({volume_results['removed_volume_dm3']:.3f} dm³)"
        )
        lines.append(f"  Površina tlorisa:           {volume_results['surface_area_m2']:.4f} m²")
        lines.append(
            f"  Neprazne celice:            "
            f"{format_int_sl(volume_results['occupied_cells'])} / "
            f"{format_int_sl(volume_results['total_cells'])}"
        )
    else:
        lines.append("  2.5D volumen ni bil izračunan.")

    lines.append("")
    lines.append("3. GEODETSKA IZRAVNAVA PO METODI NAJMANJŠIH KVADRATOV (MNK)")
    lines.append(sep_minor)
    lines.append(f"  Geometrijski model:         {result.model_name}")
    lines.append(f"  Konvergenca:                {'DA' if result.converged else 'NE (doseženo max. št. iteracij)'}")
    lines.append(f"  Število iteracij:           {result.iterations}")
    lines.append(f"  Število točk (n):           {format_int_sl(result.n_points)}")
    lines.append(f"  Število neznank (u):        {result.n_unknowns}")
    lines.append(f"  Prostostne stopnje (n - u): {format_int_sl(result.degrees_of_freedom)}")

    if adj_output.get("outlier_3sigma_enabled"):
        rejected = adj_output.get("rejected_3sigma", 0)
        first_s0 = adj_output.get("first_pass_sigma0")
        if first_s0 is not None:
            lines.append(
                f"  Izločanje > 3·σ₀:           AKTIVNO "
                f"(izločenih {format_int_sl(rejected)} točk, 1. iteracija σ₀={first_s0 * 1000.0:.3f} mm)"
            )
    else:
        lines.append("  Izločanje > 3·σ₀:           Izklopljeno")

    lines.append(
        f"  Aposteriorni std. odklon:   σ₀ = {result.sigma0:.6f} m ({result.sigma0 * 1000.0:.3f} mm)"
    )

    lines.append("")
    lines.append("  Izravnani parametri modela:")
    lines.append(f"  {'Parameter':<36} {'Vrednost':>15} {'± σ [m/en.]':>14} {'± σ [mm]':>12}")
    lines.append("  " + "-" * 76)
    for key, val in result.params.items():
        std_val = result.param_stds.get(key, float("nan"))
        label, _, is_meter = _PARAM_META.get(key, (key, "-", False))
        mm_str = f"{std_val * 1000.0:.4f}" if is_meter else "-"
        lines.append(
            f"  {f'{label} ({key})':<36} {val:>15.6f} {std_val:>14.6f} {mm_str:>12}"
        )
    if result.model_name == "Pokončni stožec" and "alpha_deg" in result.extra:
        lines.append(
            f"  {'Polovični kot vrha α [°]':<36} "
            f"{result.extra['alpha_deg']:>15.4f} "
            f"{result.extra['sigma_alpha_deg']:>14.4f} {'-':>12}"
        )

    mean_res = float(np.mean(result.residuals))
    std_res = float(np.std(result.residuals))
    lines.append("")
    lines.append("  Statistika pravokotnih odstopanj (rezidualov):")
    lines.append(f"    Min. odstopanje:          {result.min_residual:+.6f} m ({result.min_residual * 1000.0:+.3f} mm)")
    lines.append(f"    Max. odstopanje:          {result.max_residual:+.6f} m ({result.max_residual * 1000.0:+.3f} mm)")
    lines.append(f"    Povprečno odstopanje:     {mean_res:+.6f} m ({mean_res * 1000.0:+.4f} mm)")
    lines.append(f"    Std. odklon odstopanj:    {std_res:.6f} m ({std_res * 1000.0:.3f} mm)")
    lines.append(f"    RMSE:                     {result.rmse:.6f} m ({result.rmse * 1000.0:.3f} mm)")

    lines.append("")
    lines.append("  Kofaktorska matrika Q_dd = N^(-1):")
    q_labels = _get_qdd_labels(result)
    header_row = "    " + f"{'':<10}" + "".join(f"{lbl:>15}" for lbl in q_labels)
    lines.append(header_row)
    for row_lbl, row_vals in zip(q_labels, result.Q_dd):
        vals_str = "".join(f"{v:>15.6e}" for v in row_vals)
        lines.append(f"    {row_lbl:<10}{vals_str}")

    if result.model_name == "Ravnina" and "fixed_axis" in result.extra:
        lines.append(
            f"  (Opomba: pri izravnavi ravnine je bila za preprečitev singularnosti "
            f"fiksirana komponenta osi '{result.extra['fixed_axis']}' = 1.0.)"
        )

    lines.append(sep_major)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _generate_residuals_csv(points: np.ndarray, residuals: np.ndarray) -> bytes:
    """Generira CSV datoteko s stolpci ``X,Y,Z,odstopanje_m,odstopanje_mm``."""
    df = pd.DataFrame(
        {
            "X": np.round(points[:, 0], 6),
            "Y": np.round(points[:, 1], 6),
            "Z": np.round(points[:, 2], 6),
            "odstopanje_m": np.round(residuals, 6),
            "odstopanje_mm": np.round(residuals * 1000.0, 4),
        }
    )
    buf = io.StringIO()
    df.to_csv(buf, index=False, float_format="%.6f")
    return buf.getvalue().encode("utf-8")


def _render_tab_adjustment(
    source_name: str,
    processed_sig: tuple[Any, ...],
    use_local_center: bool,
    marker_size: float,
) -> None:
    """Izriše Zavihek 3: Geodetska izravnava po MNK, 3D ploskev, histogram in izvoz poročila."""
    st.subheader("Geodetska izravnava po metodi najmanjših kvadratov (MNK)")

    input_points = st.session_state.get("processed_points")
    if input_points is None:
        input_points = st.session_state.get("raw_points")

    if input_points is None or input_points.shape[0] == 0:
        st.warning(
            "Oblak točk za izravnavo je prazen. V zavihku **Pregled in predobdelava** "
            "prilagodite filtre, da ostanejo veljavne točke."
        )
        return

    st.caption(
        f"Vhodni oblak za izravnavo: **{format_int_sl(int(input_points.shape[0]))}** točk "
        f"(`{input_points.dtype}`). Splošni Gauss-Helmertov model: `A·v + B·Δ = f` "
        f"v lokalnem težiščnem sistemu."
    )

    # 1. Nastavitve izravnave
    cfg_col1, cfg_col2 = st.columns([2, 3])
    with cfg_col1:
        model_choice = st.selectbox(
            "Geometrijsko telo",
            options=tuple(ADJUSTMENT_MODELS.keys()),
            key="mnk_model_choice",
        )
    with cfg_col2:
        st.write("")
        outlier_3sigma = st.checkbox(
            "Avtomatsko izloči točke z odstopanjem > 3 * sigma0 in ponovi izravnavo",
            key="mnk_outlier_3sigma",
            help=(
                "Po prvi izravnavi odstrani vse točke z |odstopanje| > 3·σ₀ "
                "in izravnavo ponovi na očiščenem naboru."
            ),
        )

    run_clicked = st.button("Zaženi izravnavo po MNK", type="primary")

    if run_clicked:
        fit_fn = ADJUSTMENT_MODELS[model_choice]
        try:
            with st.spinner(f"Izvajanje MNK izravnave ({model_choice}) ..."):
                res_first = fit_fn(input_points)
                used_points = input_points
                final_res = res_first
                rejected_3sigma = 0
                first_pass_sigma0: float | None = res_first.sigma0

                if outlier_3sigma and res_first.sigma0 > 0.0:
                    threshold = 3.0 * res_first.sigma0
                    inlier_mask = np.abs(res_first.residuals) <= threshold
                    rejected_3sigma = int(input_points.shape[0] - np.count_nonzero(inlier_mask))
                    if rejected_3sigma > 0:
                        if int(np.count_nonzero(inlier_mask)) <= res_first.n_unknowns:
                            st.warning(
                                "Izločanje > 3·σ₀ bi pustilo premalo točk za ponovitev izravnave; "
                                "ohranjen je rezultat prve izravnave."
                            )
                            rejected_3sigma = 0
                        else:
                            used_points = np.ascontiguousarray(
                                input_points[inlier_mask], dtype=np.float64
                            )
                            final_res = fit_fn(used_points)

                st.session_state["adjustment_output"] = {
                    "result": final_res,
                    "points_used": used_points,
                    "processed_sig": processed_sig,
                    "model_choice": model_choice,
                    "outlier_3sigma_enabled": outlier_3sigma,
                    "rejected_3sigma": rejected_3sigma,
                    "first_pass_sigma0": first_pass_sigma0,
                }
        except ValueError as exc:
            st.error(f"Izravnava ni uspela: {exc}")
            return

    adj_output = st.session_state.get("adjustment_output")
    if adj_output is None:
        st.info(
            "Izberite geometrijsko telo in kliknite **Zaženi izravnavo po MNK** "
            "za izračun parametrov, oceno natančnosti in 3D prikaz odstopanj."
        )
        return

    if adj_output.get("processed_sig") != processed_sig:
        st.warning(
            "Očiščen oblak točk v zavihku 1 se je spremenil po zadnji izravnavi. "
            "Za posodobljene rezultate ponovno kliknite **Zaženi izravnavo po MNK**."
        )

    result: AdjustmentResult = adj_output["result"]
    points_used: np.ndarray = adj_output["points_used"]

    if not result.converged:
        st.warning(
            f"Izravnava po {result.iterations} iteracijah ni dosegla predpisane tolerance. "
            "Preverite ustreznost izbranega geometrijskega modela."
        )

    if adj_output.get("outlier_3sigma_enabled"):
        rej = adj_output.get("rejected_3sigma", 0)
        first_s0 = adj_output.get("first_pass_sigma0", 0.0)
        if rej > 0:
            st.info(
                f"Po prvi izravnavi (σ₀ = {first_s0 * 1000.0:.3f} mm) je bilo avtomatsko "
                f"izločenih **{format_int_sl(rej)}** grobih napak (|v| > 3·σ₀ = {3.0 * first_s0 * 1000.0:.3f} mm) "
                f"in izravnava je bila ponovljena na **{format_int_sl(result.n_points)}** točkah."
            )
        else:
            st.info(
                f"Vse točke so znotraj meje ±3·σ₀ (±{3.0 * first_s0 * 1000.0:.3f} mm) – "
                "nobena točka ni bila izločena."
            )

    # 2. Prikaz numeričnih rezultatov
    k_col1, k_col2, k_col3, k_col4 = st.columns(4)
    k_col1.metric(
        "Aposteriorni std. odklon σ₀",
        f"{result.sigma0 * 1000.0:.3f} mm",
        delta=f"{result.sigma0:.6f} m",
        delta_color="off",
    )
    k_col2.metric(
        "RMSE odstopanj",
        f"{result.rmse * 1000.0:.3f} mm",
        delta=f"min {result.min_residual * 1000.0:+.2f} / max {result.max_residual * 1000.0:+.2f} mm",
        delta_color="off",
    )
    k_col3.metric(
        "Število iteracij",
        f"{result.iterations}",
        delta="Konvergiralo" if result.converged else "Ni konvergiralo",
        delta_color="off",
    )
    k_col4.metric(
        "Prostostne stopnje (n - u)",
        format_int_sl(result.degrees_of_freedom),
        delta=f"n = {format_int_sl(result.n_points)}, u = {result.n_unknowns}",
        delta_color="off",
    )

    st.markdown("#### Izravnani parametri modela")
    params_df = _build_params_dataframe(result)
    st.dataframe(params_df, width="stretch", hide_index=True)

    with st.expander("Kofaktorska matrika Q_ΔΔ", expanded=False):
        q_labels = _get_qdd_labels(result)
        q_df = pd.DataFrame(
            [[f"{val:.6e}" for val in row] for row in result.Q_dd],
            index=q_labels,
            columns=q_labels,
        )
        st.dataframe(q_df, width="stretch")
        if result.model_name == "Ravnina" and "fixed_axis" in result.extra:
            st.caption(
                f"Pri ravnini je bila za preprečitev singularnosti fiksirana komponenta normale "
                f"po osi `{result.extra['fixed_axis']}` = 1.0."
            )

    # 3. 3D vizualizacija izravnanega telesa in odstopanj
    st.markdown("#### 3D vizualizacija izravnanega telesa in odstopanj")
    fig_3d, was_subsampled = build_adjustment_3d_figure(
        points_used,
        result,
        use_local_center=use_local_center,
        marker_size=marker_size,
    )
    if was_subsampled:
        st.info(
            f"Za odziven 3D izris je prikazan naključni podvzorec **{format_int_sl(MAX_PLOT_POINTS)}** "
            f"od **{format_int_sl(result.n_points)}** točk. Izravnava in histogram temeljita na vseh točkah."
        )
    st.plotly_chart(fig_3d, width="stretch")

    st.markdown("#### Histogram odstopanj (rezidualov)")
    fig_hist = build_residuals_histogram(result)
    st.plotly_chart(fig_hist, width="stretch")

    # 4. Izvoz za poročilo
    st.markdown("#### Izvoz rezultatov za poročilo")
    stem = Path(source_name).stem
    model_slug = (
        result.model_name.lower()
        .replace(" ", "_")
        .replace("č", "c")
        .replace("ž", "z")
        .replace("š", "s")
    )

    report_bytes = _generate_report_txt(
        source_name=source_name,
        prep_info=st.session_state.get("preprocessing_info"),
        volume_results=st.session_state.get("volume_results"),
        adj_output=adj_output,
    )
    csv_bytes = _generate_residuals_csv(points_used, result.residuals)

    dl_col1, dl_col2 = st.columns(2)
    with dl_col1:
        st.download_button(
            label="Prenesi poročilo (.txt)",
            data=report_bytes,
            file_name=f"{stem}_porocilo_{model_slug}.txt",
            mime="text/plain",
            width="stretch",
        )
    with dl_col2:
        st.download_button(
            label="Prenesi odstopanja (.csv)",
            data=csv_bytes,
            file_name=f"{stem}_odstopanja_{model_slug}.csv",
            mime="text/csv",
            width="stretch",
        )


# ---------------------------------------------------------------------------
# Zavihek 0 (prvi zavihek): Hitra obdelava obrezanih elementov (.txt)
# ---------------------------------------------------------------------------


def _detect_model_from_filename(filename: str) -> tuple[str, bool]:
    """Iz imena datoteke samodejno zazna predizbrani tip geometrijskega telesa.

    Vrne ``(model_name, was_auto_detected)``.
    """
    stem = (
        Path(filename)
        .stem.lower()
        .replace("č", "c")
        .replace("š", "s")
        .replace("ž", "z")
        .replace("ć", "c")
        .replace("đ", "d")
    )
    if "krogla" in stem or "sphere" in stem:
        return "Krogla", True
    if "valj" in stem or "cyl" in stem:
        return "Pokončni valj", True
    if "stozec" in stem or "cone" in stem:
        return "Pokončni stožec", True
    if "ravnin" in stem or "plane" in stem or "plosk" in stem:
        return "Ravnina", True
    return "Krogla", False


def _on_quick_model_change() -> None:
    """Ob izbiri modela 'Pokončni stožec' samodejno obkljuka 2.5D izračun volumna."""
    selected_model = st.session_state.get("quick_model", "Krogla")
    st.session_state["quick_calc_volume"] = selected_model == "Pokončni stožec"


def _sync_quick_element_widgets(
    element_id: str,
    filename: str,
    raw_points: np.ndarray,
) -> bool:
    """Sinhronizira privzete nastavitve ob preklopu na drug obrezan element."""
    detected_model, was_detected = _detect_model_from_filename(filename)

    if st.session_state.get("quick_active_element_id") != element_id:
        st.session_state["quick_active_element_id"] = element_id
        st.session_state["quick_model"] = detected_model
        st.session_state["quick_use_sor"] = True
        st.session_state["quick_sor_k"] = 20
        st.session_state["quick_sor_std"] = 2.0
        st.session_state["quick_use_voxel"] = False
        st.session_state["quick_voxel_size"] = 0.005
        st.session_state["quick_outlier_3sigma"] = True
        st.session_state["quick_calc_volume"] = detected_model == "Pokončni stožec"
        st.session_state["quick_ground_z"] = float(raw_points[:, 2].min())
        st.session_state["quick_grid_step"] = 0.01

    return was_detected


def _render_tab_quick_elements(
    use_local_center: bool,
    marker_size: float,
) -> None:
    """Izriše prvi zavihek: Hitra obdelava obrezanih elementov (.txt) iz CloudCompare."""
    st.subheader("Hitra obdelava obrezanih elementov (.txt)")
    st.caption(
        "Naložite posamezne v CloudCompare že obrezane geometrijske elemente "
        "(npr. `krogla.txt`, `valj.txt`, `stozec.txt`, `ravnina.txt`) in z enim klikom "
        "izvedite čiščenje, MNK izravnavo, opcijski 2.5D volumen ter izvoz poročila."
    )

    # 1. Nalaganje ene ali več datotek (.txt / .pts) ali izbira iz lokalne mape data/
    quick_source = st.radio(
        "Vir obrezanih elementov:",
        options=("Naloži datoteke (.txt / .pts)", "Izberi iz lokalne mape data/"),
        horizontal=True,
        key="quick_source_mode",
    )

    loaded_elements: list[dict[str, Any]] = []

    if quick_source == "Naloži datoteke (.txt / .pts)":
        uploaded_files = st.file_uploader(
            "Naložite eno ali več datotek obrezanih elementov iz CloudCompare (.txt, .pts)",
            type=["txt", "pts", "csv", "las", "laz"],
            accept_multiple_files=True,
            key="quick_file_uploader",
            help=(
                "Podprti so CloudCompare izvozi z glavo `//X Y Z ...`, `.pts` in ostali formati. "
                "Vse koordinate se berejo v `np.float64`."
            ),
        )
        if uploaded_files:
            for uf in uploaded_files:
                try:
                    raw_bytes = uf.getvalue()
                    pts = load_uploaded_cloud_cached(raw_bytes, uf.name)
                    digest = hashlib.md5(raw_bytes).hexdigest()
                    loaded_elements.append(
                        {
                            "id": f"quick_upload::{uf.name}::{len(raw_bytes)}::{digest}",
                            "name": uf.name,
                            "points": pts,
                        }
                    )
                except Exception as exc:
                    st.error(f"Napaka pri branju datoteke `{uf.name}`: {exc}")
    else:
        local_files = list_local_point_clouds(DATA_DIR)
        if not local_files:
            st.info(
                "Mapa `data/` trenutno ne vsebuje podprtih datotek "
                "(`.txt`, `.pts`, `.csv`, `.las`, `.laz`). "
                "Naložite datoteke zgoraj ali jih dodajte v mapo `data/`."
            )
        else:
            selected_paths = st.multiselect(
                "Izberite eno ali več datotek iz mape data/:",
                options=local_files,
                default=local_files,
                format_func=lambda p: p.name,
                key="quick_local_multiselect",
            )
            for path in selected_paths:
                try:
                    stat = path.stat()
                    pts = load_local_cloud_cached(str(path), stat.st_mtime_ns, stat.st_size)
                    loaded_elements.append(
                        {
                            "id": f"quick_local::{path.resolve()}::{stat.st_mtime_ns}::{stat.st_size}",
                            "name": path.name,
                            "points": pts,
                        }
                    )
                except Exception as exc:
                    st.error(f"Napaka pri branju lokalne datoteke `{path.name}`: {exc}")

    if not loaded_elements:
        st.info(
            "Za začetek naložite eno ali več obrezanih `.txt` / `.pts` datotek iz CloudCompare "
            "(npr. `krogla.txt`, `valj.txt`, `stozec.txt`, `ravnina.txt`)."
        )
        return

    # Izbirnik aktivnega elementa na vrhu (kadar je naloženih več datotek)
    if len(loaded_elements) > 1:
        elem_names = [e["name"] for e in loaded_elements]
        selected_idx = st.selectbox(
            f"Izberite element za pregled in obdelavo (naloženih: {len(loaded_elements)}):",
            options=list(range(len(loaded_elements))),
            format_func=lambda idx: elem_names[idx],
            key="quick_selected_element_idx",
        )
        active_elem = loaded_elements[selected_idx]
    else:
        active_elem = loaded_elements[0]

    elem_id: str = active_elem["id"]
    elem_name: str = active_elem["name"]
    raw_pts: np.ndarray = active_elem["points"]

    was_auto_detected = _sync_quick_element_widgets(elem_id, elem_name, raw_pts)

    # Osnovni podatki o naloženem obrezanem oblaku
    raw_min = raw_pts.min(axis=0)
    raw_max = raw_pts.max(axis=0)
    raw_dims = raw_max - raw_min
    raw_center = 0.5 * (raw_min + raw_max)

    st.markdown(f"#### Obrezan element: `{elem_name}`")
    info_c1, info_c2, info_c3, info_c4 = st.columns(4)
    info_c1.metric("Število točk (uvoženo)", format_int_sl(int(raw_pts.shape[0])))
    info_c2.metric("Box dX", f"{raw_dims[0]:.4f} m")
    info_c3.metric("Box dY", f"{raw_dims[1]:.4f} m")
    info_c4.metric("Box dZ", f"{raw_dims[2]:.4f} m")

    ctr_c1, ctr_c2, ctr_c3 = st.columns(3)
    ctr_c1.metric("Središče X", f"{raw_center[0]:.4f} m")
    ctr_c2.metric("Središče Y", f"{raw_center[1]:.4f} m")
    ctr_c3.metric("Središče Z", f"{raw_center[2]:.4f} m")

    # 2. Nastavitve in gumb "Izvedi celotno obdelavo"
    st.markdown("#### Nastavitve hitre obdelave")
    model_col, hint_col = st.columns([2, 3])
    with model_col:
        model_choice = st.selectbox(
            "Tip geometrijskega telesa",
            options=tuple(ADJUSTMENT_MODELS.keys()),
            key="quick_model",
            on_change=_on_quick_model_change,
        )
    with hint_col:
        st.write("")
        if was_auto_detected:
            st.caption(
                f"Tip telesa je bil samodejno predizbran iz imena datoteke `{elem_name}`."
            )
        else:
            st.caption(
                "Tip telesa lahko kadarkoli spremenite (imena z `krogla`, `valj`, `stozec`, "
                "`ravnina` se zaznajo samodejno)."
            )

    opt_c1, opt_c2 = st.columns(2)
    with opt_c1:
        use_sor = st.checkbox(
            "Odstrani osamelce (SOR filter)",
            key="quick_use_sor",
        )
        sor_sub1, sor_sub2 = st.columns(2)
        with sor_sub1:
            sor_k = int(
                st.number_input(
                    "SOR sosedi (k)",
                    min_value=2,
                    max_value=200,
                    step=1,
                    key="quick_sor_k",
                    disabled=not use_sor,
                )
            )
        with sor_sub2:
            sor_std = float(
                st.number_input(
                    "SOR std_ratio",
                    min_value=0.1,
                    max_value=10.0,
                    step=0.1,
                    format="%.2f",
                    key="quick_sor_std",
                    disabled=not use_sor,
                )
            )

        use_voxel = st.checkbox(
            "Razredči oblak (Voxel resample)",
            key="quick_use_voxel",
        )
        voxel_size = float(
            st.number_input(
                "Velikost voksla [m]",
                min_value=0.0005,
                max_value=2.0,
                step=0.001,
                format="%.4f",
                key="quick_voxel_size",
                disabled=not use_voxel,
            )
        )

    with opt_c2:
        outlier_3sigma = st.checkbox(
            "Izloči grobe napake > 3 * sigma0 in ponovi izravnavo po MNK",
            key="quick_outlier_3sigma",
        )
        calc_volume = st.checkbox(
            "Izračunaj tudi 2.5D volumen",
            key="quick_calc_volume",
            help="Samodejno obkljukano za 'Pokončni stožec'.",
        )
        vol_sub1, vol_sub2 = st.columns(2)
        with vol_sub1:
            ground_z = float(
                st.number_input(
                    "Referenčna višina ground_z [m]",
                    step=0.005,
                    format="%.4f",
                    key="quick_ground_z",
                    disabled=not calc_volume,
                )
            )
        with vol_sub2:
            grid_step = float(
                st.number_input(
                    "Korak mreže grid_step [m]",
                    min_value=0.001,
                    max_value=5.0,
                    step=0.005,
                    format="%.4f",
                    key="quick_grid_step",
                    disabled=not calc_volume,
                )
            )

    run_quick = st.button("Izvedi celotno obdelavo", type="primary", key="quick_run_button")

    quick_store: dict[str, dict[str, Any]] = st.session_state.setdefault("quick_results", {})

    if run_quick:
        try:
            with st.spinner(f"Izvajanje celotne obdelave za `{elem_name}` ({model_choice}) ..."):
                current = raw_pts
                sor_removed = 0
                if use_sor and current.shape[0] > 1:
                    current, sor_removed = statistical_outlier_removal(
                        current,
                        k_neighbors=sor_k,
                        std_ratio=sor_std,
                    )
                after_sor = int(current.shape[0])

                if use_voxel and current.shape[0] > 0:
                    current = resample_cloud(
                        current,
                        method="voxel",
                        voxel_size=voxel_size,
                    )
                cleaned_pts = current

                fit_fn = ADJUSTMENT_MODELS[model_choice]
                res_first = fit_fn(cleaned_pts)
                points_used = cleaned_pts
                final_res = res_first
                rejected_3sigma = 0
                first_pass_sigma0 = res_first.sigma0

                if outlier_3sigma and res_first.sigma0 > 0.0:
                    inlier_mask = np.abs(res_first.residuals) <= 3.0 * res_first.sigma0
                    rejected_3sigma = int(cleaned_pts.shape[0] - np.count_nonzero(inlier_mask))
                    if rejected_3sigma > 0:
                        if int(np.count_nonzero(inlier_mask)) <= res_first.n_unknowns:
                            st.warning(
                                "Izločanje > 3·σ₀ bi pustilo premalo točk za ponovitev izravnave; "
                                "ohranjen je rezultat prve izravnave."
                            )
                            rejected_3sigma = 0
                        else:
                            points_used = np.ascontiguousarray(
                                cleaned_pts[inlier_mask], dtype=np.float64
                            )
                            final_res = fit_fn(points_used)

                vol_results: dict[str, Any] | None = None
                vol_z_matrix: np.ndarray | None = None
                if calc_volume:
                    vol_results, vol_z_matrix = compute_25d_volume(
                        points_used,
                        ground_z=ground_z,
                        grid_step=grid_step,
                    )

                prep_info = {
                    "cropped_externally": True,
                    "raw_count": int(raw_pts.shape[0]),
                    "processed_count": int(cleaned_pts.shape[0]),
                    "use_sor": use_sor,
                    "sor_k": sor_k,
                    "sor_std": sor_std,
                    "sor_removed": sor_removed,
                    "after_sor": after_sor,
                    "use_dbscan": False,
                    "resample_mode": "Voxel" if use_voxel else "Brez redčenja",
                    "voxel_size": voxel_size,
                    "target_count": int(cleaned_pts.shape[0]),
                }

                adj_output = {
                    "result": final_res,
                    "points_used": points_used,
                    "model_choice": model_choice,
                    "outlier_3sigma_enabled": outlier_3sigma,
                    "rejected_3sigma": rejected_3sigma,
                    "first_pass_sigma0": first_pass_sigma0,
                }

                quick_store[elem_id] = {
                    "elem_name": elem_name,
                    "prep_info": prep_info,
                    "adj_output": adj_output,
                    "vol_results": vol_results,
                    "vol_z_matrix": vol_z_matrix,
                }
        except ValueError as exc:
            st.error(f"Obdelava ni uspela: {exc}")
            return

    bundle = quick_store.get(elem_id)
    if bundle is None:
        st.info(
            "Preverite izbrani tip geometrijskega telesa in kliknite "
            "**Izvedi celotno obdelavo** za zagon čiščenja, MNK izravnave in vizualizacije."
        )
        return

    # 3. Prikaz rezultatov celotne obdelave
    st.divider()
    prep_info = bundle["prep_info"]
    adj_output = bundle["adj_output"]
    result: AdjustmentResult = adj_output["result"]
    points_used: np.ndarray = adj_output["points_used"]
    vol_results: dict[str, Any] | None = bundle["vol_results"]
    vol_z_matrix: np.ndarray | None = bundle["vol_z_matrix"]

    st.markdown(f"### Rezultati obdelave: `{elem_name}` ({result.model_name})")

    if not result.converged:
        st.warning(
            f"Izravnava po {result.iterations} iteracijah ni dosegla predpisane tolerance."
        )

    clean_summary_parts = [f"Uvoženih: **{format_int_sl(prep_info['raw_count'])}**"]
    if prep_info["use_sor"]:
        clean_summary_parts.append(f"SOR odstranil: **{format_int_sl(prep_info['sor_removed'])}**")
    if prep_info["resample_mode"] == "Voxel":
        clean_summary_parts.append(
            f"Po Voxel ({prep_info['voxel_size']:.4f} m): **{format_int_sl(prep_info['processed_count'])}**"
        )
    if adj_output["outlier_3sigma_enabled"]:
        clean_summary_parts.append(
            f"Izločenih > 3·σ₀: **{format_int_sl(adj_output['rejected_3sigma'])}**"
        )
    clean_summary_parts.append(f"Končnih v MNK: **{format_int_sl(result.n_points)}**")
    st.caption(" | ".join(clean_summary_parts))

    # Metrike MNK
    qm1, qm2, qm3, qm4, qm5 = st.columns(5)
    qm1.metric(
        "σ₀ (aposteriorni)",
        f"{result.sigma0 * 1000.0:.3f} mm",
        delta=f"{result.sigma0:.6f} m",
        delta_color="off",
    )
    qm2.metric(
        "RMSE odstopanj",
        f"{result.rmse * 1000.0:.3f} mm",
        delta=f"min {result.min_residual * 1000.0:+.2f} / max {result.max_residual * 1000.0:+.2f} mm",
        delta_color="off",
    )
    qm3.metric("Število točk (n)", format_int_sl(result.n_points))
    qm4.metric(
        "Prostostne stopnje (n - u)",
        format_int_sl(result.degrees_of_freedom),
        delta=f"u = {result.n_unknowns}",
        delta_color="off",
    )
    qm5.metric(
        "Število iteracij",
        f"{result.iterations}",
        delta="Konvergiralo" if result.converged else "Ni konvergiralo",
        delta_color="off",
    )

    # Tabela parametrov + kofaktorska matrika
    st.markdown("#### Izravnani parametri modela")
    st.dataframe(_build_params_dataframe(result), width="stretch", hide_index=True)

    with st.expander("Kofaktorska matrika Q_ΔΔ", expanded=False):
        q_labels = _get_qdd_labels(result)
        q_df = pd.DataFrame(
            [[f"{val:.6e}" for val in row] for row in result.Q_dd],
            index=q_labels,
            columns=q_labels,
        )
        st.dataframe(q_df, width="stretch")
        if result.model_name == "Ravnina" and "fixed_axis" in result.extra:
            st.caption(
                f"Pri ravnini je bila fiksirana komponenta normale po osi "
                f"`{result.extra['fixed_axis']}` = 1.0."
            )

    # Opcijski 2.5D volumen in 2D heatmap
    if vol_results is not None and vol_z_matrix is not None:
        st.markdown("#### Izračun 2.5D volumna")
        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Skupni volumen [m³]", f"{vol_results['volume_m3']:.6f} m³")
        vc2.metric("Skupni volumen [dm³ / L]", f"{vol_results['volume_dm3']:.3f} dm³")
        vc3.metric("Površina tlorisa [m²]", f"{vol_results['surface_area_m2']:.4f} m²")
        vc4.metric(
            "Dodani / odvzeti volumen",
            f"+{vol_results['added_volume_dm3']:.2f} / -{vol_results['removed_volume_dm3']:.2f} dm³",
        )
        fig_qheat, was_h_down = build_volume_heatmap(
            vol_z_matrix,
            vol_results["x_coords"],
            vol_results["y_coords"],
            ground_z=vol_results["ground_z"],
            relative_height=False,
        )
        if was_h_down:
            st.info(
                "Toplotni zemljevid je pomanjšan za odziven prikaz; volumen je izračunan na polni mreži."
            )
        st.plotly_chart(fig_qheat, width="stretch")

    # 3D Plotly graf: točke po odstopanjih + polprosojna ploskev
    st.markdown("#### 3D prikaz odstopanj in izravnane ploskve")
    fig_q3d, was_sub = build_adjustment_3d_figure(
        points_used,
        result,
        use_local_center=use_local_center,
        marker_size=marker_size,
    )
    if was_sub:
        st.info(
            f"Za odziven 3D izris je prikazan naključni podvzorec **{format_int_sl(MAX_PLOT_POINTS)}** "
            f"od **{format_int_sl(result.n_points)}** točk. Vsi izračuni temeljijo na vseh točkah."
        )
    st.plotly_chart(fig_q3d, width="stretch")

    # Histogram odstopanj
    st.markdown("#### Histogram odstopanj (rezidualov)")
    st.plotly_chart(build_residuals_histogram(result), width="stretch")

    # 4. Izvoz za poročilo
    st.markdown("#### Izvoz poročila za ta element")
    stem = Path(elem_name).stem
    model_slug = (
        result.model_name.lower()
        .replace(" ", "_")
        .replace("č", "c")
        .replace("ž", "z")
        .replace("š", "s")
    )
    vol_for_report = (
        {k: v for k, v in vol_results.items() if k not in ("x_coords", "y_coords")}
        if vol_results is not None
        else None
    )
    report_bytes = _generate_report_txt(
        source_name=elem_name,
        prep_info=prep_info,
        volume_results=vol_for_report,
        adj_output=adj_output,
    )
    csv_bytes = _generate_residuals_csv(points_used, result.residuals)

    qdl1, qdl2 = st.columns(2)
    with qdl1:
        st.download_button(
            label="Prenesi poročilo za ta element (.txt)",
            data=report_bytes,
            file_name=f"{stem}_porocilo_{model_slug}.txt",
            mime="text/plain",
            width="stretch",
            key=f"quick_dl_report_{elem_id}",
        )
    with qdl2:
        st.download_button(
            label="Prenesi odstopanja (.csv)",
            data=csv_bytes,
            file_name=f"{stem}_odstopanja_{model_slug}.csv",
            mime="text/csv",
            width="stretch",
            key=f"quick_dl_csv_{elem_id}",
        )


def main() -> None:
    """Glavna vstopna točka Streamlit uporabniškega vmesnika."""
    st.title("GMS Vaja 2 – Obdelava oblaka točk")
    st.caption(
        "Hitra obdelava obrezanih elementov ter celotna predobdelava (CloudCompare orodja), "
        "2.5D volumetrija in geodetska izravnava po MNK za Leica MS60 / Leica Infinity."
    )

    st.sidebar.header("1. Vir surovega oblaka (za ročno obdelavo)")
    source_mode = st.sidebar.radio(
        "Izberite način nalaganja:",
        options=("Lokalna mapa data/", "Naloži datoteko"),
    )

    points: np.ndarray | None = None
    active_source_name: str | None = None
    source_id: str | None = None

    if source_mode == "Lokalna mapa data/":
        local_files = list_local_point_clouds(DATA_DIR)
        if not local_files:
            st.sidebar.info(
                "Mapa `data/` trenutno ne vsebuje podprtih datotek "
                "(`.pts`, `.txt`, `.csv`, `.las`, `.laz`)."
            )
        else:
            selected_file = st.sidebar.selectbox(
                "Izberite oblak točk iz mape data/:",
                options=local_files,
                format_func=lambda p: p.name,
            )
            if selected_file is not None:
                try:
                    stat = selected_file.stat()
                    points = load_local_cloud_cached(
                        str(selected_file),
                        stat.st_mtime_ns,
                        stat.st_size,
                    )
                    active_source_name = selected_file.name
                    source_id = (
                        f"local::{selected_file.resolve()}::{stat.st_mtime_ns}::{stat.st_size}"
                    )
                except Exception as exc:
                    st.error(f"Napaka pri branju datoteke `{selected_file.name}`: {exc}")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Naložite oblak točk",
            type=["pts", "las", "laz", "txt", "csv"],
            help="Podprti formati: .pts, .las, .laz, .txt, .csv (koordinate v np.float64).",
        )
        if uploaded_file is not None:
            try:
                file_bytes = uploaded_file.getvalue()
                points = load_uploaded_cloud_cached(file_bytes, uploaded_file.name)
                active_source_name = uploaded_file.name
                digest = hashlib.md5(file_bytes).hexdigest()
                source_id = f"upload::{uploaded_file.name}::{len(file_bytes)}::{digest}"
            except Exception as exc:
                st.error(f"Napaka pri branju naložene datoteke `{uploaded_file.name}`: {exc}")

    has_sidebar_cloud = (
        points is not None and source_id is not None and active_source_name is not None
    )

    if has_sidebar_cloud:
        assert points is not None and source_id is not None and active_source_name is not None
        x_bounds, y_bounds, z_bounds = _sync_session_state_for_source(
            source_id=source_id,
            source_name=active_source_name,
            raw_points=points,
        )
    else:
        st.session_state.pop("active_source_id", None)
        st.session_state.pop("raw_points", None)
        st.session_state.pop("processed_points", None)
        st.session_state.pop("adjustment_output", None)
        st.session_state.pop("volume_results", None)
        st.session_state.pop("preprocessing_info", None)
        if "use_local_center" not in st.session_state:
            st.session_state["use_local_center"] = True

    # Nastavitve 3D prikaza v stranski vrstici (veljajo za vse zavihke)
    st.sidebar.divider()
    st.sidebar.header("Nastavitve 3D prikaza")
    use_local_center = st.sidebar.checkbox(
        "Centriraj 3D pogled na središče oblaka",
        key="use_local_center",
        help=(
            "Pri velikih geodetskih koordinatah WebGL (float32) v brskalniku povzroči "
            "stopničavost izrisa. Centriranje ohrani natančno geometrijo v 3D pogledu, "
            "medtem ko obvestilo ob miški in barvna lestvica vedno prikazujeta prave vrednosti."
        ),
    )
    marker_size = st.sidebar.slider(
        "Velikost točk v 3D pogledu",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
    )

    tab_quick, tab1, tab2, tab3 = st.tabs(
        [
            "Hitra obdelava obrezanih elementov (.txt)",
            "Pregled in predobdelava (CloudCompare)",
            "Izračun volumna (2.5D)",
            "Izravnava po MNK",
        ]
    )

    with tab_quick:
        _render_tab_quick_elements(
            use_local_center=use_local_center,
            marker_size=marker_size,
        )

    if not has_sidebar_cloud:
        sidebar_prompt = (
            "Za ročno obrezovanje in korak-po-koraku obdelavo v stranski vrstici izberite "
            "surovi oblak točk iz mape `data/` ali naložite datoteko (`.pts`, `.las`, `.laz`, `.txt`, `.csv`). "
            "Če že imate obrezane elemente iz CloudCompare, uporabite prvi zavihek "
            "**Hitra obdelava obrezanih elementov (.txt)**."
        )
        with tab1:
            st.info(sidebar_prompt)
        with tab2:
            st.info(sidebar_prompt)
        with tab3:
            st.info(sidebar_prompt)
        return

    assert points is not None and source_id is not None and active_source_name is not None

    with tab1:
        processed_points, processed_sig = _render_tab_preprocessing(
            source_id=source_id,
            source_name=active_source_name,
            raw_points=points,
            x_bounds=x_bounds,
            y_bounds=y_bounds,
            z_bounds=z_bounds,
            use_local_center=use_local_center,
            marker_size=marker_size,
        )

    with tab2:
        _render_tab_volume(
            processed_points=processed_points,
            processed_sig=processed_sig,
        )

    with tab3:
        _render_tab_adjustment(
            source_name=active_source_name,
            processed_sig=processed_sig,
            use_local_center=use_local_center,
            marker_size=marker_size,
        )


if __name__ == "__main__":
    main()
