"""Glavna Streamlit aplikacija za obdelavo oblakov točk (GMS Vaja 2).

Namenjena obdelavi oblakov točk, posnetih z elektronskim tahimetrom Leica MS60
in izvoženih iz programske opreme Leica Infinity.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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

    # Shranimo očiščen oblak v session_state za naslednje zavihke
    st.session_state["processed_points"] = processed_points

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
        st.warning(str(exc))
        return

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


def main() -> None:
    """Glavna vstopna točka Streamlit uporabniškega vmesnika."""
    st.title("GMS Vaja 2 – Obdelava oblaka točk")
    st.caption(
        "Predogled, predobdelava (CloudCompare orodja) in volumetrija oblakov točk, "
        "posnetih z elektronskim tahimetrom Leica MS60 (izvoz iz Leica Infinity)."
    )

    st.sidebar.header("1. Vir podatkov")
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

    if points is None or source_id is None or active_source_name is None:
        st.session_state.pop("active_source_id", None)
        st.session_state.pop("raw_points", None)
        st.session_state.pop("processed_points", None)
        st.info(
            "Za začetek izberite oblak točk iz mape `data/` ali pa v stranski vrstici "
            "naložite datoteko (`.pts`, `.las`, `.laz`, `.txt`, `.csv`)."
        )
        return

    x_bounds, y_bounds, z_bounds = _sync_session_state_for_source(
        source_id=source_id,
        source_name=active_source_name,
        raw_points=points,
    )

    # Nastavitve 3D prikaza v stranski vrstici
    st.sidebar.divider()
    st.sidebar.header("Nastavitve 3D prikaza")
    use_local_center = st.sidebar.checkbox(
        "Centriraj 3D pogled na središče oblaka",
        key="use_local_center",
        help=(
            "Pri velikih geodetskih koordinatah WebGL (float32) v brskalniku povzroči "
            "stopničavost izrisa. Centriranje ohrani natančno geometrijo v 3D pogledu, "
            "medtem ko obvestilo ob miški in barvna lestvica Z vedno prikazujeta prave koordinate."
        ),
    )
    marker_size = st.sidebar.slider(
        "Velikost točk v 3D pogledu",
        min_value=1.0,
        max_value=5.0,
        value=2.0,
        step=0.5,
    )

    tab1, tab2, tab3 = st.tabs(
        [
            "Pregled in predobdelava (CloudCompare)",
            "Izračun volumna (2.5D)",
            "Izravnava po MNK",
        ]
    )

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
        st.subheader("Izravnava po metodi najmanjših kvadratov (MNK)")
        st.info("V pripravi – 3. korak")


if __name__ == "__main__":
    main()
