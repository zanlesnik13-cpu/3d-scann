"""Glavna Streamlit aplikacija za obdelavo oblakov točk (GMS Vaja 2).

Namenjena obdelavi oblakov točk, posnetih z elektronskim tahimetrom Leica MS60
in izvoženih iz programske opreme Leica Infinity.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.io_handler import (
    list_local_point_clouds,
    load_point_cloud,
    load_point_cloud_from_bytes,
)

DATA_DIR = Path(__file__).resolve().parent / "data"
MAX_PLOT_POINTS = 30_000
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


def format_int_sl(value: int) -> str:
    """Oblikuje celo število s presledki kot ločili tisočic."""
    return f"{value:,}".replace(",", " ")


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
        height=650,
        scene=dict(
            aspectmode="data",
            xaxis_title=x_title,
            yaxis_title=y_title,
            zaxis_title=z_title,
        ),
    )
    return fig


def main() -> None:
    """Glavna vstopna točka Streamlit uporabniškega vmesnika."""
    st.title("GMS Vaja 2 – Obdelava oblaka točk")
    st.caption(
        "Predogled in obdelava oblakov točk, posnetih z elektronskim tahimetrom "
        "Leica MS60 (izvoz iz Leica Infinity)."
    )

    st.sidebar.header("1. Vir podatkov")
    source_mode = st.sidebar.radio(
        "Izberite način nalaganja:",
        options=("Lokalna mapa data/", "Naloži datoteko"),
    )

    points: np.ndarray | None = None
    active_source_name: str | None = None

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
            except Exception as exc:
                st.error(f"Napaka pri branju naložene datoteke `{uploaded_file.name}`: {exc}")

    if points is None:
        st.info(
            "Za začetek izberite oblak točk iz mape `data/` ali pa v stranski vrstici "
            "naložite datoteko (`.pts`, `.las`, `.laz`, `.txt`, `.csv`)."
        )
        return

    # Shranimo celoten oblak v session_state za nadaljnje korake obdelave
    st.session_state["raw_points"] = points
    st.session_state["active_source_name"] = active_source_name

    n_points = int(points.shape[0])
    min_xyz = points.min(axis=0)
    max_xyz = points.max(axis=0)
    dims = max_xyz - min_xyz
    center = 0.5 * (min_xyz + max_xyz)

    st.subheader(f"Pregled oblaka točk: `{active_source_name}`")

    # Metrike: število točk in dimenzije omejevalnega kvadra
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Skupno število točk", format_int_sl(n_points))
    col2.metric("Dimenzija dX", f"{dims[0]:.4f} m")
    col3.metric("Dimenzija dY", f"{dims[1]:.4f} m")
    col4.metric("Dimenzija dZ", f"{dims[2]:.4f} m")

    # Metrike: koordinate središča omejevalnega kvadra
    c_col1, c_col2, c_col3 = st.columns(3)
    c_col1.metric("Središče X (Box center)", f"{center[0]:.4f} m")
    c_col2.metric("Središče Y (Box center)", f"{center[1]:.4f} m")
    c_col3.metric("Središče Z (Box center)", f"{center[2]:.4f} m")

    with st.expander("Podrobnosti omejevalnega kvadra (float64 natančnost)", expanded=False):
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
        st.caption(f"Shranjen tip koordinat v pomnilniku: `{points.dtype}` (oblika: `{points.shape}`).")

    # Nastavitve 3D prikaza v stranski vrstici
    st.sidebar.divider()
    st.sidebar.header("Nastavitve 3D prikaza")
    has_large_coords = bool(np.max(np.abs(center)) > LARGE_COORD_THRESHOLD)
    use_local_center = st.sidebar.checkbox(
        "Centriraj 3D pogled na središče oblaka",
        value=has_large_coords,
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

    st.subheader("Interaktivni 3D predogled")
    if n_points > MAX_PLOT_POINTS:
        rng = np.random.default_rng(42)
        sample_indices = rng.choice(n_points, size=MAX_PLOT_POINTS, replace=False)
        plot_points = points[sample_indices]
        st.info(
            f"Za odziven 3D izris v brskalniku je prikazan naključni podvzorec "
            f"**{format_int_sl(MAX_PLOT_POINTS)}** od skupno **{format_int_sl(n_points)}** točk. "
            f"Za vse nadaljnje izračune se v ozadju ohranja celoten oblak točk (`float64`)."
        )
    else:
        plot_points = points

    fig = build_point_cloud_figure(
        plot_points,
        center,
        use_local_center=use_local_center,
        marker_size=marker_size,
    )
    st.plotly_chart(fig, width="stretch")


if __name__ == "__main__":
    main()
