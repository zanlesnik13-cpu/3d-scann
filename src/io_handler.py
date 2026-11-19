"""Modul za branje in shranjevanje oblakov točk (Leica MS60 / Leica Infinity).

Podprti vhodni formati: ``.pts``, ``.txt``, ``.csv``, ``.las``, ``.laz``.
Vse koordinate ``(X, Y, Z)`` so vedno shranjene v 64-bitni natančnosti (``np.float64``),
da se ohrani popolna natančnost velikih geodetskih koordinat (npr. D96/TM).
"""

from __future__ import annotations

import io
import math
from pathlib import Path
from typing import Sequence

import laspy
import numpy as np
import pandas as pd

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pts", ".txt", ".csv", ".las", ".laz")
ASCII_EXTENSIONS: tuple[str, ...] = (".pts", ".txt", ".csv")
LAS_EXTENSIONS: tuple[str, ...] = (".las", ".laz")

_CANDIDATE_DELIMITERS: tuple[str | None, ...] = (";", ",", "\t", None)
_COMMENT_PREFIXES: tuple[str, ...] = ("#", "//")
_SAMPLE_LINE_LIMIT: int = 200


def list_local_point_clouds(data_dir: str | Path = "data") -> list[Path]:
    """Vrne urejen seznam vseh podprtih datotek oblakov točk v mapi ``data_dir``.

    Če mapa ne obstaja ali ne vsebuje podprtih datotek, vrne prazen seznam.
    """
    directory = Path(data_dir)
    if not directory.exists() or not directory.is_dir():
        return []

    files = [
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda p: p.name.lower())


def _is_finite_float(token: str) -> bool:
    """Preveri, ali je niz veljavno končno realno število."""
    try:
        return math.isfinite(float(token))
    except ValueError:
        return False


def _split_line(line: str, delimiter: str | None) -> list[str]:
    """Razdeli vrstico glede na podano ločilo (ali poljubne presledke pri ``None``)."""
    stripped = line.strip()
    if not stripped:
        return []
    if delimiter is None:
        return stripped.split()
    return [part.strip() for part in stripped.split(delimiter)]


def _extract_first_three_numeric_indices(tokens: Sequence[str]) -> list[int] | None:
    """Poišče indekse prvih treh numeričnih stolpcev v razdeljeni vrstici."""
    numeric_indices: list[int] = []
    for idx, token in enumerate(tokens):
        if token and _is_finite_float(token):
            numeric_indices.append(idx)
            if len(numeric_indices) == 3:
                return numeric_indices
    return None


def _detect_ascii_format(lines: Sequence[str]) -> tuple[str | None, int, list[int]]:
    """Samodejno zazna ločilo, število vrstic glave in indekse stolpcev X, Y, Z.

    Preskoči prazne vrstice, komentarje, tekstovne glave in enostolpčne vrstice
    (npr. število točk v prvi vrstici Leica/CloudCompare ``.pts`` datotek).

    Vrne:
        ``(delimiter, skiprows, usecols)``
    """
    best_candidate: tuple[int, str | None, int, list[int]] | None = None

    for delimiter in _CANDIDATE_DELIMITERS:
        first_data_row: int | None = None
        first_cols: list[int] | None = None
        matching_rows = 0

        for line_idx, raw_line in enumerate(lines):
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(_COMMENT_PREFIXES):
                continue

            tokens = _split_line(stripped, delimiter)
            if len(tokens) < 3:
                continue

            numeric_cols = _extract_first_three_numeric_indices(tokens)
            if numeric_cols is None:
                continue

            if first_data_row is None:
                first_data_row = line_idx
                first_cols = numeric_cols
                matching_rows = 1
            elif numeric_cols == first_cols:
                matching_rows += 1
            else:
                # Če vrstica ne ustreza vzorcu, nadaljuj iskanje
                continue

        if first_data_row is not None and first_cols is not None:
            # Preferiramo ločilo z največ ujemajočimi se vrsticami; pri izenačenju
            # ima prednost eksplicitno ločilo pred presledki (vrstni red _CANDIDATE_DELIMITERS).
            if best_candidate is None or matching_rows > best_candidate[0]:
                best_candidate = (matching_rows, delimiter, first_data_row, first_cols)

    if best_candidate is None:
        raise ValueError(
            "V datoteki ni bilo mogoče zaznati veljavnih vrstic s koordinatami (X, Y, Z)."
        )

    _, delimiter, skiprows, usecols = best_candidate
    return delimiter, skiprows, usecols


def _parse_ascii_fallback(
    text_stream: io.TextIOBase,
    delimiter: str | None,
    usecols: Sequence[int],
) -> np.ndarray:
    """Robusten vrstični razčlenjevalnik za primer neenotnih vrstic (npr. združeni .pts)."""
    max_col = max(usecols)
    c0, c1, c2 = usecols
    rows: list[tuple[float, float, float]] = []

    for raw_line in text_stream:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith(_COMMENT_PREFIXES):
            continue

        tokens = _split_line(stripped, delimiter)
        if len(tokens) <= max_col:
            continue

        try:
            x = float(tokens[c0])
            y = float(tokens[c1])
            z = float(tokens[c2])
        except ValueError:
            continue

        if math.isfinite(x) and math.isfinite(y) and math.isfinite(z):
            rows.append((x, y, z))

    if not rows:
        return np.empty((0, 3), dtype=np.float64)
    return np.asarray(rows, dtype=np.float64)


def _decode_text_bytes(data: bytes) -> str:
    """Dekodira besedilno datoteko (UTF-8 z ali brez BOM, rezervno CP1250/Latin-1)."""
    for encoding in ("utf-8-sig", "cp1250", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _read_ascii_from_text_buffer(text_buffer: io.StringIO) -> np.ndarray:
    """Prebere koordinate X, Y, Z (``np.float64``) iz besedilnega toka."""
    sample_lines: list[str] = []
    for _ in range(_SAMPLE_LINE_LIMIT):
        line = text_buffer.readline()
        if not line:
            break
        sample_lines.append(line)

    delimiter, skiprows, usecols = _detect_ascii_format(sample_lines)
    sep = r"\s+" if delimiter is None else delimiter

    text_buffer.seek(0)
    try:
        df = pd.read_csv(
            text_buffer,
            sep=sep,
            header=None,
            skiprows=skiprows,
            usecols=usecols,
            dtype=np.float64,
            skip_blank_lines=True,
            engine="c",
        )
        # Uredi stolpce v prvotnem vrstnem redu usecols
        df = df[usecols]
        points = df.to_numpy(dtype=np.float64, copy=True)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        text_buffer.seek(0)
        points = _parse_ascii_fallback(text_buffer, delimiter, usecols)

    return _validate_points_array(points)


def _read_ascii_from_file(filepath: Path) -> np.ndarray:
    """Prebere koordinate X, Y, Z (``np.float64``) iz ASCII datoteke (.pts, .txt, .csv)."""
    encoding = "utf-8-sig"
    sample_lines: list[str] = []

    for candidate_encoding in ("utf-8-sig", "cp1250", "latin-1"):
        try:
            with filepath.open("r", encoding=candidate_encoding) as f:
                sample_lines = []
                for _ in range(_SAMPLE_LINE_LIMIT):
                    line = f.readline()
                    if not line:
                        break
                    sample_lines.append(line)
            encoding = candidate_encoding
            break
        except UnicodeDecodeError:
            continue

    delimiter, skiprows, usecols = _detect_ascii_format(sample_lines)
    sep = r"\s+" if delimiter is None else delimiter

    try:
        df = pd.read_csv(
            filepath,
            sep=sep,
            header=None,
            skiprows=skiprows,
            usecols=usecols,
            dtype=np.float64,
            skip_blank_lines=True,
            encoding=encoding,
            engine="c",
        )
        df = df[usecols]
        points = df.to_numpy(dtype=np.float64, copy=True)
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        with filepath.open("r", encoding=encoding, errors="replace") as f:
            points = _parse_ascii_fallback(f, delimiter, usecols)

    return _validate_points_array(points)


def _read_las_object(las: laspy.LasData) -> np.ndarray:
    """Izlušči koordinate X, Y, Z iz objekta ``laspy.LasData`` kot ``np.float64``."""
    x = np.asarray(las.x, dtype=np.float64)
    y = np.asarray(las.y, dtype=np.float64)
    z = np.asarray(las.z, dtype=np.float64)
    points = np.column_stack((x, y, z))
    return _validate_points_array(points)


def _validate_points_array(points: np.ndarray) -> np.ndarray:
    """Zagotovi obliko ``(N, 3)``, tip ``np.float64`` ter odstrani morebitne NaN/Inf točke."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(
            f"Pričakovana oblika koordinat je (N, 3), dobljena pa {arr.shape}."
        )

    finite_mask = np.all(np.isfinite(arr), axis=1)
    if not np.all(finite_mask):
        arr = arr[finite_mask]

    if arr.shape[0] == 0:
        raise ValueError("Oblak točk ne vsebuje nobene veljavne končne točke.")

    return np.ascontiguousarray(arr, dtype=np.float64)


def load_point_cloud(filepath: str | Path) -> np.ndarray:
    """Prebere oblak točk iz lokalne datoteke in vrne tabelo ``(N, 3)`` tipa ``np.float64``.

    Podprti formati: ``.pts``, ``.txt``, ``.csv``, ``.las``, ``.laz``.
    """
    path = Path(filepath)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Datoteka ne obstaja: {path}")

    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Nepodprt format datoteke '{ext}'. Podprti formati: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext in LAS_EXTENSIONS:
        las = laspy.read(path)
        return _read_las_object(las)

    return _read_ascii_from_file(path)


def load_point_cloud_from_bytes(data: bytes, filename: str) -> np.ndarray:
    """Prebere oblak točk iz bajtov (npr. Streamlit ``UploadedFile``).

    Vrne tabelo ``(N, 3)`` tipa ``np.float64``.
    """
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Nepodprt format datoteke '{ext}'. Podprti formati: {', '.join(SUPPORTED_EXTENSIONS)}"
        )

    if ext in LAS_EXTENSIONS:
        las = laspy.read(io.BytesIO(data))
        return _read_las_object(las)

    text = _decode_text_bytes(data)
    return _read_ascii_from_text_buffer(io.StringIO(text))


def point_cloud_to_txt_bytes(points: np.ndarray) -> bytes:
    """Pretvori oblak točk ``(N, 3)`` v bajte ``.txt`` zapisa s 6 decimalnimi mesti.

    Uporablja ločilo presledek in natančnost ``%.6f``, enako kot ``save_point_cloud_txt``.
    """
    raw = np.asarray(points, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] < 3:
        raise ValueError(
            f"Pričakovana 2D tabela z vsaj 3 stolpci (X, Y, Z), dobljena pa oblika {raw.shape}."
        )
    arr = _validate_points_array(raw[:, :3])
    buffer = io.BytesIO()
    np.savetxt(buffer, arr, fmt="%.6f", delimiter=" ")
    return buffer.getvalue()


def save_point_cloud_txt(points: np.ndarray, filepath: str | Path) -> Path:
    """Shrani očiščen oblak točk ``(N, 3)`` v ``.txt`` format s 6 decimalnimi mesti.

    Uporablja ločilo presledek in natančnost ``%.6f``, primerno za geodetske koordinate.
    """
    txt_bytes = point_cloud_to_txt_bytes(points)
    out_path = Path(filepath)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(txt_bytes)
    return out_path
