"""Scientific transforms of a loaded record's retained paths.

Pure functions: a :class:`~channels4world.Record` in, new (writable) NumPy arrays out.
They describe only the retained paths (at most 16 per link above -150 dB) and never
reconstruct discarded paths or absent links.

Coefficient convention (see docs/conventions.md): ``a_real + 1j*a_imag`` is Sionna RT's
passband path coefficient ``Paths.a``. It does not include the carrier-delay phase
exp(-j*2*pi*fc*tau); ``tau`` is the absolute propagation delay in seconds and fc is the
record's ``frequency_hz``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

__all__ = ["complex_gain", "path_power_db", "delay_spread", "baseband_coefficients", "frequency_response",
           "pad_paths", "pad_interactions", "MAX_PATHS", "MAX_INTERACTIONS"]

MAX_PATHS = 16
"""Retained paths per link: at most the 16 strongest above -150 dB."""
MAX_INTERACTIONS = 4
"""Interactions per path: line of sight has none; the simulation allowed up to four."""

_REFERENCES = ("absolute", "first_retained")


def _link_of_path(rec: Any) -> np.ndarray:
    counts = np.diff(np.asarray(rec.core["link_path_offsets"], dtype=np.int64))
    return np.repeat(np.arange(len(counts)), counts)


def complex_gain(rec: Any) -> np.ndarray:
    """Per-path passband coefficient ``a_real + 1j*a_imag`` as ``complex64`` (exact), shape (paths,)."""
    g = np.empty(rec.n_paths, dtype=np.complex64)
    g.real = rec.core["a_real"]
    g.imag = rec.core["a_imag"]
    return g


def path_power_db(rec: Any) -> np.ndarray:
    """Per-path power ``10*log10(a_real**2 + a_imag**2)`` in dB relative to unit power (float64), shape (paths,)."""
    re = rec.core["a_real"].astype(np.float64)
    im = rec.core["a_imag"].astype(np.float64)
    with np.errstate(divide="ignore"):
        return 10.0 * np.log10(re * re + im * im)


def delay_spread(rec: Any) -> np.ndarray:
    """Per-link power-weighted RMS delay spread of the retained paths, seconds (float64), shape (links,).

    Links whose retained paths carry zero total power give NaN.
    """
    link = _link_of_path(rec)
    n = rec.n_links
    re = rec.core["a_real"].astype(np.float64)
    im = rec.core["a_imag"].astype(np.float64)
    w = re * re + im * im
    tau = rec.core["tau"].astype(np.float64)
    total = np.bincount(link, weights=w, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(link, weights=w * tau, minlength=n) / total
        dev = tau - mean[link] if len(link) else tau
        var = np.bincount(link, weights=w * dev * dev, minlength=n) / total
    return np.sqrt(np.maximum(var, 0.0))


def _relative_delay(rec: Any, delay_reference: str) -> np.ndarray:
    if delay_reference not in _REFERENCES:
        raise ValueError(f"delay_reference must be one of {_REFERENCES}, not {delay_reference!r}")
    tau = rec.core["tau"].astype(np.float64)
    if delay_reference == "absolute" or tau.size == 0:
        return tau
    link = _link_of_path(rec)
    first = np.full(rec.n_links, np.inf)
    np.minimum.at(first, link, tau)
    return tau - first[link]


def baseband_coefficients(rec: Any, delay_reference: str = "absolute") -> np.ndarray:
    """Per-path baseband coefficients ``a * exp(-j*2*pi*fc*(tau - tau_ref))`` (complex128), shape (paths,).

    ``delay_reference="absolute"``: tau_ref = 0. ``"first_retained"``: tau_ref is each link's
    earliest retained delay (matches Sionna's default delay normalization only when the
    link's earliest path was retained). The reference changes phase only, never magnitude.
    These coefficients can be passed, with ``tau - tau_ref``, to Sionna PHY's
    ``cir_to_ofdm_channel`` / ``cir_to_time_channel``.
    """
    rel = _relative_delay(rec, delay_reference)
    a = complex_gain(rec).astype(np.complex128)
    return a * np.exp(-2j * np.pi * _frac_cycles(float(rec.frequency_hz), rel))


def frequency_response(rec: Any, offsets_hz: Any, delay_reference: str = "absolute",
                       max_elements: int = 1 << 22) -> np.ndarray:
    """Per-link frequency response at ``fc + offsets_hz`` (complex128), shape (links, len(offsets_hz)).

    H[l, k] = sum over retained paths p of link l of a_p * exp(-j*2*pi*(fc + offsets_hz[k])*(tau_p - tau_ref)).
    ``offsets_hz`` are offsets from the record's carrier (e.g. subcarrier offsets); fc always
    comes from the record. Coefficients are taken at the carrier (frequency-flat per path).
    """
    off = np.atleast_1d(np.asarray(offsets_hz, dtype=np.float64))
    if off.ndim != 1:
        raise ValueError("offsets_hz must be one-dimensional")
    rel = _relative_delay(rec, delay_reference)
    a = complex_gain(rec).astype(np.complex128)
    fc = float(rec.frequency_hz)
    carrier = a * np.exp(-2j * np.pi * _frac_cycles(fc, rel))
    out = np.zeros((rec.n_links, len(off)), dtype=np.complex128)
    offsets = np.asarray(rec.core["link_path_offsets"], dtype=np.int64)
    step = max(1, max_elements // max(1, len(off)))
    lo = 0
    n_links = rec.n_links
    while lo < n_links:  # chunk over links to bound memory (paths x frequencies)
        hi = lo + 1
        while hi < n_links and offsets[hi + 1] - offsets[lo] <= step:
            hi += 1
        p0, p1 = offsets[lo], offsets[hi]
        if p1 > p0:
            terms = carrier[p0:p1, None] * np.exp(-2j * np.pi * rel[p0:p1, None] * off[None, :])
            starts = offsets[lo:hi] - p0
            nonempty = offsets[lo + 1:hi + 1] > offsets[lo:hi]
            out[lo:hi][nonempty] = np.add.reduceat(terms, starts[nonempty], axis=0)
        lo = hi
    return out


def _frac_cycles(fc: float, tau: np.ndarray) -> np.ndarray:
    """fc*tau reduced modulo 1 in float64 (keeps the large carrier phase well conditioned)."""
    cycles = fc * tau
    return cycles - np.floor(cycles)


# ---------------------------------------------------------------------------- fixed-size arrays
def _pad(offsets: Any, values: np.ndarray, width: int, what: str) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(offsets, dtype=np.int64)
    if offsets.ndim != 1 or offsets.size == 0 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ValueError(f"{what} offsets must start at 0 and never decrease")
    if values.shape[:1] != (int(offsets[-1]),):
        raise ValueError(f"expected {int(offsets[-1])} values along the first axis, got {values.shape[:1]}")
    counts = np.diff(offsets)
    if counts.size and int(counts.max()) > width:
        raise ValueError(f"a row holds {int(counts.max())} {what}, more than the padded width {width}")
    rows = np.repeat(np.arange(counts.size), counts)
    cols = np.arange(int(offsets[-1])) - np.repeat(offsets[:-1], counts)
    padded = np.zeros((counts.size, width) + values.shape[1:], values.dtype)
    padded[rows, cols] = values
    mask = np.arange(width) < counts[:, None]
    return padded, mask


def pad_paths(core: Mapping[str, Any], values: str | Any, max_paths: int = MAX_PATHS
              ) -> tuple[np.ndarray, np.ndarray]:
    """A per-path array as one row per link: ``(links, max_paths, ...)`` plus a boolean mask.

    ``core`` is any mapping with ``link_path_offsets``: ``record.core``, an opened NPZ, or an
    HF Datasets row with restored shapes. ``values`` names a core array (``"tau"``) or is an
    array with one entry per path, such as the output of :func:`pad_interactions`, which gives
    ``(links, max_paths, max_interactions, ...)``. Row ``i`` belongs to UE
    ``core["link_ue_index"][i]``. Padded entries are zero; use the mask to tell them apart.
    Nothing is truncated: a link with more paths than ``max_paths`` raises ``ValueError``.
    """
    arr = np.asarray(core[values]) if isinstance(values, str) else np.asarray(values)
    return _pad(core["link_path_offsets"], arr, max_paths, "paths")


def pad_interactions(detail: Mapping[str, Any], values: str | Any,
                     max_interactions: int = MAX_INTERACTIONS) -> tuple[np.ndarray, np.ndarray]:
    """A per-interaction array as one row per path: ``(paths, max_interactions, ...)`` plus a mask.

    ``detail`` is ``record.detail``, ``link.detail``, an opened detail NPZ or a restored HF row;
    rows follow core's path order. ``"interaction_vertex"`` gives ``(paths, 4, 3)``. Pass the
    result to :func:`pad_paths` for one row per link.
    """
    arr = np.asarray(detail[values]) if isinstance(values, str) else np.asarray(values)
    return _pad(detail["path_interaction_offsets"], arr, max_interactions, "interactions")
