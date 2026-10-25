# Channel coefficient conventions

*The published `schema/conventions.json` is authoritative for the fields it declares; this
page explains how to use them.*

**Coefficients.** `a_real + 1j*a_imag` is the passband path coefficient `Paths.a` from
Sionna RT, stored without modification. It does **not** include the carrier-delay phase
exp(-j2π·fc·τ). `tau` is the absolute propagation delay in seconds, also unmodified. Each
record's `frequency_hz` equals the carrier used for that record's ray tracing.

The channels were computed with Sionna RT 2.0.1, whose `Paths.a` excludes the carrier
phase (Sionna adds it in `Paths.cir()` and `Paths.cfr()`). The stored values are `Paths.a`
and `Paths.tau` as computed: no carrier multiplication and no delay normalization.

**Antennas.** Every BS and UE uses one isotropic, vertically polarized element (V/V). The
pattern and polarization are already included in each coefficient; applying another
element pattern or polarization on top would count them twice.

**Retained paths only.** Each link keeps at most its 16 strongest paths above -150 dB
(relative to unit power). Transforms describe these retained paths; they do not
reconstruct discarded paths, and links that are not retained stay absent.

**Path order and ranks.** Within a link, paths are stored by decreasing power computed in
float32; recomputing power in float64 can swap paths whose powers differ by less than
float32 resolution. The published `schema/conventions.json` defines `path_rank` as the
rank before the -150 dB threshold. The threshold removes only the weakest candidates, so
the kept ranks are always the first ones and `path_rank` equals the path's 0-based
position within its link; `interaction_rank` is likewise the 0-based position along the
path. In all of Tiny both ranks are contiguous and start at 0. Ranks order paths within
one record only: path 0 at 3.5 GHz and path 0 at 7 GHz need not be the same route.

**Transforms** (`channels4world.transforms`):

| Function | Result |
|---|---|
| `complex_gain(rec)` | stored passband coefficient, `complex64` |
| `path_power_db(rec)` | 10·log10(\|a\|²), dB relative to unit power |
| `delay_spread(rec)` | per-link power-weighted RMS delay over retained paths |
| `baseband_coefficients(rec, delay_reference="absolute")` | a·exp(-j2π·fc·(τ − τ_ref)); input for Sionna PHY `cir_to_ofdm_channel` / `cir_to_time_channel` |
| `frequency_response(rec, offsets_hz, delay_reference="absolute")` | Σ a·exp(-j2π(fc + Δf)(τ − τ_ref)) with Δf the offset from the carrier; fc always comes from the record |
| `pad_paths(core, name, max_paths=16)` | one row per link, zero padding and a mask; no truncation |
| `pad_interactions(detail, name, max_interactions=4)` | one row per path, zero padding and a mask; no truncation |

`delay_reference="first_retained"` subtracts each link's earliest retained delay. It
matches Sionna's default delay normalization only when the earliest path was retained. In
every case it changes phase only, never magnitude.

**Precision.** Phases are computed in float64 from the stored float32 delays: roughly
10⁻³ rad at microsecond delays and 10⁻² rad near 10 µs at 7 GHz.
