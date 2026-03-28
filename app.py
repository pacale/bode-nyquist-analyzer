"""
Analizzatore Interattivo Bode & Nyquist
=======================================
Applicazione Streamlit per l'analisi interattiva di funzioni di trasferimento
G(s). Fornisce diagrammi di Bode (esatto + approssimato) e diagrammi polari
di Nyquist usando SymPy per il parsing simbolico e python-control per la
risposta in frequenza.
"""

from __future__ import annotations

import re as _re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import control as ctrl  # type: ignore
import numpy as np
import pandas as pd
import plotly.graph_objects as go  # type: ignore
import streamlit as st
import sympy
from plotly.subplots import make_subplots  # type: ignore

# ---------------------------------------------------------------------------
# Configurazione pagina (deve essere il primo comando Streamlit)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Analizzatore Bode & Nyquist",
    page_icon="📈",
    layout="wide",
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
_EXACT_COLOR = "#1f77b4"
_APPROX_COLOR = "#ff7f0e"
_EXACT_WIDTH = 2.0
_APPROX_WIDTH = 1.5
_CURSOR_COLOR = "red"
_QUERY_COLOR = "green"


# ---------------------------------------------------------------------------
# Contenitore dati
# ---------------------------------------------------------------------------
@dataclass
class SystemInfo:
    """Contenitore per le informazioni della funzione di trasferimento."""

    expr: sympy.Expr
    num_expr: sympy.Expr
    den_expr: sympy.Expr
    zeros: list[complex]
    poles: list[complex]
    num_coeffs: list[float]
    den_coeffs: list[float]
    order: int
    system_type: int
    static_gain: Optional[float]
    tf: ctrl.TransferFunction
    break_freqs_zeros: list[float] = field(default_factory=list)
    break_freqs_poles: list[float] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
# 1. PARSING
# ═══════════════════════════════════════════════════════════════════════════

def parse_transfer_function(expr_str: str) -> SystemInfo:
    """Parsa una stringa G(s) e restituisce un oggetto *SystemInfo*.

    Solleva *ValueError* in caso di errore.
    """
    s = sympy.Symbol("s")

    try:
        expr = sympy.sympify(expr_str, locals={"s": s})
    except (sympy.SympifyError, SyntaxError, TypeError) as exc:
        raise ValueError(
            f"Espressione non valida. Controlla la sintassi.\n"
            f"Dettaglio: {exc}"
        ) from exc

    num_expr, den_expr = sympy.fraction(sympy.cancel(expr))

    try:
        zeros_sym = sympy.solve(num_expr, s)
        poles_sym = sympy.solve(den_expr, s)
    except Exception as exc:
        raise ValueError(
            f"Impossibile calcolare poli/zeri: {exc}"
        ) from exc

    zeros = [complex(z) for z in zeros_sym]
    poles = [complex(p) for p in poles_sym]

    try:
        num_poly = sympy.Poly(sympy.expand(num_expr), s)
        den_poly = sympy.Poly(sympy.expand(den_expr), s)
    except sympy.GeneratorsNeeded:
        num_poly = sympy.Poly(sympy.expand(num_expr), s, domain="ZZ")
        den_poly = sympy.Poly(sympy.expand(den_expr), s, domain="ZZ")

    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]

    order = den_poly.degree()
    system_type = sum(1 for p in poles if abs(p) < 1e-10)

    try:
        gain_sym = sympy.limit(expr, s, 0)
        if gain_sym.is_finite:
            static_gain = float(abs(complex(gain_sym)))
        else:
            static_gain = None
    except Exception:
        static_gain = None

    tf = ctrl.TransferFunction(num_coeffs, den_coeffs)
    bz, bp = compute_break_frequencies(zeros, poles)

    return SystemInfo(
        expr=expr,
        num_expr=num_expr,
        den_expr=den_expr,
        zeros=zeros,
        poles=poles,
        num_coeffs=num_coeffs,
        den_coeffs=den_coeffs,
        order=order,
        system_type=system_type,
        static_gain=static_gain,
        tf=tf,
        break_freqs_zeros=bz,
        break_freqs_poles=bp,
    )


# ═══════════════════════════════════════════════════════════════════════════
# 2. FREQUENZE DI ROTTURA
# ═══════════════════════════════════════════════════════════════════════════

def compute_break_frequencies(
    zeros: list[complex],
    poles: list[complex],
) -> tuple[list[float], list[float]]:
    """Restituisce liste ordinate di frequenze di rottura finite e non nulle."""
    bz = sorted({abs(z) for z in zeros if 1e-10 < abs(z) < 1e15})
    bp = sorted({abs(p) for p in poles if 1e-10 < abs(p) < 1e15})
    return bz, bp


def _compute_omega_range(
    info: SystemInfo, n_points: int = 500,
) -> np.ndarray:
    """Vettore ω log-spaziato auto-ristretto a [min_break/20, max_break×20]."""
    all_breaks = sorted(
        set(info.break_freqs_zeros) | set(info.break_freqs_poles),
    )
    if all_breaks:
        omega_min = all_breaks[0] / 20.0
        omega_max = all_breaks[-1] * 20.0
    else:
        omega_min, omega_max = 0.01, 1000.0
    return np.logspace(np.log10(omega_min), np.log10(omega_max), n_points)


# ═══════════════════════════════════════════════════════════════════════════
# 3. BODE APPROSSIMATO (modulo + fase)
# ═══════════════════════════════════════════════════════════════════════════

def compute_asymptotic_bode(
    omega: np.ndarray,
    info: SystemInfo,
) -> tuple[np.ndarray, np.ndarray]:
    """Restituisce *(approx_mag_dB, approx_phase_deg)*."""

    # ── Modulo ────────────────────────────────────────────────────────────
    events: list[tuple[float, float]] = []
    for z in info.zeros:
        wz = abs(z)
        if wz > 1e-10:
            events.append((wz, +20.0))
    for p in info.poles:
        wp = abs(p)
        if wp > 1e-10:
            events.append((wp, -20.0))
    events.sort(key=lambda e: e[0])

    n_zeros_origin = sum(1 for z in info.zeros if abs(z) < 1e-10)
    n_poles_origin = sum(1 for p in info.poles if abs(p) < 1e-10)
    initial_slope = 20.0 * (n_zeros_origin - n_poles_origin)

    try:
        resp0 = info.tf(1j * omega[0])
        gain_at_w0 = 20.0 * np.log10(max(abs(resp0), 1e-30))
    except Exception:
        gain_at_w0 = 0.0

    log_omega = np.log10(omega)
    log_w0 = log_omega[0]
    bp_list = [(np.log10(wb), ds) for wb, ds in events]

    mag_db = np.empty_like(log_omega)
    for i, lw in enumerate(log_omega):
        db = gain_at_w0
        slope = initial_slope
        prev = log_w0
        for log_wb, ds in bp_list:
            if lw <= log_wb:
                db += slope * (lw - prev)
                break
            db += slope * (log_wb - prev)
            slope += ds
            prev = log_wb
        else:
            db += slope * (lw - prev)
        mag_db[i] = db

    # ── Fase (gradi) ──────────────────────────────────────────────────────
    phase = np.zeros_like(log_omega)
    for _ in range(n_poles_origin):
        phase -= 90.0
    for _ in range(n_zeros_origin):
        phase += 90.0

    for p in info.poles:
        wp = abs(p)
        if wp < 1e-10:
            continue
        lw_b = np.log10(wp)
        for i, lw in enumerate(log_omega):
            if lw < lw_b - 1:
                pass
            elif lw > lw_b + 1:
                phase[i] -= 90.0
            else:
                phase[i] -= 45.0 * (lw - (lw_b - 1))

    for z in info.zeros:
        wz = abs(z)
        if wz < 1e-10:
            continue
        lw_b = np.log10(wz)
        for i, lw in enumerate(log_omega):
            if lw < lw_b - 1:
                pass
            elif lw > lw_b + 1:
                phase[i] += 90.0
            else:
                phase[i] += 45.0 * (lw - (lw_b - 1))

    return mag_db, phase


# ═══════════════════════════════════════════════════════════════════════════
# 4. NYQUIST APPROSSIMATO
# ═══════════════════════════════════════════════════════════════════════════

def compute_asymptotic_nyquist(
    omega: np.ndarray,
    info: SystemInfo,
) -> np.ndarray:
    """Restituisce G(jω) approssimato tramite fattori del primo ordine."""
    n = len(omega)
    mag_total = np.ones(n)
    phase_total = np.zeros(n)  # gradi

    # Guadagno statico (rapporto coefficienti costanti)
    try:
        dc_gain = abs(info.num_coeffs[-1] / info.den_coeffs[-1])
    except (ZeroDivisionError, IndexError):
        dc_gain = 1.0
    mag_total *= dc_gain

    log_omega = np.log10(omega)

    n_zeros_origin = sum(1 for z in info.zeros if abs(z) < 1e-10)
    n_poles_origin = sum(1 for p in info.poles if abs(p) < 1e-10)

    # Poli/zeri nell'origine
    for _ in range(n_poles_origin):
        mag_total /= omega
        phase_total -= 90.0
    for _ in range(n_zeros_origin):
        mag_total *= omega
        phase_total += 90.0

    # Poli non all'origine
    for p in info.poles:
        wp = abs(p)
        if wp < 1e-10:
            continue
        lw_b = np.log10(wp)
        for i, (w, lw) in enumerate(zip(omega, log_omega)):
            if w >= wp:
                mag_total[i] *= (wp / w)
            if lw < lw_b - 1:
                pass
            elif lw > lw_b + 1:
                phase_total[i] -= 90.0
            else:
                phase_total[i] -= 45.0 * (lw - (lw_b - 1))

    # Zeri non all'origine
    for z in info.zeros:
        wz = abs(z)
        if wz < 1e-10:
            continue
        lw_b = np.log10(wz)
        for i, (w, lw) in enumerate(zip(omega, log_omega)):
            if w >= wz:
                mag_total[i] *= (w / wz)
            if lw < lw_b - 1:
                pass
            elif lw > lw_b + 1:
                phase_total[i] += 90.0
            else:
                phase_total[i] += 45.0 * (lw - (lw_b - 1))

    phase_rad = np.deg2rad(phase_total)
    return mag_total * (np.cos(phase_rad) + 1j * np.sin(phase_rad))


# ═══════════════════════════════════════════════════════════════════════════
# 5. DIAGRAMMA DI BODE (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def plot_bode(
    omega: np.ndarray,
    mag_db: np.ndarray,
    phase_deg: np.ndarray,
    approx_mag_db: np.ndarray,
    approx_phase_deg: np.ndarray,
    info: SystemInfo,
    phase_in_radians: bool = False,
    cursor_omega: Optional[float] = None,
) -> go.Figure:
    """Due sottografi impilati: modulo (dB) e fase."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Modulo (dB)", "Fase"),
    )

    # ── Modulo ────────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=omega, y=mag_db, mode="lines", name="Esatta",
        line=dict(color=_EXACT_COLOR, width=_EXACT_WIDTH),
        legendgroup="exact", showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=omega, y=approx_mag_db, mode="lines", name="Approssimata",
        line=dict(color=_APPROX_COLOR, width=_APPROX_WIDTH, dash="dash"),
        legendgroup="approx", showlegend=True,
    ), row=1, col=1)

    # ── Intercetta sinistra ───────────────────────────────────────────────
    omega_min = omega[0]
    val_esatta = mag_db[0]
    val_approx = approx_mag_db[0]
    
    fig.add_annotation(
        x=np.log10(omega_min),
        y=val_esatta,
        xref="x", yref="y1",
        text=f"<b>{val_esatta:.2f} dB</b>",
        showarrow=True, arrowhead=2, arrowcolor=_EXACT_COLOR,
        ax=-50, ay=0,
        font=dict(color=_EXACT_COLOR, size=12),
        bgcolor="white", bordercolor=_EXACT_COLOR, borderwidth=1, borderpad=4,
        xanchor="right"
    )
    fig.add_trace(go.Scatter(
        x=[omega_min], y=[val_esatta], mode="markers",
        marker=dict(symbol="circle", size=8, color=_EXACT_COLOR),
        name="Intercetta sinistra (esatta)", showlegend=False
    ), row=1, col=1)

    fig.add_annotation(
        x=np.log10(omega_min),
        y=val_approx,
        xref="x", yref="y1",
        text=f"<b>{val_approx:.2f} dB</b>",
        showarrow=True, arrowhead=2, arrowcolor=_APPROX_COLOR,
        ax=-50, ay=20,
        font=dict(color=_APPROX_COLOR, size=12),
        bgcolor="white", bordercolor=_APPROX_COLOR, borderwidth=1, borderpad=4,
        xanchor="right"
    )
    fig.add_trace(go.Scatter(
        x=[omega_min], y=[val_approx], mode="markers",
        marker=dict(symbol="circle", size=8, color=_APPROX_COLOR),
        name="Intercetta sinistra (appross.)", showlegend=False
    ), row=1, col=1)

    # ── Fase ──────────────────────────────────────────────────────────────
    if phase_in_radians:
        exact_phase_plot = phase_deg / 180.0
        approx_phase_plot = approx_phase_deg / 180.0
        phase_label = "Fase (×π rad)"
    else:
        exact_phase_plot = phase_deg
        approx_phase_plot = approx_phase_deg
        phase_label = "Fase (°)"

    fig.add_trace(go.Scatter(
        x=omega, y=exact_phase_plot, mode="lines", name="Esatta",
        line=dict(color=_EXACT_COLOR, width=_EXACT_WIDTH),
        legendgroup="exact", showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=omega, y=approx_phase_plot, mode="lines", name="Approssimata",
        line=dict(color=_APPROX_COLOR, width=_APPROX_WIDTH, dash="dash"),
        legendgroup="approx", showlegend=False,
    ), row=2, col=1)

    # ── Annotazioni verticali alle frequenze di rottura ───────────────────
    all_breaks = sorted(
        set(info.break_freqs_zeros) | set(info.break_freqs_poles),
    )
    for wb in all_breaks:
        label = f"ωr = {wb:.4g} rad/s"
        for row in (1, 2):
            fig.add_vline(
                x=wb, line_width=1, line_dash="dot",
                line_color=_APPROX_COLOR,
                annotation_text=label if row == 1 else "",
                annotation_position="top right",
                annotation_font_size=10,
                row=row, col=1,
            )

    # ── Cursore frequenza ─────────────────────────────────────────────────
    if cursor_omega is not None:
        idx = int(np.argmin(np.abs(omega - cursor_omega)))
        wc = omega[idx]

        # Marcatore modulo
        fig.add_trace(go.Scatter(
            x=[wc], y=[mag_db[idx]],
            mode="markers",
            marker=dict(color=_CURSOR_COLOR, size=10, symbol="circle"),
            name=f"ω = {wc:.4g}",
            showlegend=False,
        ), row=1, col=1)

        # Marcatore fase
        fig.add_trace(go.Scatter(
            x=[wc], y=[exact_phase_plot[idx]],
            mode="markers",
            marker=dict(color=_CURSOR_COLOR, size=10, symbol="circle"),
            name=f"ω = {wc:.4g}",
            showlegend=False,
        ), row=2, col=1)

        # Linea verticale tratteggiata rossa su entrambi i sottografi
        for row in (1, 2):
            fig.add_vline(
                x=wc, line_width=1.5, line_dash="dash",
                line_color=_CURSOR_COLOR,
                row=row, col=1,
            )

    # ── Assi ──────────────────────────────────────────────────────────────
    log_min = np.log10(omega[0])
    log_max = np.log10(omega[-1])

    fig.update_xaxes(type="log", range=[log_min, log_max], row=1, col=1)
    fig.update_xaxes(
        type="log", range=[log_min, log_max],
        title_text="ω [rad/s]",
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="red", spikethickness=1, spikedash="dot",
        row=2, col=1,
    )

    # Y modulo: ±5 dB dalla curva esatta – auto-range disabilitato
    y_mag_min = float(np.nanmin(mag_db))
    y_mag_max = float(np.nanmax(mag_db))
    fig.update_yaxes(
        title_text="Modulo (dB)",
        range=[y_mag_min - 5, y_mag_max + 5],
        showspikes=True, spikecolor="gray", spikethickness=1, spikedash="dot",
        row=1, col=1,
    )

    # Y fase
    if phase_in_radians:
        ph_min = float(np.nanmin(exact_phase_plot))
        ph_max = float(np.nanmax(exact_phase_plot))
        tickvals = [
            -2, -1.75, -1.5, -1.25, -1, -0.75, -0.5, -0.25,
            0,
            0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 2,
        ]
        ticktext = [
            "-2π", "-7π/4", "-3π/2", "-5π/4", "-π", "-3π/4",
            "-π/2", "-π/4", "0",
            "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4", "2π",
        ]
        fig.update_yaxes(
            title_text=phase_label,
            range=[ph_min - 0.25, ph_max + 0.25],
            tickvals=tickvals,
            ticktext=ticktext,
            showspikes=True, spikecolor="gray",
            spikethickness=1, spikedash="dot",
            row=2, col=1,
        )
    else:
        ph_min = float(np.nanmin(exact_phase_plot))
        ph_max = float(np.nanmax(exact_phase_plot))
        tv = list(np.arange(-360, 361, 45))
        tt = [f"{int(v)}°" for v in tv]
        fig.update_yaxes(
            title_text=phase_label,
            range=[ph_min - 20, ph_max + 20],
            tickvals=tv,
            ticktext=tt,
            showspikes=True, spikecolor="gray",
            spikethickness=1, spikedash="dot",
            row=2, col=1,
        )

    # ── Hover spikes ──────────────────────────────────────────────────────
    fig.update_xaxes(
        showspikes=True, spikemode="across", spikesnap="cursor",
        spikecolor="red", spikethickness=1, spikedash="dot",
        row=1, col=1,
    )
    fig.update_yaxes(
        showspikes=True, spikecolor="gray",
        spikethickness=1, spikedash="dot",
        row=1, col=1,
    )



    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        height=750,
        template="plotly_white",
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="x unified",
        hoverdistance=50,
        legend=dict(
            orientation="v", yanchor="top", y=0.99,
            xanchor="right", x=0.99,
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. DIAGRAMMA POLARE / NYQUIST (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def plot_nyquist(
    omega: np.ndarray,
    resp: np.ndarray,
    approx_resp: np.ndarray,
    cursor_omega: Optional[float] = None,
    cursor_resp: Optional[complex] = None,
) -> go.Figure:
    """Diagramma polare con traccia esatta e approssimata."""
    re_e, im_e = resp.real, resp.imag
    re_a, im_a = approx_resp.real, approx_resp.imag

    fig = go.Figure()

    # Traccia esatta
    fig.add_trace(go.Scatter(
        x=re_e, y=im_e, mode="lines", name="Nyquist Esatto",
        line=dict(color=_EXACT_COLOR, width=_EXACT_WIDTH),
        hovertemplate="Re: %{x:.4f}<br>Im: %{y:.4f}<extra></extra>",
    ))

    # Traccia approssimata
    fig.add_trace(go.Scatter(
        x=re_a, y=im_a, mode="lines", name="Nyquist Approssimato",
        line=dict(color=_APPROX_COLOR, width=_APPROX_WIDTH, dash="dash"),
        hovertemplate="Re: %{x:.4f}<br>Im: %{y:.4f}<extra></extra>",
    ))

    # Frecce direzionali ogni 50 punti sulla traccia esatta
    step = 50
    for idx in range(step, len(omega) - 1, step):
        dx = re_e[idx + 1] - re_e[idx]
        dy = im_e[idx + 1] - im_e[idx]
        fig.add_annotation(
            x=re_e[idx], y=im_e[idx],
            ax=re_e[idx] - dx * 8,
            ay=im_e[idx] - dy * 8,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5,
            arrowwidth=1.5, arrowcolor=_EXACT_COLOR,
        )

    # Inizio (ω → 0)
    fig.add_trace(go.Scatter(
        x=[re_e[0]], y=[im_e[0]],
        mode="markers+text",
        marker=dict(color="green", size=12, symbol="circle"),
        text=["ω→0"], textposition="top right",
        textfont=dict(size=13, color="green"),
        name="ω→0",
    ))

    # Fine (ω → +∞)
    fig.add_trace(go.Scatter(
        x=[re_e[-1]], y=[im_e[-1]],
        mode="markers+text",
        marker=dict(color="red", size=12, symbol="square"),
        text=["ω→+∞"], textposition="top right",
        textfont=dict(size=13, color="red"),
        name="ω→+∞",
    ))

    # Punto critico (−1, 0)
    fig.add_trace(go.Scatter(
        x=[-1], y=[0],
        mode="markers+text",
        marker=dict(color="black", size=12, symbol="x"),
        text=["(-1, 0)"], textposition="bottom right",
        textfont=dict(size=11, color="black"),
        name="Punto critico (−1, 0)",
    ))

    # Cursore frequenza
    if cursor_resp is not None and cursor_omega is not None:
        fig.add_trace(go.Scatter(
            x=[cursor_resp.real], y=[cursor_resp.imag],
            mode="markers+text",
            marker=dict(color=_CURSOR_COLOR, size=12, symbol="circle"),
            text=[f"ω = {cursor_omega:.4g}"],
            textposition="top right",
            textfont=dict(size=11, color=_CURSOR_COLOR),
            name=f"ω = {cursor_omega:.4g} rad/s",
        ))

    fig.update_layout(
        xaxis_title="Re{G(jω)}",
        yaxis_title="Im{G(jω)}",
        height=600,
        template="plotly_white",
        yaxis_scaleanchor="x",
        showlegend=True,
        hovermode="closest",
        legend=dict(
            orientation="v", yanchor="top", y=0.99,
            xanchor="right", x=0.99,
        ),
        margin=dict(l=60, r=40, t=30, b=60),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. CURSORE FREQUENZA — metriche
# ═══════════════════════════════════════════════════════════════════════════

def render_cursor_metrics(
    wc: float,
    resp_exact: complex,
    phase_in_radians: bool,
) -> None:
    """Mostra 4 metriche in riga per la frequenza selezionata dal cursore."""
    mag_db = 20.0 * np.log10(max(abs(resp_exact), 1e-30))
    phase_deg = np.degrees(np.angle(resp_exact))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ω", f"{wc:.4f} rad/s")
    c2.metric("|G(jω)|", f"{mag_db:.2f} dB")

    if phase_in_radians:
        phase_pi = phase_deg / 180.0
        c3.metric("∠G(jω)", f"{phase_pi:.4f}π")
    else:
        c3.metric("∠G(jω)", f"{phase_deg:.2f}°")

    c4.metric("G(jω)", f"{resp_exact.real:.4f} {resp_exact.imag:+.4f}j")


# ═══════════════════════════════════════════════════════════════════════════
# 7b. INTERROGAZIONE PUNTUALE — funzione di calcolo
# ═══════════════════════════════════════════════════════════════════════════

def query_single_frequency(
    omega: float,
    info: SystemInfo,
    omega_min: float,
    omega_max: float,
) -> dict:
    """Calcola le grandezze esatte e approssimate a una singola frequenza.

    Restituisce un dizionario con tutte le grandezze richieste.
    """
    # Esatto via python-control
    g_exact = info.tf(1j * omega)
    mag_exact = abs(g_exact)
    mag_exact_dB = 20.0 * np.log10(max(mag_exact, 1e-30))
    phase_exact_deg = float(np.degrees(np.angle(g_exact)))
    phase_exact_pi = phase_exact_deg / 180.0

    # Approssimato: calcola il Bode approssimato sull'intero range e poi
    # interpola al punto desiderato.  Questo garantisce l'ancoraggio
    # corretto del guadagno alla prima frequenza del range (come nel
    # diagramma disegnato).
    omega_full = _compute_omega_range(info, n_points=500)
    approx_mag_full, approx_phase_full = compute_asymptotic_bode(
        omega_full, info,
    )
    mag_approx_dB = float(np.interp(
        np.log10(omega), np.log10(omega_full), approx_mag_full,
    ))
    phase_approx_deg = float(np.interp(
        np.log10(omega), np.log10(omega_full), approx_phase_full,
    ))
    mag_approx = 10.0 ** (mag_approx_dB / 20.0)
    phase_approx_pi = phase_approx_deg / 180.0
    # Ricostruisci il fasore approssimato
    g_approx = mag_approx * (
        np.cos(np.deg2rad(phase_approx_deg))
        + 1j * np.sin(np.deg2rad(phase_approx_deg))
    )

    return {
        "omega": omega,
        "omega_min": omega_min,
        "omega_max": omega_max,
        "in_range": omega_min <= omega <= omega_max,
        # Esatto
        "g_exact": g_exact,
        "mag_exact": mag_exact,
        "mag_exact_dB": mag_exact_dB,
        "phase_exact_deg": phase_exact_deg,
        "phase_exact_pi": phase_exact_pi,
        # Approssimato
        "g_approx": g_approx,
        "mag_approx": mag_approx,
        "mag_approx_dB": mag_approx_dB,
        "phase_approx_deg": phase_approx_deg,
        "phase_approx_pi": phase_approx_pi,
        # Scarti
        "delta_mag_dB": abs(mag_exact_dB - mag_approx_dB),
        "delta_phase_deg": abs(phase_exact_deg - phase_approx_deg),
        "delta_phase_pi": abs(phase_exact_pi - phase_approx_pi),
    }


def _render_query_table(res: dict) -> None:
    """Renderizza la tabella di confronto esatta/approssimata (semplificata)."""
    import pandas as pd
    
    mag_exact_dB = res["mag_exact_dB"]
    mag_approx_dB = res["mag_approx_dB"]
    
    if st.session_state.phase_unit == "Radianti (π)":
        ph_exact = f"{res['phase_exact_deg'] / 180.0:.4f}π rad"
        ph_approx = f"{res['phase_approx_deg'] / 180.0:.4f}π rad"
    else:
        ph_exact = f"{res['phase_exact_deg']:.4f}°"
        ph_approx = f"{res['phase_approx_deg']:.4f}°"

    rows = [
        {
            "Grandezza": "Modulo",
            "Curva Esatta": f"{mag_exact_dB:.4f} dB",
            "Curva Approssimata": f"{mag_approx_dB:.4f} dB",
        },
        {
            "Grandezza": "Fase",
            "Curva Esatta": ph_exact,
            "Curva Approssimata": ph_approx,
        }
    ]
    df = pd.DataFrame(rows)
    st.table(df.set_index("Grandezza"))


# ═══════════════════════════════════════════════════════════════════════════
# 8. SIDEBAR — Pannello Informazioni
# ═══════════════════════════════════════════════════════════════════════════

def _to_readable(expr: sympy.Expr) -> str:
    """Espressione SymPy → stringa polinomiale leggibile (es. "s² + 3·s + 2")."""
    superscripts = str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹")
    text = str(sympy.expand(expr))
    text = _re.sub(
        r"\*\*(\d+)",
        lambda m: m.group(1).translate(superscripts),
        text,
    )
    text = text.replace("*", "·")
    return text


def format_tf_forms(zeros: list[complex], poles: list[complex], K: float) -> dict[str, str]:
    """Genera le tre forme standard della funzione di trasferimento."""
    import sympy
    s = sympy.Symbol('s')
    
    def _fmt(r: complex, form: str) -> str:
        if abs(r) < 1e-10:
            return "s"
        if abs(r.imag) < 1e-10:
            a = r.real
            if form == "evans":
                return f"(s {'+' if a <= 0 else '-'} {abs(a):.4g})"
            else:
                return f"(1 {'+' if a <= 0 else '-'} s/{abs(a):.4g})"
        else:
            r_str = f"{r.real:.4g}{r.imag:+.4g}j"
            if form == "evans":
                return f"(s - ({r_str}))"
            else:
                return f"(1 - s/({r_str}))"

    num_evans = " · ".join([_fmt(z, "evans") for z in zeros]) or "1"
    den_evans = " · ".join([_fmt(p, "evans") for p in poles]) or "1"
    
    K_bode = complex(K)
    for z in zeros:
        if abs(z) > 1e-10: K_bode *= (-z)
    for p in poles:
        if abs(p) > 1e-10: K_bode /= (-p)
        
    K_str = f"{K:.4g}" if abs(K.imag) < 1e-10 else f"({K.real:.4g}{K.imag:+.4g}j)"
    Kb_str = f"{K_bode.real:.4g}" if abs(K_bode.imag) < 1e-10 else f"({K_bode.real:.4g}{K_bode.imag:+.4g}j)"
    
    num_bode = " · ".join([_fmt(z, "bode") for z in zeros]) or "1"
    den_bode = " · ".join([_fmt(p, "bode") for p in poles]) or "1"
    
    n_expr = sympy.sympify(K)
    for z in zeros: n_expr *= (s - z)
    d_expr = sympy.sympify(1.0)
    for p in poles: d_expr *= (s - p)
    
    return {
        "bode": f"{Kb_str} · ({num_bode}) / ({den_bode})" if num_bode != "1" else f"{Kb_str} / ({den_bode})",
        "evans": f"{K_str} · ({num_evans}) / ({den_evans})" if num_evans != "1" else f"{K_str} / ({den_evans})",
        "poly": f"({_to_readable(n_expr)}) / ({_to_readable(d_expr)})"
    }


def _show_sidebar_info(info: SystemInfo) -> None:
    """Visualizza le informazioni del sistema nella sidebar."""
    st.sidebar.header("📋 Informazioni Sistema")

    # G(s) come frazione HTML
    st.sidebar.subheader("G(s)")
    num_str = _to_readable(info.num_expr)
    den_str = _to_readable(info.den_expr)
    st.sidebar.markdown(f"""
    <div style="display:flex; flex-direction:column; align-items:center;
                text-align:center; font-family:monospace; font-size:1.1rem;
                margin:0.5rem 0 1rem 0;">
        <span>{num_str}</span>
        <hr style="width:90%; border:none; border-top:2px solid #555;
                   margin:2px 0;" />
        <span>{den_str}</span>
    </div>
    """, unsafe_allow_html=True)

    # Converti K da SymPy
    try:
        K_evans = float(info.num_coeffs[0] / info.den_coeffs[0]) if info.num_coeffs and info.den_coeffs else 1.0
    except Exception:
        K_evans = 1.0
    
    forms = format_tf_forms(info.zeros, info.poles, K_evans)
    with st.sidebar.expander("Visualizza in tutte le forme"):
        st.markdown("**Forma di Bode:**")
        st.code(forms["bode"], language=None)
        st.markdown("**Forma di Evans:**")
        st.code(forms["evans"], language=None)
        st.markdown("**Forma Polinomiale:**")
        st.code(forms["poly"], language=None)
    # SymPy normalizza automaticamente tutte le forme algebriche equivalenti

    # Zeri
    st.sidebar.subheader("Zeri")
    if info.zeros:
        for idx, z in enumerate(info.zeros, 1):
            if abs(z.imag) > 1e-10:
                st.sidebar.text_input(
                    f"Zero {idx}",
                    value=f"{z.real:.4f}{z.imag:+.4f}j",
                    disabled=True, key=f"z_{idx}",
                )
            else:
                st.sidebar.metric(f"Zero {idx}", f"{z.real:.4f}")
    else:
        st.sidebar.write("Nessuno")

    # Poli
    st.sidebar.subheader("Poli")
    if info.poles:
        for idx, p in enumerate(info.poles, 1):
            if abs(p.imag) > 1e-10:
                st.sidebar.text_input(
                    f"Polo {idx}",
                    value=f"{p.real:.4f}{p.imag:+.4f}j",
                    disabled=True, key=f"p_{idx}",
                )
            else:
                st.sidebar.metric(f"Polo {idx}", f"{p.real:.4f}")
    else:
        st.sidebar.write("Nessuno")

    # Scalari
    st.sidebar.metric("Ordine del Sistema", int(info.order))
    st.sidebar.metric("Tipo del Sistema", int(info.system_type))

    if info.static_gain is None:
        st.sidebar.metric(
            "Guadagno Statico K",
            "∞" if info.system_type > 0 else "0",
        )
    elif info.static_gain == 0.0:
        st.sidebar.metric("Guadagno Statico K", "0")
    else:
        st.sidebar.metric("Guadagno Statico K", round(info.static_gain, 4))


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Punto di ingresso principale dell'applicazione."""
    st.title("📈 Analizzatore Interattivo Bode & Nyquist")
    st.markdown(
        "Inserisci il **numeratore** e il **denominatore** di G(s) qui sotto, "
        "poi premi **Analizza**."
    )

    # ── Inizializza session_state ─────────────────────────────────────────
    if "phase_unit" not in st.session_state:
        st.session_state.phase_unit = "Gradi (°)"
    if "cursor_omega" not in st.session_state:
        st.session_state.cursor_omega = None
    if "analyzed" not in st.session_state:
        st.session_state.analyzed = False

    # ── Input a frazione visiva ───────────────────────────────────────────
    forma = st.radio(
        "Forma di inserimento",
        options=["Forma di Bode  →  (1 + s/ω)", "Forma di Evans  →  (s + a)", "Forma Polinomiale  →  as² + bs + c"],
        horizontal=True,
        key="forma_inserimento"
    )

    if "Bode" in forma:
        def_num = "s*(1+s/10)"
        def_den = "(1+s)*(1+s/100)"
        f_key = "bode"
    elif "Evans" in forma:
        def_num = "s*(s+10)"
        def_den = "(s+1)*(s+100)"
        f_key = "evans"
    else:
        def_num = "s**2 + 10*s"
        def_den = "s**2 + 101*s + 100"
        f_key = "poly"
        
    num_str = st.text_input(
        "Numeratore N(s)",
        value=def_num,
        key=f"num_{f_key}",
        help="Sintassi Python/SymPy: * per la moltiplicazione, ** per le potenze.",
    )
    st.markdown("<hr style='margin:4px 0'>", unsafe_allow_html=True)
    den_str = st.text_input(
        "Denominatore D(s)",
        value=def_den,
        key=f"den_{f_key}",
        help="Sintassi Python/SymPy. Lascia '1' per nessun denominatore.",
    )

    # Toggle unità fase (persiste nello stato)
    phase_unit = st.radio(
        "Unità fase",
        ["Gradi (°)", "Radianti (π)"],
        horizontal=True,
        index=0 if st.session_state.phase_unit == "Gradi (°)" else 1,
        key="phase_unit_radio",
    )
    st.session_state.phase_unit = phase_unit
    phase_in_radians = phase_unit == "Radianti (π)"

    analyze_clicked = st.button("🔍 Analizza", type="primary")

    if analyze_clicked:
        st.session_state.analyzed = True

    if not st.session_state.analyzed:
        st.info("Inserisci una funzione di trasferimento e premi **Analizza**.")
        return

    # Ricostruisci G(s)
    if not den_str.strip() or den_str.strip() == "1":
        g_string = f"({num_str})"
    else:
        g_string = f"({num_str})/({den_str})"

    # ── Parsing ───────────────────────────────────────────────────────────
    try:
        with st.spinner("Calcolo in corso..."):
            info = parse_transfer_function(g_string)
    except (ValueError, Exception) as exc:
        st.error(f"⚠️ {exc}")
        return

    _show_sidebar_info(info)

    # ── Risposta in frequenza ─────────────────────────────────────────────
    omega = _compute_omega_range(info, n_points=500)
    # Nyquist usa 1000 punti
    omega_ny = _compute_omega_range(info, n_points=1000)

    try:
        with st.spinner("Calcolo in corso..."):
            # Bode
            resp = info.tf(1j * omega)
            mag = np.abs(resp)
            mag_db = 20.0 * np.log10(np.where(mag > 0, mag, 1e-30))
            phase_deg = np.degrees(np.unwrap(np.angle(resp)))
            approx_mag_db, approx_phase_deg = compute_asymptotic_bode(
                omega, info,
            )

            # Nyquist
            resp_ny = info.tf(1j * omega_ny)
            approx_resp_ny = compute_asymptotic_nyquist(omega_ny, info)
    except Exception as exc:
        st.error(f"⚠️ Errore di calcolo: {exc}")
        return

    # ── Determina omega_min / omega_max dal vettore ────────────────────────
    omega_min_val = float(omega[0])
    omega_max_val = float(omega[-1])

    # ── Recupera risultato query precedente (session_state) ───────────────
    if "omega_query_result" not in st.session_state:
        st.session_state.omega_query_result = None

    query_pt = st.session_state.omega_query_result

    # ── Diagramma di Bode ─────────────────────────────────────────────────
    st.subheader("Diagramma di Bode")
    bode_fig = plot_bode(
        omega, mag_db, phase_deg,
        approx_mag_db, approx_phase_deg, info,
        phase_in_radians=phase_in_radians,
        cursor_omega=st.session_state.cursor_omega,
    )
    st.plotly_chart(bode_fig, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════
    # INTERROGAZIONE PUNTUALE (dopo Bode, prima del cursore slider)
    # ══════════════════════════════════════════════════════════════════════
    st.markdown("---")
    st.subheader("🔎 Interrogazione Puntuale")

    qcol1, qcol2, qcol3 = st.columns([2, 1, 1])
    with qcol1:
        omega_input = st.number_input(
            label="Inserisci una frequenza ω [rad/s]",
            min_value=0.0,
            value=None,
            step=None,
            format="%f",
            placeholder="es. 0.1",
            key="omega_query",
        )
    with qcol2:
        omega_unit = st.selectbox(
            label="Unità inserimento",
            options=["rad/s", "Hz"],
            key="omega_query_unit",
        )
    with qcol3:
        st.markdown("<br>", unsafe_allow_html=True)
        btn_query = st.button("Calcola punto", key="btn_query")

    if btn_query and omega_input is not None:
        # Conversione Hz → rad/s se necessario
        if omega_unit == "Hz":
            omega_q = omega_input * 2.0 * np.pi
        else:
            omega_q = omega_input

        if omega_q <= 0:
            st.warning("La frequenza deve essere un valore positivo.")
        else:
            if omega_q < omega_min_val or omega_q > omega_max_val:
                st.warning(
                    f"Frequenza fuori dal range calcolato "
                    f"[{omega_min_val:.4f}, {omega_max_val:.4f}] rad/s. "
                    f"Il valore verrà comunque calcolato ma potrebbe "
                    f"non essere significativo."
                )
            res = query_single_frequency(
                omega_q, info, omega_min_val, omega_max_val,
            )
            st.session_state.omega_query_result = res

    # Mostra tabella risultati se presente in session_state
    if st.session_state.omega_query_result is not None:
        res = st.session_state.omega_query_result
        st.markdown(
            f"#### Risultati per ω = {res['omega']:.6f} rad/s"
        )
        _render_query_table(res)

    # ── Slider cursore ────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🎯 Cursore Frequenza")
    omega_list = omega.tolist()
    omega_cursor = st.select_slider(
        "Cursore frequenza ω [rad/s]",
        options=omega_list,
        value=(
            st.session_state.cursor_omega
            if st.session_state.cursor_omega in omega_list
            else omega_list[len(omega_list) // 2]
        ),
        format_func=lambda x: f"{x:.4f} rad/s",
        key="cursor_slider",
    )
    st.session_state.cursor_omega = omega_cursor

    # Metriche cursore
    wc = omega_cursor
    resp_at_wc = info.tf(1j * wc)
    render_cursor_metrics(wc, resp_at_wc, phase_in_radians)

    # ── Diagramma Polare / Nyquist ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Diagramma Polare (Nyquist)")
    cursor_resp_ny = None
    if st.session_state.cursor_omega is not None:
        cursor_resp_ny = info.tf(1j * st.session_state.cursor_omega)
    nyquist_fig = plot_nyquist(
        omega_ny, resp_ny, approx_resp_ny,
        cursor_omega=st.session_state.cursor_omega,
        cursor_resp=cursor_resp_ny,
    )
    st.plotly_chart(nyquist_fig, use_container_width=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
