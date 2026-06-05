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
import plotly.graph_objects as go  # type: ignore
import streamlit as st
import sympy
from plotly.subplots import make_subplots  # type: ignore

# ---------------------------------------------------------------------------
# Configurazione pagina (deve essere il primo comando Streamlit)
# ---------------------------------------------------------------------------
import os
from PIL import Image

logo_path = "logo.png"
if not os.path.exists(logo_path):
    # Cerca nella cartella superiore
    parent_logo = os.path.join("..", "logo.png")
    if os.path.exists(parent_logo):
        logo_path = parent_logo
    else:
        # Cerca rispetto alla cartella dello script
        script_dir = os.path.dirname(__file__)
        script_logo = os.path.join(script_dir, "logo.png")
        if os.path.exists(script_logo):
            logo_path = script_logo
        else:
            parent_script_logo = os.path.join(script_dir, "..", "logo.png")
            if os.path.exists(parent_script_logo):
                logo_path = parent_script_logo

try:
    logo_img = Image.open(logo_path)
except Exception:
    logo_img = "📈"

st.set_page_config(
    page_title="Analizzatore Bode & Nyquist",
    page_icon=logo_img,
    layout="wide",
)


warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Costanti
# ---------------------------------------------------------------------------
_EXACT_COLOR  = "#4d9de0"   # blu chiaro – leggibile su entrambi i temi
_APPROX_COLOR = "#f4a261"   # arancione caldo
_BREAKPT_COLOR = "#e05252"  # rosso per linee verticali ω_r
_CRITICAL_COLOR = "#ff6b6b" # rosso per punto critico Nyquist
_EXACT_WIDTH  = 2.0
_APPROX_WIDTH = 1.5
_CURSOR_COLOR = "#e05252"
_QUERY_COLOR  = "#2ecc71"

# ── CSS temi ─────────────────────────────────────────────────────────────
_DARK_CSS = """
<style>
/* === ROOT E BODY === */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.stApp, .main, .main .block-container {
    background-color: #0e1117 !important;
    color: #e8e8f0 !important;
}
/* === HEADER === */
header[data-testid="stHeader"],
header[data-testid="stHeader"] * {
    background-color: #0e1117 !important;
}
/* === SIDEBAR === */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {
    background-color: #12131f !important;
    border-right: 1px solid #2a2a3e !important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #c8c8d8 !important;
}
/* === METRIC === */
[data-testid="stMetricValue"] { color: #7eb8f7 !important; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #9999b8 !important; font-size: 0.78rem !important; }
/* === INPUT === */
.stTextInput input, .stTextInput textarea,
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background-color: #1c1e2e !important;
    color: #e8e8f0 !important;
    border: 1px solid #3a3a5c !important;
    border-radius: 8px !important;
}
.stTextInput input:focus, [data-testid="stTextInput"] input:focus {
    border-color: #5c6bc0 !important;
    box-shadow: 0 0 0 2px rgba(92,107,192,0.3) !important;
}
/* === SELECT === */
.stSelectbox select, [data-testid="stSelectbox"] * {
    background-color: #1c1e2e !important;
    color: #e8e8f0 !important;
}
/* === PULSANTE ANALIZZA === */
.stButton > button[kind="primary"] {
    background-color: #e05252 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background-color: #c04040 !important;
    transform: translateY(-1px) !important;
}
/* === GRAFICI === */
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    background-color: #0e1117 !important;
    border-radius: 10px !important;
}
/* === TESTO === */
p, h1, h2, h3, h4, label, .stMarkdown, .stCaption { color: #c8c8d8 !important; }
hr { border-color: #2a2a3e !important; }

/* === ALERT/WARNING BOX === */
[data-testid="stAlert"], div[role="alert"], .stAlert {
    background-color: #1e2040 !important;
    color: #c8c8e8 !important;
    border-left: 4px solid #5c6bc0 !important;
}
</style>
"""

_LIGHT_CSS = """
<style>
/* === ROOT E BODY === */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main,
.main .block-container {
    background-color: #f7f8fc !important;
    color: #1a1a2e !important;
}

/* === HEADER === */
header[data-testid="stHeader"],
header[data-testid="stHeader"] * {
    background-color: #f7f8fc !important;
    color: #1a1a2e !important;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {
    background-color: #eef0f8 !important;
    border-right: 1px solid #d0d4e8 !important;
}

/* Testo sidebar — TUTTI i livelli */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] * {
    color: #1a1a2e !important;
}

/* === METRIC (numeri grandi nella sidebar) === */
[data-testid="stMetricValue"] {
    color: #1a3a6e !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: #444466 !important;
    font-size: 0.78rem !important;
}
[data-testid="stMetricDelta"] {
    color: #2a6a2a !important;
}

/* === INPUT FIELDS — sovrascrivi dark mode residua === */
.stTextInput input,
.stTextInput textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] textarea,
[data-testid="stNumberInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background-color: #ffffff !important;
    color: #1a1a2e !important;
    border: 1px solid #c0c4d8 !important;
    border-radius: 8px !important;
}
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: #4d6cc0 !important;
    box-shadow: 0 0 0 2px rgba(77, 108, 192, 0.2) !important;
    background-color: #ffffff !important;
}

/* === BOX ESEMPI FORMA (copertura completa e forzata) === */
.stApp div[data-testid="stMarkdownContainer"] code,
.stApp .stCode, 
.stApp pre, 
.stApp code,
.stApp [data-testid="stCode"],
.stApp [data-testid="stCode"] pre,
.stApp [data-testid="stCode"] code,
.stApp div[data-testid="stCode"] > div,
.stApp input:disabled,
.stApp [data-testid="stTextInput"] input:disabled,
.stApp div[data-baseweb="input"] input[disabled],
.stApp div[data-baseweb="input"] input[readonly] {
    background-color: #e8eaf0 !important;
    color: #2a2a4a !important;
    border: 1px solid #c8cae0 !important;
    border-radius: 6px !important;
}

/* === SELECT/DROPDOWN === */
div[data-baseweb="select"] *,
.stSelectbox * {
    background-color: #ffffff !important;
    color: #1a1a2e !important;
}

/* === PULSANTE ANALIZZA === */
.stButton > button {
    background-color: #e05252 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.stButton > button:hover {
    background-color: #c04040 !important;
}

/* === TESTO PRINCIPALE === */
p, h1, h2, h3, h4, li {
    color: #1a1a2e !important;
}
.stMarkdown p, .stMarkdown span {
    color: #1a1a2e !important;
}

/* === CAPTION / TESTO SECONDARIO === */
.stCaption, small, [data-testid="stCaptionContainer"] {
    color: #555577 !important;
}

/* === DIVIDERS === */
hr {
    border-color: #d0d4e8 !important;
}

/* === TOGGLE LABEL === */
[data-testid="stToggle"] label,
[data-testid="stToggle"] span {
    color: #1a1a2e !important;
}

/* === RADIO BUTTON LABEL === */
[data-testid="stRadio"] label,
[data-testid="stRadio"] span {
    color: #1a1a2e !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label span {
    color: #555577 !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label div {
    border-color: #9090aa !important;
}

/* === ALERT/WARNING BOX === */
[data-testid="stAlert"], div[role="alert"], .stAlert {
    background-color: #e8eaf6 !important;
    color: #1a1a2e !important;
    border-left: 4px solid #5c6bc0 !important;
}

/* === CONTENITORE GRAFICI === */
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    background-color: #ffffff !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

/* === SLIDER === */
[data-testid="stSlider"] * {
    color: #1a1a2e !important;
}
</style>
"""


# ---------------------------------------------------------------------------
# Helper: applica tema Plotly ai grafici (Problema 2)
# ---------------------------------------------------------------------------
def applica_tema_plotly(fig: go.Figure, dark_mode: bool) -> go.Figure:
    """Applica colori e font coerenti al tema scelto su ogni figura Plotly."""
    if dark_mode:
        bg       = "#0e1117"
        paper_bg = "#13141f"
        grid_col = "#1e1e32"
        zeroline = "#2a2a4a"
        font_col = "#c8c8d8"
        spike_col = "#e05252"
        template = "plotly_dark"
    else:
        bg       = "#ffffff"
        paper_bg = "#f8f9fc"
        grid_col = "#e8eaf0"
        zeroline = "#c5cae9"
        font_col = "#1a1a2e"
        spike_col = "#e05252"
        template = "plotly_white"

    fig.update_layout(
        template=template,
        paper_bgcolor=paper_bg,
        plot_bgcolor=bg,
        font=dict(color=font_col, size=12, family="'Inter', 'Segoe UI', sans-serif"),
        title_font=dict(color=font_col, size=14),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=font_col, size=11),
            bordercolor="rgba(0,0,0,0)",
        ),
    )
    # Aggiorna tutti gli assi presenti (gestisce subplot automaticamente)
    for key in fig.layout:
        obj = getattr(fig.layout, key, None)
        if obj is None:
            continue
        if hasattr(obj, "gridcolor"):
            obj.update(
                gridcolor=grid_col,
                zerolinecolor=zeroline,
                zerolinewidth=1,
                tickfont=dict(color=font_col, size=11),
                title_font=dict(color=font_col, size=12),
                linecolor=grid_col,
                showgrid=True,
                spikecolor=spike_col,
            )
    return fig


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

from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations,
    implicit_multiplication_application, convert_xor
)
from latex2sympy2_extended import latex2sympy
from sympy import symbols, Poly

TRANSFORMATIONS = (
    standard_transformations +
    (implicit_multiplication_application, convert_xor)
)

def _parse_single_expr(raw: str, s):
    """
    Tenta il parsing con due strategie in cascata:
    1. parse_expr con implicit_multiplication (per input naturale)
    2. latex2sympy (per input LaTeX da copia-incolla o utenti avanzati)
    Lancia ValueError se entrambi falliscono.
    """
    raw = raw.strip()
    if not raw:
        raise ValueError("Il campo è vuoto.")

    # Strategia 1: input naturale (2s, s^2, s*(1+s/10), ecc.)
    try:
        return parse_expr(
            raw,
            local_dict={"s": s},
            transformations=TRANSFORMATIONS
        )
    except Exception:
        pass

    # Strategia 2: LaTeX puro (\frac{s}{1+s}, s^{2}, ecc.)
    try:
        expr = latex2sympy(raw)
        # Rimappa il simbolo al nostro s
        free = expr.free_symbols
        if free and s not in free:
            expr = expr.subs(list(free)[0], s)
        return expr
    except Exception as e2:
        raise ValueError(
            f"Espressione non riconosciuta. "
            f"Prova a scrivere in forma naturale (es: s*(s+1)) "
            f"oppure in LaTeX (es: \\frac{{s}}{{s+1}}). "
            f"Dettaglio tecnico: {e2}"
        )

def parse_transfer_function(latex_num: str, latex_den: str) -> SystemInfo:
    s = symbols("s")
    num_expr = _parse_single_expr(latex_num, s)
    den_expr = _parse_single_expr(latex_den, s)
    try:
        num_poly = Poly(num_expr.expand(), s)
        den_poly = Poly(den_expr.expand(), s)
        num_coeffs = [float(c) for c in num_poly.all_coeffs()]
        den_coeffs = [float(c) for c in den_poly.all_coeffs()]
    except Exception as e:
        raise ValueError(
            f"Impossibile costruire la funzione di trasferimento. "
            f"Verifica che il risultato sia un polinomio razionale in s. "
            f"Dettaglio: {e}"
        )
        
    expr = num_expr / den_expr
    num_expr, den_expr = sympy.fraction(sympy.cancel(expr))

    # Usa all_roots() per preservare la molteplicità dei poli/zeri
    # sympy.solve() restituisce solo radici distinte, perdendo i poli doppi
    try:
        num_poly = sympy.Poly(sympy.expand(num_expr), s)
        den_poly = sympy.Poly(sympy.expand(den_expr), s)
    except sympy.GeneratorsNeeded:
        num_poly = sympy.Poly(sympy.expand(num_expr), s, domain="ZZ")
        den_poly = sympy.Poly(sympy.expand(den_expr), s, domain="ZZ")

    # Estrai coefficienti e filtra artefatti numerici
    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]
    
    num_coeffs = [c if abs(c) > 1e-10 else 0.0 for c in num_coeffs]
    den_coeffs = [c if abs(c) > 1e-10 else 0.0 for c in den_coeffs]

    # Usa np.roots come richiesto, ma forza a 0 gli zeri vicinissimi all'origine per evitare artefatti
    raw_zeros = np.roots(num_coeffs)
    raw_poles = np.roots(den_coeffs)
    zeros = [complex(z) if abs(z) > 1e-10 else 0j for z in raw_zeros]
    poles = [complex(p) if abs(p) > 1e-10 else 0j for p in raw_poles]

    order = len(den_coeffs) - 1
    system_type = sum(1 for p in poles if abs(p) < 1e-10)

    try:
        if abs(den_coeffs[-1]) < 1e-15:
            static_gain = None
        else:
            static_gain = float(num_coeffs[-1] / den_coeffs[-1])
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
# 3. DECOMPOSIZIONE NELLA FORMA ALTERNATIVA PER BODE
#    (Metodo Basile & Chiacchio, "Lezioni di Automatica")
# ═══════════════════════════════════════════════════════════════════════════

import numpy as np

def compute_approximated_bode(
    omega: np.ndarray,
    info: 'SystemInfo',
) -> tuple[np.ndarray, np.ndarray]:
    """Calcola diagramma di Bode APPROSSIMATO (asintotico) corretto."""
    import numpy as np
    
    num = info.num_coeffs
    den = info.den_coeffs
    
    # 1. Trova zeri e poli
    zeros = np.roots(num)
    poles = np.roots(den)
    
    # Separa poli e zeri nell'origine
    tol = 1e-8
    z_origin = np.sum(np.abs(zeros) < tol)
    p_origin = np.sum(np.abs(poles) < tol)
    g = z_origin - p_origin
    
    zeros_fin = zeros[np.abs(zeros) >= tol]
    poles_fin = poles[np.abs(poles) >= tol]
    
    K_num = num[0] if len(num) > 0 else 1.0
    K_den = den[0] if len(den) > 0 else 1.0
    
    breakpoints = []
    
    processed_z = np.zeros(len(zeros_fin), dtype=bool)
    for i, z in enumerate(zeros_fin):
        if processed_z[i]: continue
        if np.abs(z.imag) < tol:
            breakpoints.append(('zero_real', np.abs(z.real), z.real))
            processed_z[i] = True
        else:
            for j in range(i+1, len(zeros_fin)):
                if not processed_z[j] and np.abs(z - np.conj(zeros_fin[j])) < 1e-6:
                    wn = np.abs(z)
                    breakpoints.append(('zero_complex', wn, z.real))
                    processed_z[i] = processed_z[j] = True
                    break
    
    processed_p = np.zeros(len(poles_fin), dtype=bool)
    for i, p in enumerate(poles_fin):
        if processed_p[i]: continue
        if np.abs(p.imag) < tol:
            breakpoints.append(('pole_real', np.abs(p.real), p.real))
            processed_p[i] = True
        else:
            for j in range(i+1, len(poles_fin)):
                if not processed_p[j] and np.abs(p - np.conj(poles_fin[j])) < 1e-6:
                    wn = np.abs(p)
                    breakpoints.append(('pole_complex', wn, p.real))
                    processed_p[i] = processed_p[j] = True
                    break
    
    # ── Calcolo K_b con segno corretto (metodo Basile & Chiacchio) ────────
    # K_b = (a_n / b_m) * product(-z_i_fin) / product(-p_j_fin)
    # dove a_n e b_m sono i leading coefficients.
    # Usare (-root) anziché |root| preserva il segno per poli/zeri RHP.
    K_b = complex(K_num / K_den)
    for z in zeros_fin:
        K_b *= (-z)
    for p in poles_fin:
        K_b /= (-p)
    K_b_real = float(K_b.real)  # la parte immaginaria è ~0 per sistemi reali
    K_bode_abs = abs(K_b_real)
    
    mag_dB = 20 * np.log10(max(K_bode_abs, 1e-30)) + g * 20 * np.log10(omega)
    
    breakpoints.sort(key=lambda x: x[1])
    for tipo, wr, _ in breakpoints:
        if tipo == 'zero_real':
            slope = +20
        elif tipo == 'pole_real':
            slope = -20
        elif tipo == 'zero_complex':
            slope = +40
        elif tipo == 'pole_complex':
            slope = -40
        
        mask = omega > wr
        mag_dB[mask] += slope * np.log10(omega[mask] / wr)
    
    # ── Fase iniziale ─────────────────────────────────────────────────────
    # G(s) ~ K_b * s^g  per ω → 0
    # arg(G(jω)) = arg(K_b) + g * arg(jω) = arg(K_b) + g * 90°
    # + 180° se K_b < 0 (segno negativo del guadagno)
    phase_deg = np.full(len(omega), float(g * 90.0))
    if K_b_real < 0:
        phase_deg += 180.0
    
    # ── Delta fase per ogni breakpoint ────────────────────────────────────
    # Fattore LHP (1+s/wr): arg va 0 → +90°, contributo a G = -90° (polo) o +90° (zero)
    # Fattore RHP (1-s/wr): arg va 0 → -90°, contributo a G = +90° (polo) o -90° (zero)
    # Il segno si inverte per radici RHP perché il fattore normalizzato ha fase opposta.
    def get_delta_fase(tipo: str, parte_reale: float) -> float:
        lhp = (parte_reale <= 1e-8)
        if tipo == 'zero_real':    return +90.0 if lhp else -90.0
        if tipo == 'pole_real':    return -90.0 if lhp else +90.0
        if tipo == 'zero_complex': return +180.0 if lhp else -180.0
        if tipo == 'pole_complex': return -180.0 if lhp else +180.0
        return 0.0

    for tipo, wr, parte_reale in breakpoints:
        w_start = wr / 10.0
        w_end   = wr * 10.0
        
        delta = get_delta_fase(tipo, parte_reale)
        
        mask_ramp  = (omega >= w_start) & (omega <= w_end)
        mask_after = omega > w_end
        
        if np.any(mask_ramp):
            phase_deg[mask_ramp] += delta * (
                np.log10(omega[mask_ramp]) - np.log10(w_start)
            ) / (np.log10(w_end) - np.log10(w_start))
        
        phase_deg[mask_after] += delta
    
    return mag_dB, phase_deg




# ═══════════════════════════════════════════════════════════════════════════
# 4. NYQUIST APPROSSIMATO
# ═══════════════════════════════════════════════════════════════════════════

def calcola_polare_approssimato(mag_dB_approx, phase_deg_approx, omega):
    """Ricava il diagramma polare dalla versione approssimata del Bode."""
    mag_lin = 10 ** (mag_dB_approx / 20.0)
    phase_rad = np.deg2rad(phase_deg_approx)
    
    re = mag_lin * np.cos(phase_rad)
    im = mag_lin * np.sin(phase_rad)
    
    return re, im


# ═══════════════════════════════════════════════════════════════════════════
# 5. DIAGRAMMA DI BODE (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def plot_bode(
    plotly_template: str,
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
                line_color=_BREAKPT_COLOR,
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

    # Y modulo: auto-range basato sul minimo/massimo di entrambe le curve
    y_mag_min = float(min(np.nanmin(mag_db), np.nanmin(approx_mag_db)))
    y_mag_max = float(max(np.nanmax(mag_db), np.nanmax(approx_mag_db)))
    fig.update_yaxes(
        title_text="Modulo (dB)",
        range=[y_mag_min - 5, y_mag_max + 5],
        showspikes=True, spikecolor="gray", spikethickness=1, spikedash="dot",
        row=1, col=1,
    )

    # Y fase
    ph_min = float(min(np.nanmin(exact_phase_plot), np.nanmin(approx_phase_plot)))
    ph_max = float(max(np.nanmax(exact_phase_plot), np.nanmax(approx_phase_plot)))

    if phase_in_radians:
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
        tv = list(np.arange(int(ph_min) - 45, int(ph_max) + 46, 45))
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
        height=700,
        template=plotly_template,
        margin=dict(l=60, r=40, t=50, b=60),
        hovermode="x unified",
        hoverdistance=50,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. DIAGRAMMA POLARE / NYQUIST (Plotly)
# ═══════════════════════════════════════════════════════════════════════════

def plot_nyquist(
    plotly_template: str,
    omega: np.ndarray,
    resp: np.ndarray,
    cursor_omega: Optional[float] = None,
    cursor_resp: Optional[complex] = None,
) -> go.Figure:
    """Diagramma polare completo (ω > 0 e ω < 0 specchiato)."""
    re_pos = resp.real
    im_pos = resp.imag
    # Ramo ω < 0: complesso coniugato (specchiato rispetto asse reale)
    re_neg = resp.real
    im_neg = -resp.imag

    _NEG_COLOR = "#b07be0"  # viola per il ramo ω < 0

    fig = go.Figure()

    # ── Ramo ω > 0 (traccia principale) ───────────────────────────────────
    fig.add_trace(go.Scatter(
        x=re_pos, y=im_pos, mode="lines",
        name="ω > 0",
        line=dict(color=_EXACT_COLOR, width=_EXACT_WIDTH),
        hovertemplate="Re: %{x:.4f}<br>Im: %{y:.4f}<extra>ω > 0</extra>",
    ))

    # ── Ramo ω < 0 (specchiato) ───────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=re_neg, y=im_neg, mode="lines",
        name="ω < 0",
        line=dict(color=_NEG_COLOR, width=_APPROX_WIDTH, dash="dash"),
        hovertemplate="Re: %{x:.4f}<br>Im: %{y:.4f}<extra>ω < 0</extra>",
    ))

    # ── Frecce direzionali ramo ω > 0 ─────────────────────────────────────
    step = max(1, len(omega) // 15)
    for idx in range(step, len(omega) - 1, step):
        dx = re_pos[idx + 1] - re_pos[idx]
        dy = im_pos[idx + 1] - im_pos[idx]
        fig.add_annotation(
            x=re_pos[idx], y=im_pos[idx],
            ax=re_pos[idx] - dx * 8,
            ay=im_pos[idx] - dy * 8,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5,
            arrowwidth=1.5, arrowcolor=_EXACT_COLOR,
        )

    # ── Frecce direzionali ramo ω < 0 ─────────────────────────────────────
    for idx in range(step, len(omega) - 1, step):
        dx = re_neg[idx + 1] - re_neg[idx]
        dy = im_neg[idx + 1] - im_neg[idx]
        fig.add_annotation(
            x=re_neg[idx], y=im_neg[idx],
            ax=re_neg[idx] - dx * 8,
            ay=im_neg[idx] - dy * 8,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.2,
            arrowwidth=1.2, arrowcolor=_NEG_COLOR,
        )

    # ── Marker di inizio e fine ───────────────────────────────────────────
    # ω → 0+ (inizio ramo positivo)
    fig.add_trace(go.Scatter(
        x=[re_pos[0]], y=[im_pos[0]],
        mode="markers+text",
        marker=dict(color="#2ecc71", size=12, symbol="circle"),
        text=["ω→0⁺"], textposition="top right",
        textfont=dict(size=13, color="#2ecc71"),
        name="ω→0⁺",
    ))
    # ω → 0⁻ (inizio ramo negativo, coniugato)
    fig.add_trace(go.Scatter(
        x=[re_neg[0]], y=[im_neg[0]],
        mode="markers+text",
        marker=dict(color="#27ae60", size=10, symbol="diamond"),
        text=["ω→0⁻"], textposition="bottom right",
        textfont=dict(size=12, color="#27ae60"),
        name="ω→0⁻",
    ))

    # ω → +∞
    fig.add_trace(go.Scatter(
        x=[re_pos[-1]], y=[im_pos[-1]],
        mode="markers+text",
        marker=dict(color="#e74c3c", size=12, symbol="square"),
        text=["ω→+∞"], textposition="top right",
        textfont=dict(size=13, color="#e74c3c"),
        name="ω→+∞",
    ))
    # ω → -∞
    fig.add_trace(go.Scatter(
        x=[re_neg[-1]], y=[im_neg[-1]],
        mode="markers+text",
        marker=dict(color="#c0392b", size=10, symbol="square"),
        text=["ω→-∞"], textposition="bottom right",
        textfont=dict(size=12, color="#c0392b"),
        name="ω→-∞",
    ))

    # ── Punto critico (−1, 0) ─────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=[-1], y=[0],
        mode="markers+text",
        marker=dict(color=_CRITICAL_COLOR, size=14, symbol="x",
                     line=dict(width=2, color=_CRITICAL_COLOR)),
        text=["(-1, 0)"], textposition="bottom right",
        textfont=dict(size=11, color=_CRITICAL_COLOR),
        name="Punto critico (−1, 0)",
    ))

    # ── Cursore frequenza ─────────────────────────────────────────────────
    if cursor_resp is not None and cursor_omega is not None:
        # Punto sul ramo ω > 0
        fig.add_trace(go.Scatter(
            x=[cursor_resp.real], y=[cursor_resp.imag],
            mode="markers+text",
            marker=dict(color=_CURSOR_COLOR, size=12, symbol="circle"),
            text=[f"ω = {cursor_omega:.4g}"],
            textposition="top right",
            textfont=dict(size=11, color=_CURSOR_COLOR),
            name=f"ω = {cursor_omega:.4g} rad/s",
        ))
        # Punto simmetrico sul ramo ω < 0
        fig.add_trace(go.Scatter(
            x=[cursor_resp.real], y=[-cursor_resp.imag],
            mode="markers",
            marker=dict(color=_CURSOR_COLOR, size=10, symbol="circle-open",
                         line=dict(width=2, color=_CURSOR_COLOR)),
            name=f"ω = −{cursor_omega:.4g} rad/s",
            showlegend=False,
        ))

    fig.update_layout(
        xaxis_title="Re{G(jω)}",
        yaxis_title="Im{G(jω)}",
        height=700,
        template=plotly_template,
        yaxis_scaleanchor="x",
        showlegend=True,
        hovermode="closest",
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
        margin=dict(l=60, r=40, t=50, b=60),
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
    approx_mag_full, approx_phase_full = compute_approximated_bode(
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


def _format_roots(roots: list[complex], st_container, label_prefix: str):
    if not roots:
        st_container.write("Nessuno")
        return
    
    roots_sorted = sorted(roots, key=lambda r: r.real)
    processed = set()
    for i, r in enumerate(roots_sorted):
        if i in processed:
            continue
            
        if abs(r.imag) > 1e-10:
            conj_index = -1
            for j in range(i+1, len(roots_sorted)):
                if j not in processed and abs(roots_sorted[j].real - r.real) < 1e-10 and abs(roots_sorted[j].imag + r.imag) < 1e-10:
                    conj_index = j
                    break
            
            if conj_index != -1:
                processed.add(i)
                processed.add(conj_index)
                # Problema 6: formatta polo complesso come stringa leggibile
                sign = "+" if r.imag >= 0 else "−"
                valore = f"{r.real:.4f} {sign} {abs(r.imag):.4f}j"
                st_container.metric(
                    label=f"{label_prefix} {i+1}-{conj_index+1} (complessi coniugati)",
                    value=valore,
                )
                wn = abs(r)
                zeta = -r.real / wn if wn > 1e-10 else 0.0
                st_container.caption(f"ωₙ = {wn:.4f} rad/s  |  ζ = {zeta:.4f}")
            else:
                processed.add(i)
                st_container.text_input(
                    label=f"{label_prefix} {i+1}",
                    value=f"{r.real:.4f}{r.imag:+.4f}j",
                    disabled=True, key=f"{label_prefix}_{i}"
                )
        else:
            processed.add(i)
            val = r.real
            if abs(val) < 1e-10:
                st_container.metric(f"{label_prefix} {i+1}", "0 (origine)")
            else:
                st_container.metric(f"{label_prefix} {i+1}", f"{val:.4f}")

def compute_stability_margins(sys_tf, omega: np.ndarray) -> dict:
    import control
    freq_resp = control.frequency_response(sys_tf, omega)
    mag = np.abs(freq_resp.fresp.squeeze())
    phase_deg = np.angle(freq_resp.fresp.squeeze(), deg=True)
    mag_dB = 20 * np.log10(np.where(mag > 0, mag, 1e-12))

    result = {
        "omega_gc": [],
        "omega_pc": [],
        "GM_dB":    None,
        "PM_deg":   None,
        "stabile":  None,
    }

    for i in range(len(mag_dB) - 1):
        if mag_dB[i] * mag_dB[i+1] <= 0:
            ogc = float(np.interp(0, [mag_dB[i], mag_dB[i+1]],
                                     [omega[i], omega[i+1]]))
            result["omega_gc"].append(ogc)

    phase_shifted = phase_deg + 180
    for i in range(len(phase_shifted) - 1):
        if phase_shifted[i] * phase_shifted[i+1] <= 0:
            opc = float(np.interp(0, [phase_shifted[i], phase_shifted[i+1]],
                                     [omega[i], omega[i+1]]))
            result["omega_pc"].append(opc)

    if result["omega_gc"]:
        result["PM_deg"] = float(
            np.interp(result["omega_gc"][0], omega, phase_deg)
        ) + 180.0

    if result["omega_pc"]:
        result["GM_dB"] = -float(
            np.interp(result["omega_pc"][0], omega, mag_dB)
        )

    gm_ok = result["GM_dB"] > 0 if result["GM_dB"] is not None else None
    pm_ok = result["PM_deg"] > 0 if result["PM_deg"] is not None else None
    if gm_ok is not None and pm_ok is not None:
        result["stabile"] = gm_ok and pm_ok

    return result


def _show_sidebar_info(info: SystemInfo, omega: np.ndarray) -> None:
    """Visualizza le informazioni del sistema nella sidebar."""
    st.sidebar.header("📋 Informazioni Sistema")

    st.sidebar.caption(f"G(s) — ordine {int(info.order)}, tipo {int(info.system_type)}")

    # Zeri
    if info.zeros:
        st.sidebar.subheader("Zeri")
        _format_roots(info.zeros, st.sidebar, "Zero")

    # Poli
    if info.poles:
        st.sidebar.subheader("Poli")
        _format_roots(info.poles, st.sidebar, "Polo")

    # Scalari
    st.sidebar.metric("Ordine del Sistema", int(info.order))
    st.sidebar.metric("Tipo del Sistema", int(info.system_type))

    K_num = float(info.num_coeffs[0]) if len(info.num_coeffs) > 0 else 1.0
    K_den = float(info.den_coeffs[0]) if len(info.den_coeffs) > 0 else 1.0
    K_b_complex = complex(K_num / K_den)
    for z in info.zeros:
        if abs(z) > 1e-8: K_b_complex *= (-z)
    for p in info.poles:
        if abs(p) > 1e-8: K_b_complex /= (-p)
    K_b = float(K_b_complex.real)

    segno = "−" if K_b < 0 else "+"
    st.sidebar.metric("Costante di Bode |K_b|", f"{abs(K_b):.4f}")
    if K_b < 0:
        st.sidebar.caption("K_b < 0 → sfasamento iniziale di +180° incluso nella fase")

    margins = compute_stability_margins(info.tf, omega)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Stabilità")

    if margins["GM_dB"] is not None:
        st.sidebar.metric(
            label="Margine di Guadagno",
            value=f"{margins['GM_dB']:.2f} dB",
            delta="stabile" if margins["GM_dB"] > 0 else "instabile",
            delta_color="normal" if margins["GM_dB"] > 0 else "inverse"
        )
    else:
        st.sidebar.metric("Margine di Guadagno", "∞")

    if margins["PM_deg"] is not None:
        st.sidebar.metric(
            label="Margine di Fase",
            value=f"{margins['PM_deg']:.2f}°",
            delta="stabile" if margins["PM_deg"] > 0 else "instabile",
            delta_color="normal" if margins["PM_deg"] > 0 else "inverse"
        )
    else:
        st.sidebar.metric("Margine di Fase", "∞")

    if margins["omega_gc"]:
        for i, ogc in enumerate(margins["omega_gc"]):
            label = "ω cross. guadagno" if len(margins["omega_gc"]) == 1 \
                    else f"ω cross. guadagno {i+1}"
            st.sidebar.metric(label=label, value=f"{ogc:.4f} rad/s")
    else:
        st.sidebar.metric("ω cross. guadagno", "—")

    if margins["omega_pc"]:
        for i, opc in enumerate(margins["omega_pc"]):
            label = "ω cross. fase" if len(margins["omega_pc"]) == 1 \
                    else f"ω cross. fase {i+1}"
            st.sidebar.metric(label=label, value=f"{opc:.4f} rad/s")
    else:
        st.sidebar.metric("ω cross. fase", "—")

    if margins["stabile"] is True:
        st.sidebar.success("✅ Sistema stabile")
    elif margins["stabile"] is False:
        st.sidebar.error("❌ Sistema instabile")
    else:
        st.sidebar.warning("⚠️ Stabilità indeterminata")


# ═══════════════════════════════════════════════════════════════════════════
# 8b. DISCRETIZZAZIONE C(s) → C(z) → u[k]
# ═══════════════════════════════════════════════════════════════════════════

def discretize_tf(num_expr_sym, den_expr_sym, Ts_val: float, method: str) -> dict:
    """
    Discretizza C(s) = num_expr_sym / den_expr_sym usando sostituzione algebrica.

    Parametri
    ---------
    num_expr_sym, den_expr_sym : espressioni SymPy in 's'
    Ts_val  : periodo di campionamento (float > 0)
    method  : 'forward' | 'backward' | 'tustin'

    Restituisce
    -----------
    dict con chiavi: steps, cz_num_coeffs, cz_den_coeffs, diff_eq_str, c_code
    """
    s_sym = sympy.Symbol('s')
    z_sym = sympy.Symbol('z')
    T_sym = sympy.Symbol('T', positive=True)

    # ── STEP 1: Sostituzione ──────────────────────────────────────────────
    if method == 'forward':
        s_sub = (z_sym - 1) / T_sym
        method_name = "Rapporto in Avanti (Forward Euler)"
        s_formula_latex = r"s = \frac{z - 1}{T_s}"
    elif method == 'backward':
        s_sub = (z_sym - 1) / (z_sym * T_sym)
        method_name = "Rapporto all'Indietro (Backward Euler)"
        s_formula_latex = r"s = \frac{z - 1}{z \cdot T_s}"
    elif method == 'tustin':
        s_sub = (sympy.Rational(2, 1) / T_sym) * (z_sym - 1) / (z_sym + 1)
        method_name = "Bilineare (Tustin)"
        s_formula_latex = r"s = \frac{2}{T_s} \cdot \frac{z - 1}{z + 1}"
    else:
        raise ValueError(f"Metodo sconosciuto: {method}")

    steps = []

    # Step 1 text
    cs_latex = r"\frac{" + sympy.latex(num_expr_sym) + r"}{" + sympy.latex(den_expr_sym) + r"}"
    step1 = (
        f"**STEP 1 — Sostituzione ({method_name})**\n\n"
        f"$$C(s) = {cs_latex}$$\n\n"
        f"Applico: ${s_formula_latex}$  con  $T_s = {Ts_val}$\n\n"
    )
    steps.append(step1)

    # ── STEP 2: Riduzione a forma razionale ───────────────────────────────
    num_z_raw = num_expr_sym.subs(s_sym, s_sub)
    den_z_raw = den_expr_sym.subs(s_sym, s_sub)

    num_z_simplified = sympy.simplify(num_z_raw)
    den_z_simplified = sympy.simplify(den_z_raw)

    Cz_expr = sympy.cancel(num_z_simplified / den_z_simplified)
    Cz_num, Cz_den = sympy.fraction(Cz_expr)

    Cz_num = sympy.expand(Cz_num.subs(T_sym, Ts_val))
    Cz_den = sympy.expand(Cz_den.subs(T_sym, Ts_val))

    try:
        num_poly_z = sympy.Poly(Cz_num, z_sym)
        den_poly_z = sympy.Poly(Cz_den, z_sym)
    except sympy.GeneratorsNeeded:
        num_poly_z = sympy.Poly(Cz_num, z_sym, domain='RR')
        den_poly_z = sympy.Poly(Cz_den, z_sym, domain='RR')

    num_coeffs_z = [float(c) for c in num_poly_z.all_coeffs()]
    den_coeffs_z = [float(c) for c in den_poly_z.all_coeffs()]

    # Normalizza per il coefficiente leader del denominatore
    a0 = den_coeffs_z[0]
    num_coeffs_z_norm = [c / a0 for c in num_coeffs_z]
    den_coeffs_z_norm = [c / a0 for c in den_coeffs_z]

    # Ricostruisci polinomi normalizzati per display pulito
    Cz_num_disp = sum(
        sympy.nsimplify(c, rational=False) * z_sym**(len(num_coeffs_z_norm) - 1 - i)
        for i, c in enumerate(num_coeffs_z_norm)
    )
    Cz_den_disp = sum(
        sympy.nsimplify(c, rational=False) * z_sym**(len(den_coeffs_z_norm) - 1 - i)
        for i, c in enumerate(den_coeffs_z_norm)
    )

    cz_latex = (r"\frac{" + sympy.latex(sympy.expand(Cz_num_disp))
                + r"}{" + sympy.latex(sympy.expand(Cz_den_disp)) + r"}")
    step2 = (
        f"**STEP 2 — Riduzione a Forma Razionale**\n\n"
        f"$$C(z) = \\frac{{U(z)}}{{E(z)}} = {cz_latex}$$\n\n"
    )
    steps.append(step2)

    # ── STEP 3: Prodotto incrociato ───────────────────────────────────────
    n_den = len(den_coeffs_z_norm)
    n_num = len(num_coeffs_z_norm)
    order_den = n_den - 1
    order_num = n_num - 1

    def _poly_terms_latex(coeffs, var_name, highest_power):
        terms = []
        for i, c in enumerate(coeffs):
            power = highest_power - i
            if abs(c) < 1e-14:
                continue
            c_str = f"{c:+.6g}" if len(terms) > 0 or c < 0 else f"{c:.6g}"
            if power > 0:
                terms.append(f"{c_str} \\cdot z^{{{power}}} {var_name}")
            else:
                terms.append(f"{c_str} \\cdot {var_name}")
        return " ".join(terms) if terms else "0"

    lhs_latex = _poly_terms_latex(den_coeffs_z_norm, "U(z)", order_den)
    rhs_latex = _poly_terms_latex(num_coeffs_z_norm, "E(z)", order_num)

    step3 = (
        f"**STEP 3 — Prodotto Incrociato**\n\n"
        f"$$U(z) \\cdot D(z) = E(z) \\cdot N(z)$$\n\n"
        f"$${lhs_latex} = {rhs_latex}$$\n\n"
    )
    steps.append(step3)

    # ── STEP 4: Antitrasformata z → k ─────────────────────────────────────
    def _time_terms(coeffs, var_name, highest_power):
        terms = []
        for i, c in enumerate(coeffs):
            power = highest_power - i
            if abs(c) < 1e-14:
                continue
            c_str = f"{c:+.6g}" if len(terms) > 0 or c < 0 else f"{c:.6g}"
            if power > 0:
                terms.append(f"{c_str} \\cdot {var_name}[k{power:+d}]")
            else:
                terms.append(f"{c_str} \\cdot {var_name}[k]")
        return " ".join(terms) if terms else "0"

    lhs_time = _time_terms(den_coeffs_z_norm, "u", order_den)
    rhs_time = _time_terms(num_coeffs_z_norm, "e", order_num)

    step4 = (
        f"**STEP 4 — Antitrasformata (dominio temporale discreto)**\n\n"
        f"Regola: $z^n \\cdot X(z) \\Rightarrow x[k+n]$\n\n"
        f"$${lhs_time} = {rhs_time}$$\n\n"
    )
    steps.append(step4)

    # ── STEP 5: Causalità ─────────────────────────────────────────────────
    max_order = max(order_den, order_num)

    def _causal_terms(coeffs, var_name, highest_power, shift):
        terms = []
        for i, c in enumerate(coeffs):
            power = highest_power - i
            k_idx = power - shift
            if abs(c) < 1e-14:
                continue
            c_str = f"{c:+.6g}" if len(terms) > 0 or c < 0 else f"{c:.6g}"
            if k_idx > 0:
                terms.append(f"{c_str} * {var_name}[k+{k_idx}]")
            elif k_idx == 0:
                terms.append(f"{c_str} * {var_name}[k]")
            else:
                terms.append(f"{c_str} * {var_name}[k{k_idx}]")
        return terms

    lhs_causal = _causal_terms(den_coeffs_z_norm, "u", order_den, max_order)
    rhs_causal = _causal_terms(num_coeffs_z_norm, "e", order_num, max_order)

    # Isola u[k]
    rhs_full = list(rhs_causal)
    for term in lhs_causal[1:]:
        if term.startswith("+"):
            rhs_full.append("-" + term[1:])
        elif term.startswith("-"):
            rhs_full.append("+" + term[1:])
        else:
            rhs_full.append("-" + term)

    diff_eq_str = "u[k] = " + " ".join(rhs_full)

    # LaTeX per step 5
    rhs_full_latex = []
    for term in rhs_causal:
        rhs_full_latex.append(term.replace("*", r" \cdot "))
    for term in lhs_causal[1:]:
        neg = term.replace("*", r" \cdot ")
        if neg.startswith("+"):
            rhs_full_latex.append("-" + neg[1:])
        elif neg.startswith("-"):
            rhs_full_latex.append("+" + neg[1:])
        else:
            rhs_full_latex.append("-" + neg)

    eq_latex = "u[k] = " + " ".join(rhs_full_latex)

    step5 = (
        f"**STEP 5 — Causalità (Equazione alle Differenze Implementabile)**\n\n"
        f"Dopo aver ritardato di {max_order} passi e isolato $u[k]$:\n\n"
        f"$${eq_latex}$$\n\n"
    )

    # ── Genera codice C/C++ ───────────────────────────────────────────────
    c_lines = []
    c_lines.append(f"// Equazione alle differenze — Metodo: {method_name}")
    c_lines.append(f"// Ts = {Ts_val} s")
    c_lines.append("")
    c_lines.append("// Variabili globali (persistenti tra i campioni)")

    max_u_delay = 0
    max_e_delay = 0
    for i, c in enumerate(den_coeffs_z_norm):
        delay = max_order - (order_den - i)
        if abs(c) > 1e-14 and delay > 0:
            max_u_delay = max(max_u_delay, delay)
    for i, c in enumerate(num_coeffs_z_norm):
        delay = max_order - (order_num - i)
        if abs(c) > 1e-14:
            max_e_delay = max(max_e_delay, delay)

    if max_u_delay > 0:
        c_lines.append(f"static double u_prev[{max_u_delay}] = {{0}};")
    if max_e_delay > 0:
        c_lines.append(f"static double e_prev[{max_e_delay}] = {{0}};")
    c_lines.append("")
    c_lines.append("double compute_control(double e_k) {")

    u_parts = []
    for i, c in enumerate(num_coeffs_z_norm):
        delay = max_order - (order_num - i)
        if abs(c) < 1e-14:
            continue
        if delay == 0:
            u_parts.append(f"({c:.10g}) * e_k")
        else:
            u_parts.append(f"({c:.10g}) * e_prev[{delay - 1}]")

    for i, c in enumerate(den_coeffs_z_norm[1:], start=1):
        delay = max_order - (order_den - i)
        if abs(c) < 1e-14:
            continue
        u_parts.append(f"({-c:.10g}) * u_prev[{delay - 1}]")

    c_lines.append("    double u_k = " + "\n                  + ".join(u_parts) + ";")
    c_lines.append("")

    if max_u_delay > 1:
        c_lines.append("    // Shift buffer u")
        for d in range(max_u_delay - 1, 0, -1):
            c_lines.append(f"    u_prev[{d}] = u_prev[{d-1}];")
    if max_u_delay > 0:
        c_lines.append("    u_prev[0] = u_k;")

    if max_e_delay > 1:
        c_lines.append("    // Shift buffer e")
        for d in range(max_e_delay - 1, 0, -1):
            c_lines.append(f"    e_prev[{d}] = e_prev[{d-1}];")
    if max_e_delay > 0:
        c_lines.append("    e_prev[0] = e_k;")

    c_lines.append("")
    c_lines.append("    return u_k;")
    c_lines.append("}")

    c_code = "\n".join(c_lines)
    step5 += f"**Codice C/C++ pronto per microcontrollore:**\n\n```c\n{c_code}\n```\n"
    steps.append(step5)

    return {
        'steps': steps,
        'cz_num_coeffs': num_coeffs_z_norm,
        'cz_den_coeffs': den_coeffs_z_norm,
        'diff_eq_str': diff_eq_str,
        'c_code': c_code,
        'method_name': method_name,
    }


def render_discretization_section(info: SystemInfo, dark_mode: bool) -> None:
    """Renderizza la sezione UI di discretizzazione."""
    st.markdown("---")
    st.subheader("🔄 Discretizzazione C(s) → C(z) → u[k]")
    st.caption(
        "Converti la funzione di trasferimento continua in discreta "
        "e ottieni l'equazione alle differenze causale per microcontrollore."
    )

    col_ts, col_method = st.columns(2)
    with col_ts:
        Ts_input = st.number_input(
            "Periodo di campionamento Tₛ [s]",
            min_value=1e-8,
            value=0.01,
            step=0.001,
            format="%.6f",
            key="disc_ts",
        )
    with col_method:
        method_label = st.selectbox(
            "Metodo di discretizzazione",
            options=[
                "Bilineare (Tustin)",
                "Rapporto in Avanti (Forward Euler)",
                "Rapporto all'Indietro (Backward Euler)",
            ],
            key="disc_method",
        )
    method_map = {
        "Bilineare (Tustin)": "tustin",
        "Rapporto in Avanti (Forward Euler)": "forward",
        "Rapporto all'Indietro (Backward Euler)": "backward",
    }
    method_key = method_map[method_label]

    btn_disc = st.button("⚙️ Discretizza", key="btn_discretize", type="primary")

    if btn_disc:
        try:
            result = discretize_tf(
                info.num_expr, info.den_expr,
                Ts_val=Ts_input,
                method=method_key,
            )
            st.session_state.disc_result = result
        except Exception as exc:
            import logging
            logging.error(f"Discretization error: {exc}", exc_info=True)
            st.error(f"⚠️ Errore nella discretizzazione: {exc}")
            st.session_state.disc_result = None

    if st.session_state.get('disc_result') is not None:
        result = st.session_state.disc_result
        for step_text in result['steps']:
            st.markdown(step_text)
            st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Punto di ingresso principale dell'applicazione."""

    with st.sidebar:
        try:
            st.image(logo_path, width=120)
        except Exception:
            pass
        st.header("⚙️ Informazioni Sistema")
        
        dark_mode = st.toggle("🌙 Modalità Scura", value=False)
        plotly_template = "plotly_dark" if dark_mode else "plotly_white"
        
        st.divider()
        
        if st.session_state.get('analizzato', False):
            poles = st.session_state['poles']
            zeros = st.session_state['zeros']
            order = st.session_state['order']
            K_static = st.session_state['K_static']
            
            st.metric("Ordine", order)
            if K_static is None:
                st.metric("Guadagno Statico K", "∞")
            else:
                st.metric("Guadagno Statico K", f"{K_static:.4f}" if not np.isinf(K_static) else "∞")
            
            st.subheader("Poli")
            if len(poles) > 0:
                for i, p in enumerate(poles):
                    if np.abs(p.imag) < 1e-8:
                        st.write(f"p{i+1} = {p.real:.4f}")
                    else:
                        st.write(f"p{i+1} = {p.real:.4f} ± {np.abs(p.imag):.4f}j")
            else:
                st.write("Nessun polo finito")
            
            if len(zeros) > 0:
                st.subheader("Zeri")
                for i, z in enumerate(zeros):
                    if np.abs(z.imag) < 1e-8:
                        st.write(f"z{i+1} = {z.real:.4f}")
                    else:
                        st.write(f"z{i+1} = {z.real:.4f} ± {np.abs(z.imag):.4f}j")
        else:
            st.info("Inserisci i coefficienti e premi Analizza")

    # Logo e Titolo in alto allineati
    logo_col, title_col = st.columns([1, 8])
    with logo_col:
        try:
            st.image(logo_path, width="stretch")
        except Exception:
            st.markdown("## 📈")
    with title_col:
        st.markdown(
            "<h1 style='margin-top: 0px; margin-bottom: 0px;'>Analizzatore Interattivo Bode & Nyquist</h1>",
            unsafe_allow_html=True
        )
    
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

    from components.mathlive_input import mathlive_input

    st.markdown("### Inserisci G(s)")
    st.markdown(
        "Scrivi le espressioni in forma naturale — "
        "**non servono asterischi** tra coefficiente e variabile. "
        "Esempi validi:"
    )

    # Esempi rapidi mostrati come codice inline
    ex_col1, ex_col2, ex_col3 = st.columns(3)
    with ex_col1:
        st.code("s*(1+s/10)", language=None)
        st.caption("Forma di Bode")
    with ex_col2:
        st.code("(s+2)^2", language=None)
        st.caption("Forma di Evans")
    with ex_col3:
        st.code("\\frac{s+1}{s^2+3s+2}", language=None)
        st.caption("LaTeX diretto")

    st.divider()

    col_num, col_den = st.columns(2)

    with col_num:
        num_str = mathlive_input(
            label="Numeratore N(s)",
            default_value="s*(1+s/10)",
            key="numeratore"
        )

    with col_den:
        den_str = mathlive_input(
            label="Denominatore D(s)",
            default_value="(1+s)*(1+s/100)",
            key="denominatore"
        )


    # Toggle unità fase (persiste nello stato)
    phase_unit = st.sidebar.radio(
        "Unità fase",
        ["rad", "°"],
        horizontal=True,
    )
    phase_in_radians = phase_unit == "rad"

    
    if dark_mode:
        st.markdown(_DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(_LIGHT_CSS, unsafe_allow_html=True)

    analyze_clicked = st.button("🔍 Analizza", type="primary")

    if analyze_clicked:
        st.session_state.analyzed = True

    if not st.session_state.analyzed:
        st.info("Inserisci una funzione di trasferimento e premi **Analizza**.")
        return

    # ── Parsing ───────────────────────────────────────────────────────────
    import logging
    try:
        with st.spinner("Calcolo in corso..."):
            info = parse_transfer_function(num_str, den_str)
    except Exception as exc:
        logging.error(f"Parsing error: {exc}", exc_info=True)
        st.error("⚠️ Sintassi non valida o errore nel parsing. Esempio corretto: `s**2 + 3*s + 2`")
        st.stop()

    # ── Risposta in frequenza ─────────────────────────────────────────────
    omega = _compute_omega_range(info, n_points=500)
    # Nyquist usa 1000 punti
    omega_ny = _compute_omega_range(info, n_points=1000)

    _show_sidebar_info(info, omega)

    try:
        with st.spinner("Calcolo in corso..."):
            # Bode
            resp = info.tf(1j * omega)
            mag = np.abs(resp)
            mag_db = 20.0 * np.log10(np.where(mag > 0, mag, 1e-30))
            phase_deg = np.degrees(np.unwrap(np.angle(resp)))
            # Nyquist
            resp_ny = info.tf(1j * omega_ny)
    except Exception as exc:
        logging.error(f"Calcolo esatto error: {exc}", exc_info=True)
        st.error("⚠️ Errore nel calcolo esatto. Verifica che il denominatore non abbia radici a zero esatto.")
        st.stop()

    try:
        approx_mag_db, approx_phase_deg = compute_approximated_bode(omega, info)
    except Exception as exc:
        logging.error(f"Calcolo approssimato error: {exc}", exc_info=True)
        st.warning("⚠️ Calcolo degli asintoti non riuscito. Viene mostrato solo il diagramma esatto.")
        approx_mag_db, approx_phase_deg = None, None

    # ── Determina omega_min / omega_max dal vettore ────────────────────────
    omega_min_val = float(omega[0])
    omega_max_val = float(omega[-1])

    # ── Recupera risultato query precedente (session_state) ───────────────
    if "omega_query_result" not in st.session_state:
        st.session_state.omega_query_result = None

    query_pt = st.session_state.omega_query_result

    # ── Formula Analitica G(s) ────────────────────────────────────────────
    st.markdown("---")
    from sympy import latex, Poly, cancel, fraction, expand as sp_expand

    col_formula, col_info = st.columns([3, 1])
    with col_formula:
        st.markdown("**G(s) analizzata:**")
        # Problema 5: normalizza il segno in modo che il numeratore abbia leading coeff > 0
        s_sym = sympy.Symbol('s')
        numer_raw, denom_raw = fraction(cancel(info.num_expr / info.den_expr))
        numer_exp = sp_expand(numer_raw)
        denom_exp = sp_expand(denom_raw)
        try:
            leading_num = Poly(numer_exp, s_sym).all_coeffs()[0]
            if leading_num < 0:
                numer_exp = sp_expand(-numer_exp)
                denom_exp = sp_expand(-denom_exp)
        except Exception:
            pass
        latex_gs = r"\frac{" + latex(numer_exp) + r"}{" + latex(denom_exp) + r"}"
        st.latex(r"G(s) = " + latex_gs)
    with col_info:
        st.metric("Ordine", int(info.order))
        st.metric("Tipo", int(info.system_type))


    # ── Diagramma di Bode ─────────────────────────────────────────────────
    st.subheader("Diagramma di Bode")
    try:
        bode_fig = plot_bode(
            plotly_template,
            omega, mag_db, phase_deg,
            approx_mag_db, approx_phase_deg, info,
            phase_in_radians=phase_in_radians,
            cursor_omega=st.session_state.cursor_omega,
        )
        bode_fig = applica_tema_plotly(bode_fig, dark_mode)
        st.plotly_chart(bode_fig, width="stretch", config={"displaylogo": False})
    except Exception as exc:
        logging.error(f"Bode plot error: {exc}", exc_info=True)
        st.error("⚠️ Errore nella generazione del diagramma di Bode.")

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
    try:
        nyquist_fig = plot_nyquist(
            plotly_template,
            omega_ny, resp_ny,
            cursor_omega=st.session_state.cursor_omega,
            cursor_resp=cursor_resp_ny,
        )
        nyquist_fig = applica_tema_plotly(nyquist_fig, dark_mode)
        st.plotly_chart(nyquist_fig, width="stretch", config={"displaylogo": False})
    except Exception as exc:
        logging.error(f"Nyquist plot error: {exc}", exc_info=True)
        st.error("⚠️ Errore nella generazione del diagramma di Nyquist.")

    # ── Discretizzazione C(s) → C(z) → u[k] ──────────────────────────────
    render_discretization_section(info, dark_mode)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
