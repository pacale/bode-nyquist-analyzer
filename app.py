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
from string import Template
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
st.set_page_config(
    page_title="Analizzatore Bode & Nyquist",
    page_icon="📈",
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
_MARGIN_PHASE_COLOR = "#2ecc71"   # verde per margine di fase
_MARGIN_GAIN_COLOR  = "#e74c3c"   # rosso per margine di ampiezza
_WC_COLOR  = "#2ecc71"            # verde per ωc
_WPI_COLOR = "#e74c3c"            # rosso per ωπ

# ── Temi (chiaro / scuro) ────────────────────────────────────────────────
# Un'unica sorgente di verità: la palette definisce sia il CSS di Streamlit
# sia i colori delle figure Plotly, così i due non possono divergere.

_PALETTE_LIGHT = {
    "app_bg": "#f7f8fc",
    "text": "#1a1a2e",
    "sidebar_bg": "#eef0f8",
    "sidebar_border": "#d0d4e8",
    "metric_val": "#1a3a6e",
    "metric_label": "#444466",
    "metric_delta": "#2a6a2a",
    "input_bg": "#ffffff",
    "input_border": "#c0c4d8",
    "input_focus": "#4d6cc0",
    "input_glow": "rgba(77, 108, 192, 0.20)",
    "code_bg": "#e8eaf0",
    "code_text": "#2a2a4a",
    "code_border": "#c8cae0",
    "caption": "#555577",
    "divider": "#d0d4e8",
    "radio_border": "#9090aa",
    "alert_bg": "#e8eaf6",
    "alert_border": "#5c6bc0",
    "chart_bg": "#ffffff",
    "chart_shadow": "rgba(0, 0, 0, 0.06)",
    "tab_active": "#1a1a2e",
    # Plotly
    "plot_bg": "#ffffff",
    "plot_paper": "#f8f9fc",
    "plot_grid": "#e8eaf0",
    "plot_zeroline": "#c5cae9",
    "plot_template": "plotly_white",
}

_PALETTE_DARK = {
    "app_bg": "#0e1117",
    "text": "#e8e9f3",
    "sidebar_bg": "#161a25",
    "sidebar_border": "#2a3040",
    "metric_val": "#7fb3f0",
    "metric_label": "#9aa0b8",
    "metric_delta": "#4ade80",
    "input_bg": "#1c2029",
    "input_border": "#333a4a",
    "input_focus": "#4d9de0",
    "input_glow": "rgba(77, 157, 224, 0.25)",
    "code_bg": "#1c2029",
    "code_text": "#d5d8e5",
    "code_border": "#333a4a",
    "caption": "#9aa0b8",
    "divider": "#2a3040",
    "radio_border": "#555f75",
    "alert_bg": "#1a2233",
    "alert_border": "#4d9de0",
    "chart_bg": "#131720",
    "chart_shadow": "rgba(0, 0, 0, 0.40)",
    "tab_active": "#e8e9f3",
    # Plotly
    "plot_bg": "#131720",
    "plot_paper": "#0e1117",
    "plot_grid": "#2a3040",
    "plot_zeroline": "#3d4560",
    "plot_template": "plotly_dark",
}

# I segnaposto usano la sintassi $nome di string.Template: il CSS contiene
# graffe e simboli % che romperebbero str.format() o il formatting con %.
_CSS_TEMPLATE = Template("""
<style>
/* === ROOT E BODY === */
html, body,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="block-container"],
.main,
.main .block-container {
    background-color: $app_bg !important;
    color: $text !important;
}

/* === HEADER === */
header[data-testid="stHeader"],
header[data-testid="stHeader"] * {
    background-color: $app_bg !important;
    color: $text !important;
}

/* === SIDEBAR === */
section[data-testid="stSidebar"],
section[data-testid="stSidebar"] > div:first-child {
    background-color: $sidebar_bg !important;
    border-right: 1px solid $sidebar_border !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div,
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] small,
section[data-testid="stSidebar"] * {
    color: $text !important;
}

/* === METRIC === */
[data-testid="stMetricValue"] {
    color: $metric_val !important;
    font-weight: 700 !important;
}
[data-testid="stMetricLabel"] {
    color: $metric_label !important;
    font-size: 0.78rem !important;
}
[data-testid="stMetricDelta"] {
    color: $metric_delta !important;
}

/* === INPUT === */
.stTextInput input,
.stTextInput textarea,
[data-testid="stTextInput"] input,
[data-testid="stTextInput"] textarea,
[data-testid="stNumberInput"] input,
div[data-baseweb="input"] input,
div[data-baseweb="textarea"] textarea {
    background-color: $input_bg !important;
    color: $text !important;
    border: 1px solid $input_border !important;
    border-radius: 8px !important;
}
.stTextInput input:focus,
[data-testid="stTextInput"] input:focus {
    border-color: $input_focus !important;
    box-shadow: 0 0 0 2px $input_glow !important;
    background-color: $input_bg !important;
}

/* === BOX ESEMPI / CODICE === */
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
    background-color: $code_bg !important;
    color: $code_text !important;
    border: 1px solid $code_border !important;
    border-radius: 6px !important;
}

/* === SELECT / DROPDOWN === */
div[data-baseweb="select"] > div {
    background-color: $input_bg !important;
    color: $text !important;
}
div[data-baseweb="popover"] div,
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] span {
    background-color: $input_bg !important;
    color: $text !important;
}

/* === PULSANTI === */
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

/* === TESTO === */
p, h1, h2, h3, h4, li {
    color: $text !important;
}
.stMarkdown p, .stMarkdown span {
    color: $text !important;
}
.stCaption, small, [data-testid="stCaptionContainer"] {
    color: $caption !important;
}

/* === TAB === */
button[data-baseweb="tab"] {
    color: $caption !important;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: $tab_active !important;
}

/* === DIVIDERS === */
hr {
    border-color: $divider !important;
}

/* === TOGGLE / RADIO === */
[data-testid="stToggle"] label,
[data-testid="stToggle"] span,
[data-testid="stRadio"] label,
[data-testid="stRadio"] span {
    color: $text !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label span {
    color: $caption !important;
}
[data-testid="stRadio"] div[role="radiogroup"] label div {
    border-color: $radio_border !important;
}

/* === ALERT === */
[data-testid="stAlert"], div[role="alert"], .stAlert {
    background-color: $alert_bg !important;
    color: $text !important;
    border-left: 4px solid $alert_border !important;
}

/* === TABELLE === */
.stApp table, .stApp th, .stApp td {
    color: $text !important;
    border-color: $divider !important;
}
.stApp thead th {
    background-color: $code_bg !important;
}

/* === GRAFICI === */
.stPlotlyChart, [data-testid="stPlotlyChart"] {
    background-color: $chart_bg !important;
    border-radius: 10px !important;
    box-shadow: 0 2px 8px $chart_shadow !important;
}

/* === SLIDER === */
[data-testid="stSlider"] * {
    color: $text !important;
}
</style>
""")

_LIGHT_CSS = _CSS_TEMPLATE.substitute(_PALETTE_LIGHT)
_DARK_CSS = _CSS_TEMPLATE.substitute(_PALETTE_DARK)


# ---------------------------------------------------------------------------
# Helper: applica il tema attivo alle figure Plotly
# ---------------------------------------------------------------------------
def applica_tema_plotly(fig: go.Figure, dark_mode: bool = False) -> go.Figure:
    """Applica colori e font coerenti col tema attivo a ogni figura Plotly."""
    pal = _PALETTE_DARK if dark_mode else _PALETTE_LIGHT
    font_col = pal["text"]
    grid_col = pal["plot_grid"]

    fig.update_layout(
        template=pal["plot_template"],
        paper_bgcolor=pal["plot_paper"],
        plot_bgcolor=pal["plot_bg"],
        font=dict(color=font_col, size=12, family="'Inter', 'Segoe UI', sans-serif"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=font_col, size=11),
            bordercolor="rgba(0,0,0,0)",
        ),
    )
    # Imposta il font del titolo solo se un titolo esiste davvero: farlo su
    # figure senza titolo porta plotly.js a renderizzare il testo "undefined".
    if fig.layout.title and fig.layout.title.text:
        fig.update_layout(title_font=dict(color=font_col, size=14))

    # I titoli dei subplot sono annotazioni: vanno ricolorati a parte.
    for ann in fig.layout.annotations or ():
        if ann.text and ann.font is not None and ann.xref == "x domain":
            ann.font.color = font_col

    # Aggiorna tutti gli assi presenti (gestisce subplot automaticamente)
    for key in fig.layout:
        obj = getattr(fig.layout, key, None)
        if obj is None:
            continue
        if hasattr(obj, "gridcolor"):
            obj.update(
                gridcolor=grid_col,
                zerolinecolor=pal["plot_zeroline"],
                zerolinewidth=1,
                tickfont=dict(color=font_col, size=11),
                title_font=dict(color=font_col, size=12),
                linecolor=grid_col,
                showgrid=True,
                spikecolor=_CURSOR_COLOR,
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
    time_delay: float = 0.0
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
        
    expr = num_expr / den_expr
    
    # --- Estrazione Ritardo di Tempo (Time Delay) ---
    time_delay = 0.0
    for arg in expr.atoms(sympy.exp):
        exponent = arg.args[0]
        if s in exponent.free_symbols:
            coeff = exponent.coeff(s)
            if coeff.is_number:
                # exp(-tau * s) -> coeff = -tau -> delay = -coeff
                time_delay -= float(coeff)
                expr = expr.subs(arg, 1)
                
    # Ricalcola numeratore e denominatore senza il ritardo
    num_expr, den_expr = sympy.fraction(sympy.cancel(expr))

    # Usa all_roots() per preservare la molteplicità dei poli/zeri
    # sympy.solve() restituisce solo radici distinte, perdendo i poli doppi
    try:
        num_poly = sympy.Poly(sympy.expand(num_expr), s)
        den_poly = sympy.Poly(sympy.expand(den_expr), s)
    except sympy.GeneratorsNeeded:
        num_poly = sympy.Poly(sympy.expand(num_expr), s, domain="ZZ")
        den_poly = sympy.Poly(sympy.expand(den_expr), s, domain="ZZ")

    # Estrai coefficienti e filtra artefatti numerici.
    # Soglia RELATIVA al coefficiente più grande: una soglia assoluta
    # azzererebbe coefficienti legittimi con frequenze di rottura molto alte
    # (es. (1+s/1e6)**2 → coeff. s² = 1e-12).
    num_coeffs = [float(c) for c in num_poly.all_coeffs()]
    den_coeffs = [float(c) for c in den_poly.all_coeffs()]

    num_scale = max(abs(c) for c in num_coeffs)
    den_scale = max(abs(c) for c in den_coeffs)
    num_coeffs = [c if num_scale and abs(c) > 1e-12 * num_scale else 0.0 for c in num_coeffs]
    den_coeffs = [c if den_scale and abs(c) > 1e-12 * den_scale else 0.0 for c in den_coeffs]

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
        time_delay=time_delay,
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

def compute_approximated_bode(
    omega: np.ndarray,
    info: 'SystemInfo',
) -> tuple[np.ndarray, np.ndarray]:
    """Calcola diagramma di Bode APPROSSIMATO (asintotico) corretto."""
    num = info.num_coeffs
    den = info.den_coeffs
    
    # 1. Trova zeri e poli
    zeros = np.roots(num)
    poles = np.roots(den)
    
    # Separa poli e zeri nell'origine
    tol = 1e-8
    z_origin = int(np.sum(np.abs(zeros) < tol))
    p_origin = int(np.sum(np.abs(poles) < tol))
    # g = tipo del sistema (poli nell'origine − zeri nell'origine)
    # Nella forma G(s) = K_b / s^g · ..., g è il numero NETTO di poli nell'origine
    g = p_origin - z_origin
    
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
    # Utilizziamo i coefficienti di grado minimo non nulli per preservare il segno
    b_k = next((c for c in reversed(num) if abs(c) > 1e-10), 0.0)
    a_h = next((c for c in reversed(den) if abs(c) > 1e-10), 1.0)
    K_b_real = float(b_k / a_h)
    K_bode_abs = abs(K_b_real)
    
    # Pendenza iniziale: −g·20 dB/dec (per poli nell'origine g>0 → pendenza negativa)
    mag_dB = 20 * np.log10(max(K_bode_abs, 1e-30)) - g * 20 * np.log10(omega)
    
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
    
    # ── Fase iniziale (Basile-Chiacchio) ────────────────────────────────────
    # G(s) = K_b / s^g · ...  →  per ω → 0+:
    #   fase da 1/s^g  →  −g · 90°
    #   fase da K_b < 0  →  −180°  (angolo di un numero reale negativo)
    phase_init = float(-g * 90.0)
    if K_b_real < 0:
        phase_init -= 180.0
    phase_deg = np.full(len(omega), phase_init)
    
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
    margins: Optional[dict] = None,
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

    # Sfondo delle etichette coerente col tema attivo
    _label_bg = (
        _PALETTE_DARK if plotly_template == "plotly_dark" else _PALETTE_LIGHT
    )["plot_bg"]

    fig.add_annotation(
        x=np.log10(omega_min),
        y=val_esatta,
        xref="x", yref="y1",
        text=f"<b>{val_esatta:.2f} dB</b>",
        showarrow=True, arrowhead=2, arrowcolor=_EXACT_COLOR,
        ax=-50, ay=0,
        font=dict(color=_EXACT_COLOR, size=12),
        bgcolor=_label_bg, bordercolor=_EXACT_COLOR, borderwidth=1, borderpad=4,
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
        bgcolor=_label_bg, bordercolor=_APPROX_COLOR, borderwidth=1, borderpad=4,
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

    # ── Linee guida ────────────────────────────────────────────────────────
    line_180_y = -1.0 if phase_in_radians else -180.0
    
    # ── Linea orizzontale di riferimento a 0 dB sul grafico del modulo ────
    fig.add_hline(
        y=0, line_width=1, line_dash="dot", line_color="#888",
        annotation_text="0 dB", annotation_position="right",
        annotation_font_size=10, annotation_font_color="#888",
        row=1, col=1,
    )

    # ── Marker e segmenti per margini di stabilità ────────────────────────
    if margins is not None:
        omega_c_val = margins["omega_gc"][0] if margins["omega_gc"] else None
        omega_pi_val = margins["omega_pc"][0] if margins["omega_pc"] else None
        pm_val = margins["PM_deg"]
        gm_val = margins["GM_dB"]
        touches_pi = bool(margins.get("omega_pc", []))

        # ωc: marker + segmento margine di fase
        if omega_c_val is not None:
            # Marker ωc sul modulo (a 0 dB)
            fig.add_trace(go.Scatter(
                x=[omega_c_val], y=[0],
                mode="markers+text",
                marker=dict(color=_WC_COLOR, size=12, symbol="diamond"),
                text=[f"ωc = {omega_c_val:.4g}"], textposition="top center",
                textfont=dict(size=10, color=_WC_COLOR),
                name="ωc (crossover 0 dB)",
                showlegend=True,
            ), row=1, col=1)

            # Linea verticale tratteggiata a ωc
            for row in (1, 2):
                fig.add_vline(
                    x=omega_c_val, line_width=1, line_dash="dash",
                    line_color=_WC_COLOR, row=row, col=1,
                )

            # Segmento margine di fase (sulla fase, da ∠G(jωc) a −180°)
            if pm_val is not None and touches_pi:
                phase_at_wc = float(np.interp(
                    np.log10(omega_c_val), np.log10(omega), phase_deg
                ))
                y_phase_wc = phase_at_wc / 180.0 if phase_in_radians else phase_at_wc

                fig.add_trace(go.Scatter(
                    x=[omega_c_val, omega_c_val],
                    y=[y_phase_wc, line_180_y],
                    mode="lines+text",
                    line=dict(color=_MARGIN_PHASE_COLOR, width=3),
                    text=[f"mφ = {pm_val:.1f}°", ""],
                    textposition="middle right",
                    textfont=dict(size=11, color=_MARGIN_PHASE_COLOR),
                    name=f"Margine di fase ({pm_val:.1f}°)",
                    showlegend=True,
                ), row=2, col=1)

                # Marker sulla fase a ωc
                fig.add_trace(go.Scatter(
                    x=[omega_c_val], y=[y_phase_wc],
                    mode="markers",
                    marker=dict(color=_WC_COLOR, size=10, symbol="circle"),
                    showlegend=False,
                ), row=2, col=1)

        # ωπ: marker + segmento margine di ampiezza
        if omega_pi_val is not None:
            # Marker ωπ sulla fase (a −180°)
            fig.add_trace(go.Scatter(
                x=[omega_pi_val], y=[line_180_y],
                mode="markers+text",
                marker=dict(color=_WPI_COLOR, size=12, symbol="diamond"),
                text=[f"ωπ = {omega_pi_val:.4g}"], textposition="top center",
                textfont=dict(size=10, color=_WPI_COLOR),
                name="ωπ (crossover −180°)",
                showlegend=True,
            ), row=2, col=1)

            # Linea verticale tratteggiata a ωπ
            for row in (1, 2):
                fig.add_vline(
                    x=omega_pi_val, line_width=1, line_dash="dash",
                    line_color=_WPI_COLOR, row=row, col=1,
                )

            # Segmento margine di ampiezza (sul modulo, da |G(jωπ)|_dB a 0 dB)
            if gm_val is not None:
                mag_at_wpi = float(np.interp(
                    np.log10(omega_pi_val), np.log10(omega), mag_db
                ))

                fig.add_trace(go.Scatter(
                    x=[omega_pi_val, omega_pi_val],
                    y=[mag_at_wpi, 0],
                    mode="lines+text",
                    line=dict(color=_MARGIN_GAIN_COLOR, width=3),
                    text=[f"mA = {gm_val:.1f} dB", ""],
                    textposition="middle right",
                    textfont=dict(size=11, color=_MARGIN_GAIN_COLOR),
                    name=f"Margine di ampiezza ({gm_val:.1f} dB)",
                    showlegend=True,
                ), row=1, col=1)

                # Marker sul modulo a ωπ
                fig.add_trace(go.Scatter(
                    x=[omega_pi_val], y=[mag_at_wpi],
                    mode="markers",
                    marker=dict(color=_WPI_COLOR, size=10, symbol="circle"),
                    showlegend=False,
                ), row=1, col=1)

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

    # Y modulo: auto-range; assicura che 0 dB sia sempre visibile
    y_mag_min = float(min(np.nanmin(mag_db), np.nanmin(approx_mag_db), 0))
    y_mag_max = float(max(np.nanmax(mag_db), np.nanmax(approx_mag_db), 0))
    fig.update_yaxes(
        title_text="Modulo (dB)",
        range=[y_mag_min - 15, y_mag_max + 15],
        showspikes=True, spikecolor="gray", spikethickness=1, spikedash="dot",
        row=1, col=1,
    )

    # Y fase
    ph_min = float(min(np.nanmin(exact_phase_plot), np.nanmin(approx_phase_plot)))
    ph_max = float(max(np.nanmax(exact_phase_plot), np.nanmax(approx_phase_plot)))
    # Assicura che −180° sia sempre visibile nel range
    if phase_in_radians:
        ph_min = min(ph_min, -1.0)
    else:
        ph_min = min(ph_min, -180.0)

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
    info: 'SystemInfo',
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

    # ── Archi all'infinito per sistemi con poli nell'origine (g > 0) ──────
    g = info.system_type
    if g > 0:
        # Raggio grande per la visualizzazione
        max_re = max(np.max(np.abs(re_pos)), 1.0)
        max_im = max(np.max(np.abs(im_pos)), 1.0)
        R = float(max(max_re, max_im)) * 1.5
        
        # L'arco all'infinito connette ω → 0- a ω → 0+
        # sul contorno di Nyquist il semicerchio salta l'origine
        # da -90° a +90° in senso antiorario nel piano s.
        # G(s) ≈ K / s^g, quindi s = r e^(jθ), θ ∈ [-π/2, π/2]
        # G(s) ≈ (K/r^g) e^(-j g θ)
        # Angolo iniziale (ω → 0-): θ = -π/2  →  fase G = +g π/2
        # Angolo finale (ω → 0+):   θ = +π/2  →  fase G = -g π/2
        # Senso orario nel piano G(s)
        
        theta_start = g * np.pi / 2
        theta_end = -g * np.pi / 2
        
        # Genera punti lungo l'arco
        # Usiamo np.linspace che include gli estremi, senso orario (start > end)
        theta_inf = np.linspace(theta_start, theta_end, 100)
        
        re_inf = R * np.cos(theta_inf)
        im_inf = R * np.sin(theta_inf)
        
        fig.add_trace(go.Scatter(
            x=re_inf, y=im_inf, mode="lines",
            name="Arco all'∞ (ω ≈ 0)",
            line=dict(color="#f39c12", width=2, dash="dashdot"),
            hovertemplate="Arco ∞<br>Re: %{x:.2f}<br>Im: %{y:.2f}<extra></extra>",
        ))
        
        # Freccia al centro dell'arco
        mid_idx = len(theta_inf) // 2
        dx = re_inf[mid_idx+1] - re_inf[mid_idx]
        dy = im_inf[mid_idx+1] - im_inf[mid_idx]
        fig.add_annotation(
            x=re_inf[mid_idx], y=im_inf[mid_idx],
            ax=re_inf[mid_idx] - dx * 8,
            ay=im_inf[mid_idx] - dy * 8,
            xref="x", yref="y", axref="x", ayref="y",
            showarrow=True,
            arrowhead=2, arrowsize=1.5,
            arrowwidth=1.5, arrowcolor="#f39c12",
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

def compute_stability_margins(
    sys_tf, omega: np.ndarray, info: 'SystemInfo' = None, time_delay: float = 0.0,
) -> dict:
    """Calcola i margini di stabilità con interpolazione logaritmica.

    Il ritardo puro e^(−sτ) va passato via *time_delay*: non è rappresentabile
    nella TransferFunction razionale ma riduce la fase (e quindi i margini).

    Ritorna un dizionario con:
    - omega_gc: lista di ωc (crossover di guadagno, |G| = 0 dB)
    - omega_pc: lista di ωπ (crossover di fase, ∠G = −180°)
    - GM_dB: margine di ampiezza in dB (None = ∞)
    - PM_deg: margine di fase in gradi (None = ∞)
    - stabile: True/False/None
    - bode_applicable: True se il criterio di Bode è applicabile
    - bode_reason: motivazione
    - N_encirclements: giri Nyquist attorno a (-1, 0)
    - P_unstable: poli a parte reale positiva
    - nyquist_stable: stabilità secondo Nyquist
    """
    import control
    freq_resp = control.frequency_response(sys_tf, omega)
    fresp = freq_resp.fresp.squeeze()
    if time_delay:
        fresp = fresp * np.exp(-1j * omega * time_delay)
    mag = np.abs(fresp)
    phase_deg = np.unwrap(np.angle(fresp, deg=False)) * 180.0 / np.pi
    mag_dB = 20 * np.log10(np.where(mag > 0, mag, 1e-12))

    result = {
        "omega_gc": [],
        "omega_pc": [],
        "GM_dB":    None,
        "PM_deg":   None,
        "stabile":  None,
        "bode_applicable": True,
        "bode_reason": "",
        "N_encirclements": 0,
        "P_unstable": 0,
        "nyquist_stable": None,
    }

    # ── ωc: crossover di guadagno (|G| = 0 dB) con interpolazione logaritmica ──
    for i in range(len(mag_dB) - 1):
        if mag_dB[i] * mag_dB[i+1] <= 0 and mag_dB[i] != mag_dB[i+1]:
            # Interpolazione lineare in dB, logaritmica in ω
            t = (0.0 - mag_dB[i]) / (mag_dB[i+1] - mag_dB[i])
            log_wc = np.log(omega[i]) + t * (np.log(omega[i+1]) - np.log(omega[i]))
            ogc = float(np.exp(log_wc))
            result["omega_gc"].append(ogc)

    # ── ωπ: crossover di fase (∠G = −180°) con interpolazione logaritmica ──
    phase_shifted = phase_deg + 180.0
    for i in range(len(phase_shifted) - 1):
        if phase_shifted[i] * phase_shifted[i+1] <= 0 and phase_shifted[i] != phase_shifted[i+1]:
            t = (0.0 - phase_shifted[i]) / (phase_shifted[i+1] - phase_shifted[i])
            log_wpc = np.log(omega[i]) + t * (np.log(omega[i+1]) - np.log(omega[i]))
            opc = float(np.exp(log_wpc))
            result["omega_pc"].append(opc)

    # ── Margine di fase: mφ = 180° + ∠G(jωc) ──
    if result["omega_gc"]:
        phase_at_wc = float(np.interp(
            np.log10(result["omega_gc"][0]), np.log10(omega), phase_deg
        ))
        result["PM_deg"] = 180.0 + phase_at_wc
    # Se ωc non esiste, PM = ∞ (None)

    # ── Margine di ampiezza: mA = −|G(jωπ)|_dB ──
    if result["omega_pc"]:
        mag_at_wpc = float(np.interp(
            np.log10(result["omega_pc"][0]), np.log10(omega), mag_dB
        ))
        result["GM_dB"] = -mag_at_wpc
    # Se ωπ non esiste, GM = ∞ (None)

    # ── Criterio di Bode: verifica applicabilità ──
    if info is not None:
        # 1. Nessun polo a parte reale positiva
        n_unstable = sum(1 for p in info.poles if p.real > 1e-10)
        result["P_unstable"] = n_unstable
        if n_unstable > 0:
            result["bode_applicable"] = False
            result["bode_reason"] = (
                f"Non applicabile: {n_unstable} polo/i a parte reale positiva."
            )
        
        # 2. K_b > 0 (calcolato internamente)
        if result["bode_applicable"]:
            b_k = next((c for c in reversed(info.num_coeffs) if abs(c) > 1e-10), 0.0)
            a_h = next((c for c in reversed(info.den_coeffs) if abs(c) > 1e-10), 1.0)
            K_b = float(b_k / a_h)
            if K_b < 0:
                result["bode_applicable"] = False
                result["bode_reason"] = "Non applicabile: guadagno di Bode K_b < 0."
        
        # 3. Unicità dell'intersezione a 0 dB
        if result["bode_applicable"] and len(result["omega_gc"]) > 1:
            result["bode_applicable"] = False
            result["bode_reason"] = (
                f"Non applicabile: {len(result['omega_gc'])} intersezioni con 0 dB "
                f"(richiesta unicità)."
            )

    # ── Stabilità secondo Bode ──
    if result["bode_applicable"]:
        gm_ok = result["GM_dB"] is None or result["GM_dB"] > 0
        pm_ok = result["PM_deg"] is None or result["PM_deg"] > 0
        result["stabile"] = gm_ok and pm_ok
        if result["stabile"]:
            result["bode_reason"] = "Stabile (mφ > 0, mA > 0 dB)."
        else:
            parts = []
            if result["PM_deg"] is not None and result["PM_deg"] <= 0:
                parts.append(f"mφ = {result['PM_deg']:.2f}° ≤ 0")
            if result["GM_dB"] is not None and result["GM_dB"] <= 0:
                parts.append(f"mA = {result['GM_dB']:.2f} dB ≤ 0")
            result["bode_reason"] = "Instabile: " + ", ".join(parts) + "."

    # ── Criterio di Nyquist: conteggio giri ──
    if info is not None:
        resp_full = fresp
        ref = -1.0 + 0j
        # Percorso completo: ω > 0 poi coniugato per ω < 0
        path_pos = resp_full - ref
        path_neg = np.conj(resp_full[::-1]) - ref
        full_path = np.concatenate([path_pos, path_neg])
        
        angles = np.angle(full_path)
        d_angles = np.diff(np.unwrap(angles))
        winding = np.sum(d_angles) / (2 * np.pi)
        N = int(np.round(winding))
        
        result["N_encirclements"] = N
        P = result["P_unstable"]
        result["nyquist_stable"] = (N == P)

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

    b_k = next((c for c in reversed(info.num_coeffs) if abs(c) > 1e-10), 0.0)
    a_h = next((c for c in reversed(info.den_coeffs) if abs(c) > 1e-10), 1.0)
    K_b = float(b_k / a_h)

    segno = "−" if K_b < 0 else "+"
    st.sidebar.metric("Costante di Bode |K_b|", f"{abs(K_b):.4f}")
    if K_b < 0:
        st.sidebar.caption("K_b < 0 → sfasamento iniziale di −180° incluso nella fase")

    margins = compute_stability_margins(
        info.tf, omega, info, time_delay=info.time_delay,
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 Stabilità")

    if margins["GM_dB"] is not None:
        st.sidebar.metric(
            label="Margine di Ampiezza (mA)",
            value=f"{margins['GM_dB']:.2f} dB",
            delta="stabile" if margins["GM_dB"] > 0 else "instabile",
            delta_color="normal" if margins["GM_dB"] > 0 else "inverse"
        )
    else:
        st.sidebar.metric("Margine di Ampiezza (mA)", "∞")
        st.sidebar.caption("ωπ non esiste → mA = ∞")

    touches_pi = bool(margins.get("omega_pc", []))
    if touches_pi:
        if margins["PM_deg"] is not None:
            st.sidebar.metric(
                label="Margine di Fase (mφ)",
                value=f"{margins['PM_deg']:.2f}°",
                delta="stabile" if margins["PM_deg"] > 0 else "instabile",
                delta_color="normal" if margins["PM_deg"] > 0 else "inverse"
            )
        else:
            st.sidebar.metric("Margine di Fase (mφ)", "∞")
            st.sidebar.caption("ωc non esiste → mφ = ∞")
    else:
        st.sidebar.caption("La fase non tocca mai -180° (mφ nascosto)")

    if margins["omega_gc"]:
        for i, ogc in enumerate(margins["omega_gc"]):
            label = "ωc (crossover 0 dB)" if len(margins["omega_gc"]) == 1 \
                    else f"ωc {i+1} (crossover 0 dB)"
            st.sidebar.metric(label=label, value=f"{ogc:.4f} rad/s")
    else:
        st.sidebar.metric("ωc (crossover 0 dB)", "—")
        st.sidebar.caption("Il modulo non interseca 0 dB")

    if margins["omega_pc"]:
        for i, opc in enumerate(margins["omega_pc"]):
            label = "ωπ (crossover −180°)" if len(margins["omega_pc"]) == 1 \
                    else f"ωπ {i+1} (crossover −180°)"
            st.sidebar.metric(label=label, value=f"{opc:.4f} rad/s")
    else:
        st.sidebar.metric("ωπ (crossover −180°)", "—")
        st.sidebar.caption("La fase non incrocia −180°")

    # Criterio di Bode
    st.sidebar.markdown("#### Criterio di Bode")
    if margins["bode_applicable"]:
        if margins["stabile"] is True:
            st.sidebar.success(f"✅ {margins['bode_reason']}")
        elif margins["stabile"] is False:
            st.sidebar.error(f"❌ {margins['bode_reason']}")
        else:
            st.sidebar.warning("⚠️ Stabilità indeterminata")
    else:
        st.sidebar.warning(f"⚠️ {margins['bode_reason']}")
    
    # Criterio di Nyquist
    st.sidebar.markdown("#### Criterio di Nyquist")
    N = margins.get("N_encirclements", 0)
    P = margins.get("P_unstable", 0)
    st.sidebar.caption(f"N (giri antiorari attorno a −1) = {N}, P (poli instabili) = {P}")
    if margins.get("nyquist_stable") is True:
        st.sidebar.success(f"✅ Stabile (N = P = {P})")
    elif margins.get("nyquist_stable") is False:
        st.sidebar.error(f"❌ Instabile (N = {N} ≠ P = {P})")
    else:
        st.sidebar.warning("⚠️ Nyquist non calcolato")


# ═══════════════════════════════════════════════════════════════════════════
# 8b. DISCRETIZZAZIONE C(s) → C(z) → u[k]
# ═══════════════════════════════════════════════════════════════════════════

def discretize_tf(info, Ts_val: float, method: str) -> dict:
    """
    Discretizza C(s) = num_expr_sym / den_expr_sym usando sostituzione algebrica.

    Parametri
    ---------
    info : SystemInfo (contiene num_expr, den_expr, poles, zeros)
    Ts_val  : periodo di campionamento (float > 0)
    method  : 'forward' | 'backward' | 'tustin' | 'matched'

    Restituisce
    -----------
    dict con chiavi: steps, cz_num_coeffs, cz_den_coeffs, diff_eq_str, c_code
    """
    s_sym = sympy.Symbol('s')
    z_sym = sympy.Symbol('z')
    T_sym = sympy.Symbol('T', positive=True)

    num_expr_sym = info.num_expr
    den_expr_sym = info.den_expr

    # ── STEP 1: Sostituzione ──────────────────────────────────────────────
    steps = []

    if method == 'matched':
        method_name = "Espansione Poli-Zeri (Matched Z-Transform)"
        import cmath
        mapped_zeros = [cmath.exp(z * Ts_val) for z in info.zeros]
        mapped_poles = [cmath.exp(p * Ts_val) for p in info.poles]
        
        d = len(info.poles) - len(info.zeros)
        if d > 0:
            mapped_zeros.extend([-1.0 + 0j] * (d - 1))
            
        def _build_poly(roots_list):
            poly = sympy.sympify(1.0)
            for r in roots_list:
                re, im = round(r.real, 12), round(r.imag, 12)
                if abs(im) < 1e-10:
                    poly *= (z_sym - re)
                else:
                    poly *= (z_sym - (re + im * sympy.I))
            return sympy.expand(poly)
            
        Cz_num_raw = _build_poly(mapped_zeros)
        Cz_den = _build_poly(mapped_poles)
        
        eps = 1e-6
        Cs_val = complex(info.expr.subs(s_sym, eps))
        Cz_unscaled_val = complex((Cz_num_raw / Cz_den).subs(z_sym, cmath.exp(eps * Ts_val)))
        Kd_real = float((Cs_val / Cz_unscaled_val).real)
        
        Cz_num = Cz_num_raw * Kd_real
        
        step1 = (
            f"**STEP 1 — Mappatura Poli e Zeri ({method_name})**\n\n"
            f"Mappatura: $z = e^{{s T_s}}$. Vengono aggiunti {max(0, d-1)} zeri in $z=-1$.\n"
            f"Guadagno statico aggiustato per la conservazione in frequenza.\n\n"
        )
        steps.append(step1)
    else:
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
        'Ts': Ts_val,
    }


def plot_bode_discreto(
    plotly_template: str,
    omega: np.ndarray,
    mag_db_cont: np.ndarray,
    phase_deg_cont: np.ndarray,
    mag_db_disc: np.ndarray,
    phase_deg_disc: np.ndarray,
    phase_in_radians: bool = False,
) -> go.Figure:
    """Diagramma di Bode di confronto: continuo vs discreto (fino a ω = π/Tₛ)."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Modulo (dB)", "Fase"),
    )

    fig.add_trace(go.Scatter(
        x=omega, y=mag_db_cont, mode="lines", name="Continuo",
        line=dict(color=_EXACT_COLOR, width=_EXACT_WIDTH),
        legendgroup="cont", showlegend=True,
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=omega, y=mag_db_disc, mode="lines", name="Discreto",
        line=dict(color=_MARGIN_GAIN_COLOR, width=_EXACT_WIDTH, dash="dash"),
        legendgroup="disc", showlegend=True,
    ), row=1, col=1)

    if phase_in_radians:
        phase_cont = phase_deg_cont / 180.0
        phase_disc = phase_deg_disc / 180.0
        phase_label = "Fase (×π rad)"
    else:
        phase_cont = phase_deg_cont
        phase_disc = phase_deg_disc
        phase_label = "Fase (°)"

    fig.add_trace(go.Scatter(
        x=omega, y=phase_cont, mode="lines", name="Continuo",
        line=dict(color=_EXACT_COLOR, width=_EXACT_WIDTH),
        legendgroup="cont", showlegend=False,
    ), row=2, col=1)

    fig.add_trace(go.Scatter(
        x=omega, y=phase_disc, mode="lines", name="Discreto",
        line=dict(color=_MARGIN_GAIN_COLOR, width=_EXACT_WIDTH, dash="dash"),
        legendgroup="disc", showlegend=False,
    ), row=2, col=1)

    log_min = np.log10(omega[0])
    log_max = np.log10(omega[-1])
    fig.update_xaxes(type="log", range=[log_min, log_max], row=1, col=1)
    fig.update_xaxes(
        type="log", range=[log_min, log_max],
        title_text="ω [rad/s]", row=2, col=1,
    )
    fig.update_yaxes(title_text="Modulo (dB)", row=1, col=1)
    fig.update_yaxes(title_text=phase_label, row=2, col=1)

    fig.update_layout(
        height=500,
        template=plotly_template,
        margin=dict(l=60, r=40, t=50, b=60),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
        ),
    )
    return fig


def _render_discrete_comparison(
    info: SystemInfo,
    result: dict,
    plotly_template: str,
    phase_in_radians: bool,
) -> None:
    """Confronto in frequenza C(s) vs C(z) e verifica dei poli discreti."""
    num_z = np.array(result['cz_num_coeffs'], dtype=float)
    den_z = np.array(result['cz_den_coeffs'], dtype=float)
    T = float(result['Ts'])

    # Range: come l'analisi continua, ma limitato alla frequenza di Nyquist π/T
    omega_full = _compute_omega_range(info, n_points=500)
    w_hi = min(float(omega_full[-1]), np.pi / T)
    w_lo = float(omega_full[0]) if float(omega_full[0]) < w_hi else w_hi / 1000.0
    omega_disc = np.logspace(np.log10(w_lo), np.log10(w_hi), 500)

    # Risposta discreta: z = e^(jωT)
    z_vals = np.exp(1j * omega_disc * T)
    resp_disc = np.polyval(num_z, z_vals) / np.polyval(den_z, z_vals)
    mag_disc = np.abs(resp_disc)
    mag_db_disc = 20.0 * np.log10(np.where(mag_disc > 0, mag_disc, 1e-30))
    phase_deg_disc = np.degrees(np.unwrap(np.angle(resp_disc)))

    # Risposta continua sulla stessa griglia (ritardo incluso)
    resp_cont = info.tf(1j * omega_disc) * np.exp(-1j * omega_disc * info.time_delay)
    mag_cont = np.abs(resp_cont)
    mag_db_cont = 20.0 * np.log10(np.where(mag_cont > 0, mag_cont, 1e-30))
    phase_deg_cont = np.degrees(np.unwrap(np.angle(resp_cont)))

    st.subheader("📉 Confronto in frequenza: C(s) vs C(z)")
    st.caption(
        f"Risposta valutata fino alla frequenza di Nyquist ω = π/Tₛ "
        f"≈ {np.pi / T:.4g} rad/s."
    )
    if info.time_delay > 0:
        st.warning(
            "⚠️ Il ritardo puro e^(−sτ) non è incluso in C(z): "
            "va aggiunto come ritardo di "
            f"{info.time_delay / T:.1f} campioni (≈ z^−{max(1, round(info.time_delay / T))})."
        )

    fig = plot_bode_discreto(
        plotly_template, omega_disc,
        mag_db_cont, phase_deg_cont,
        mag_db_disc, phase_deg_disc,
        phase_in_radians=phase_in_radians,
    )
    fig = applica_tema_plotly(fig, plotly_template == "plotly_dark")
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})

    # ── Poli del sistema discreto e stabilità (|p| < 1) ───────────────────
    st.markdown("#### Poli di C(z) e stabilità")
    poles_z = np.roots(den_z)
    if len(poles_z) == 0:
        st.write("Nessun polo finito.")
        return

    cols = st.columns(min(len(poles_z), 4))
    for i, p in enumerate(poles_z):
        modulo = abs(p)
        if abs(p.imag) < 1e-10:
            testo = f"{p.real:.4f}"
        else:
            testo = f"{p.real:.4f}{p.imag:+.4f}j"
        cols[i % len(cols)].metric(
            f"p{i + 1}", testo, delta=f"|p| = {modulo:.4f}",
            delta_color="normal" if modulo < 1 else "inverse",
        )

    if all(abs(p) < 1 - 1e-12 for p in poles_z):
        st.success("✅ Tutti i poli sono dentro il cerchio unitario: C(z) è asintoticamente stabile.")
    elif any(abs(p) > 1 + 1e-12 for p in poles_z):
        st.error("❌ Almeno un polo è fuori dal cerchio unitario: C(z) è instabile.")
    else:
        st.warning("⚠️ Poli sul cerchio unitario: C(z) è al limite di stabilità.")


def render_discretization_section(
    info: SystemInfo,
    plotly_template: str = "plotly_white",
    phase_in_radians: bool = False,
) -> None:
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
                "Espansione Poli-Zeri (Matched)",
                "Bilineare (Tustin)",
                "Rapporto in Avanti (Forward Euler)",
                "Rapporto all'Indietro (Backward Euler)",
            ],
            key="disc_method",
        )
    method_map = {
        "Espansione Poli-Zeri (Matched)": "matched",
        "Bilineare (Tustin)": "tustin",
        "Rapporto in Avanti (Forward Euler)": "forward",
        "Rapporto all'Indietro (Backward Euler)": "backward",
    }
    method_key = method_map[method_label]

    btn_disc = st.button("⚙️ Discretizza", key="btn_discretize", type="primary")

    if btn_disc:
        try:
            result = discretize_tf(
                info,
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

        try:
            _render_discrete_comparison(
                info, result, plotly_template, phase_in_radians,
            )
        except Exception as exc:
            import logging
            logging.error(f"Discrete comparison error: {exc}", exc_info=True)
            st.warning("⚠️ Confronto in frequenza non disponibile per questo sistema.")


# ═══════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    """Punto di ingresso principale dell'applicazione."""

    with st.sidebar:
        # Qui stanno solo le impostazioni; le informazioni sul sistema
        # vengono aggiunte più sotto da _show_sidebar_info().
        st.header("⚙️ Impostazioni")

        dark_mode = st.toggle(
            "🌙 Modalità scura",
            value=st.session_state.get("dark_mode", False),
            key="dark_mode",
        )
        plotly_template = "plotly_dark" if dark_mode else "plotly_white"

        st.divider()

        # Le informazioni dettagliate del sistema vengono aggiunte da
        # _show_sidebar_info() dopo il parsing; qui mostriamo solo il
        # suggerimento iniziale prima della prima analisi.
        if not st.session_state.get("analyzed", False):
            st.info("Inserisci i coefficienti e premi Analizza")

    # Il CSS del tema va applicato subito, prima di qualsiasi altro contenuto,
    # per evitare che la pagina lampeggi nel tema precedente.
    st.markdown(_DARK_CSS if dark_mode else _LIGHT_CSS, unsafe_allow_html=True)

    st.markdown(
        "<h1 style='margin-top: 0px; margin-bottom: 0px;'>Analizzatore Interattivo Bode &amp; Nyquist</h1>",
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

    analyze_clicked = st.button("🔍 Analizza", type="primary")

    if analyze_clicked:
        st.session_state.analyzed = True
        # Azzera i risultati derivati dal sistema precedente: si riferirebbero
        # a una G(s) diversa da quella appena analizzata.
        st.session_state.disc_result = None
        st.session_state.omega_query_result = None

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
    omega = _compute_omega_range(info, n_points=2000)
    # Nyquist usa 1000 punti
    omega_ny = _compute_omega_range(info, n_points=1000)

    _show_sidebar_info(info, omega)

    try:
        with st.spinner("Calcolo in corso..."):
            # Bode
            resp = info.tf(1j * omega) * np.exp(-1j * omega * info.time_delay)
            mag = np.abs(resp)
            mag_db = 20.0 * np.log10(np.where(mag > 0, mag, 1e-30))
            phase_deg = np.degrees(np.unwrap(np.angle(resp)))
            # Nyquist
            resp_ny = info.tf(1j * omega_ny) * np.exp(-1j * omega_ny * info.time_delay)
    except Exception as exc:
        logging.error(f"Calcolo esatto error: {exc}", exc_info=True)
        st.error("⚠️ Errore nel calcolo esatto. Verifica che il denominatore non abbia radici a zero esatto.")
        st.stop()

    try:
        approx_mag_db, approx_phase_deg = compute_approximated_bode(omega, info)
        # Allinea l'unwrap della fase esatta all'asintoto per evitare salti di 360° visivi
        if approx_phase_deg is not None:
            offset = approx_phase_deg[0] - phase_deg[0]
            n_360 = round(offset / 360.0)
            phase_deg += n_360 * 360.0
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


    tab_freq, tab_time, tab_rlocus, tab_disc = st.tabs([
        "📈 Bode & Nyquist", 
        "⏱️ Risposta nel Tempo", 
        "🌱 Luogo delle Radici", 
        "🔄 Discretizzazione"
    ])

    with tab_freq:
        # ── Diagramma di Bode ─────────────────────────────────────────────────
        st.subheader("Diagramma di Bode")
    
        # Calcola margini per annotazioni grafiche (ritardo incluso)
        margins_for_plot = compute_stability_margins(
            info.tf, omega, info, time_delay=info.time_delay,
        )
    
        try:
            bode_fig = plot_bode(
                plotly_template,
                omega, mag_db, phase_deg,
                approx_mag_db, approx_phase_deg, info,
                phase_in_radians=phase_in_radians,
                cursor_omega=st.session_state.cursor_omega,
                margins=margins_for_plot,
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
                omega_ny, resp_ny, info,
                cursor_omega=st.session_state.cursor_omega,
                cursor_resp=cursor_resp_ny,
            )
            nyquist_fig = applica_tema_plotly(nyquist_fig, dark_mode)
            st.plotly_chart(nyquist_fig, width="stretch", config={"displaylogo": False})
        except Exception as exc:
            logging.error(f"Nyquist plot error: {exc}", exc_info=True)
            st.error("⚠️ Errore nella generazione del diagramma di Nyquist.")


    with tab_time:
        from components.time_response import render_time_response_section
        render_time_response_section(info, plotly_template)

    with tab_rlocus:
        from components.root_locus import render_root_locus_section
        render_root_locus_section(info, plotly_template)

    with tab_disc:
        render_discretization_section(
            info, plotly_template, phase_in_radians=phase_in_radians,
        )


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
