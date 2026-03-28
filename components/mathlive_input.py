import streamlit as st

def mathlive_input(label: str, default_value: str = "", key: str = "") -> str:
    """
    Campo di input matematico semplice e affidabile.
    Nessuna anteprima (rimossa perché mostrava solo testo grezzo).
    Il rendering della formula appare DOPO l'analisi, nella sezione
    principale, usando sympy.latex().
    """
    value = st.text_input(
        label=label,
        value=st.session_state.get(f"field_{key}", default_value),
        key=f"field_input_{key}",
        placeholder="es: s*(1+s/10)  oppure  (s+2)^2  oppure  s^2+3s+2",
        help=(
            "✏️ Scrivi in forma naturale — non servono asterischi:\n"
            "• 2s → interpretato come 2·s\n"
            "• s^2 → s²\n"
            "• (1+s/10)(1+s/5) → prodotto automatico\n"
            "• LaTeX: \\frac{s}{1+s} → supportato"
        )
    )
    st.session_state[f"field_{key}"] = value
    return value
