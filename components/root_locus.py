import streamlit as st
import control as ctrl
import plotly.graph_objects as go
import numpy as np

def render_root_locus_section(info, plotly_template="plotly_white"):
    st.subheader("🌱 Luogo delle Radici")
    st.markdown("Traccia il luogo delle radici (Root Locus) del sistema al variare del guadagno proporzionale $K > 0$.")
    
    try:
        sys = info.tf
        
        # Se c'è un ritardo, usiamo l'approssimazione di Padé per poter tracciare il luogo
        if getattr(info, 'time_delay', 0.0) > 0:
            st.info("⚠️ Il sistema presenta un ritardo puro. Il luogo delle radici mostrato usa un'approssimazione di Padé del 5° ordine.")
            num_pade, den_pade = ctrl.pade(info.time_delay, n=5)
            sys_delay = ctrl.TransferFunction(num_pade, den_pade)
            sys = sys * sys_delay
        
        # Generazione dati del luogo delle radici (usando grid=False evita il plot matplotlib)
        roots, gains = ctrl.root_locus(sys, plot=False)
        
        fig = go.Figure()
        
        if roots is not None and len(roots) > 0:
            # Le radici hanno shape (len(gains), len(poles))
            n_branches = roots.shape[1]
            for i in range(n_branches):
                branch = roots[:, i]
                fig.add_trace(go.Scatter(
                    x=np.real(branch),
                    y=np.imag(branch),
                    mode='lines',
                    name=f'Ramo {i+1}',
                    line=dict(width=2)
                ))
                
        # Segna i poli open loop (croci) e gli zeri (cerchi)
        poles = info.poles
        zeros = info.zeros
        
        if len(poles) > 0:
            fig.add_trace(go.Scatter(
                x=np.real(poles),
                y=np.imag(poles),
                mode='markers',
                marker=dict(symbol='x', size=10, color='red'),
                name='Poli O.L. (K=0)'
            ))
            
        if len(zeros) > 0:
            fig.add_trace(go.Scatter(
                x=np.real(zeros),
                y=np.imag(zeros),
                mode='markers',
                marker=dict(symbol='circle-open', size=10, color='blue', line=dict(width=2)),
                name='Zeri O.L. (K→∞)'
            ))
            
        is_dark = plotly_template == "plotly_dark"

        fig.update_layout(
            title="Luogo delle Radici per K > 0",
            xaxis_title="Asse Reale",
            yaxis_title="Asse Immaginario",
            template=plotly_template,
            margin=dict(l=40, r=40, t=40, b=40),
            height=500,
            paper_bgcolor="#0e1117" if is_dark else "#f8f9fc",
            plot_bgcolor="#131720" if is_dark else "#ffffff",
            font=dict(color="#e8e9f3" if is_dark else "#1a1a2e"),
        )

        # Linea asse immaginario e asse reale
        axis_col = "#8a92ab" if is_dark else "black"
        fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=axis_col, opacity=0.5)
        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color=axis_col, opacity=0.5)

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
        
    except Exception as e:
        st.error(f"Errore nel calcolo del luogo delle radici: {e}")
