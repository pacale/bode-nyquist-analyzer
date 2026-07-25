import streamlit as st
import control as ctrl
import plotly.graph_objects as go
import numpy as np

def render_time_response_section(info, plotly_template="plotly_white"):
    st.subheader("⏱️ Risposta nel Tempo")
    
    st.markdown("Analizza la risposta al gradino e all'impulso del sistema.")
    
    # Scegli se anello aperto o chiuso
    loop_type = st.radio(
        "Configurazione di test:",
        ["Anello Aperto: G(s)", "Anello Chiuso (retroazione unitaria negativa): W(s) = G(s)/(1+G(s))"],
        horizontal=True
    )
    
    is_dark = plotly_template == "plotly_dark"
    theme_layout = dict(
        template=plotly_template,
        paper_bgcolor="#0e1117" if is_dark else "#f8f9fc",
        plot_bgcolor="#131720" if is_dark else "#ffffff",
        font=dict(color="#e8e9f3" if is_dark else "#1a1a2e"),
    )

    try:
        sys = info.tf

        # Gestione Ritardo di Tempo tramite approssimazione di Padé
        if getattr(info, 'time_delay', 0.0) > 0:
            num_pade, den_pade = ctrl.pade(info.time_delay, n=5)
            sys_delay = ctrl.TransferFunction(num_pade, den_pade)
            sys = sys * sys_delay
            
        is_closed = "Chiuso" in loop_type
        if is_closed:
            sys = ctrl.feedback(sys, 1)
        
        # Genera il vettore dei tempi e la risposta
        T_step, yout_step = ctrl.step_response(sys)
        T_imp, yout_imp = ctrl.impulse_response(sys)
        
        try:
            step_info = ctrl.step_info(sys)
        except Exception:
            step_info = {}
            
        col1, col2 = st.columns(2)
        
        with col1:
            fig_step = go.Figure()
            fig_step.add_trace(go.Scatter(x=T_step, y=yout_step, mode='lines', name='Risposta al gradino', line=dict(color='#4d9de0', width=2)))
            
            # Target line
            target_val = step_info.get('SteadyStateValue', yout_step[-1])
            if target_val is not None and not np.isnan(target_val) and not np.isinf(target_val):
                fig_step.add_hline(y=target_val, line_dash="dash", line_color="gray", opacity=0.5, annotation_text="Regime")
            
            fig_step.update_layout(
                title="Risposta al Gradino Unitario",
                xaxis_title="Tempo [s]",
                yaxis_title="Ampiezza",
                margin=dict(l=40, r=40, t=40, b=40),
                height=400,
                **theme_layout,
            )
            st.plotly_chart(fig_step, use_container_width=True, config={"displaylogo": False})
            
        with col2:
            fig_imp = go.Figure()
            fig_imp.add_trace(go.Scatter(x=T_imp, y=yout_imp, mode='lines', name="Risposta all'impulso", line=dict(color='#f4a261', width=2)))
            fig_imp.update_layout(
                title="Risposta all'Impulso",
                xaxis_title="Tempo [s]",
                yaxis_title="Ampiezza",
                margin=dict(l=40, r=40, t=40, b=40),
                height=400,
                **theme_layout,
            )
            st.plotly_chart(fig_imp, use_container_width=True, config={"displaylogo": False})
            
        # Mostriamo le metriche calcolate da control
        if step_info:
            st.markdown("#### Metriche Risposta al Gradino")
            m1, m2, m3, m4 = st.columns(4)
            ovsh = step_info.get('Overshoot', 0)
            m1.metric("Sovraelongazione (Overshoot)", f"{ovsh:.2f} %" if ovsh is not None else "N/A")
            
            rt = step_info.get('RiseTime', 0)
            m2.metric("Tempo di Salita (Rise Time)", f"{rt:.4f} s" if rt is not None and not np.isnan(rt) else "N/A")
            
            st_time = step_info.get('SettlingTime', 0)
            m3.metric("Tempo Assestamento", f"{st_time:.4f} s" if st_time is not None and not np.isnan(st_time) else "N/A")
            
            ss = step_info.get('SteadyStateValue', 0)
            m4.metric("Valore a Regime", f"{ss:.4f}" if ss is not None and not np.isnan(ss) else "N/A")
            
    except Exception as e:
        st.error(f"Errore nel calcolo della risposta nel tempo: {e}")
