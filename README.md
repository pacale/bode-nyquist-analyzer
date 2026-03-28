# 📈 Analizzatore Interattivo Bode & Nyquist

Applicazione Streamlit per l'analisi interattiva di funzioni di trasferimento G(s).
Fornisce diagrammi di Bode (esatto + approssimato) e diagrammi polari di Nyquist.

## Requisiti

- Python 3.10+

## Installazione

```bash
pip install -r requirements.txt
```

## Avvio

```bash
streamlit run app.py
```

## Funzionalità

- Input a frazione visiva (Numeratore / Denominatore)
- Diagramma di Bode: modulo (dB) e fase (°/π rad)
- Diagramma polare di Nyquist
- Curve esatta e approssimata su tutti i grafici
- Cursore frequenza interattivo sincronizzato
- Pannello laterale con poli, zeri, ordine, tipo e guadagno statico
- Interfaccia completamente in italiano
