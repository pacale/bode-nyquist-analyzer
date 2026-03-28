# 📈 Analizzatore Interattivo Bode & Nyquist

Applicazione Streamlit per l'analisi interattiva di funzioni di trasferimento G(s).
Fornisce diagrammi di Bode (esatto + approssimato) e diagrammi polari di Nyquist.

## Requisiti

- Python 3.10 o versioni successive

## Installazione delle Dipendenze

Per avviare l'app è necessario installare le librerie specificate in `requirements.txt`.
Esegui questo comando nel tuo terminale, preferibilmente all'interno di un virtual environment (venv):

```bash
pip install -r requirements.txt
```

## Come Avviare l'Applicazione

Una volta installate le dipendenze, puoi avviare l'interfaccia web di Streamlit eseguendo questo comando dalla root del progetto:

```bash
streamlit run app.py
```

Questo aprirà automaticamente una nuova scheda nel tuo browser predefinito, solitamente all'indirizzo `http://localhost:8501`.

## Funzionalità Principali

- Input a frazione visiva (Numeratore / Denominatore)
- Diagramma di Bode: modulo (dB) e fase (°/π rad)
- Diagramma polare di Nyquist
- **Interrogazione puntuale**: Tabella per confrontare valori esatti e approssimati a specifiche frequenze
- Curve esatta e approssimata su tutti i grafici
- Cursore frequenza interattivo sincronizzato
- Pannello laterale con poli, zeri, ordine, tipo e guadagno statico
- Interfaccia completamente in italiano

## Input matematico
Il campo di inserimento usa **MathLive** (caricato via CDN), lo stesso motore usato da editor matematici professionali.
- Clicca sul campo → appare la tastiera matematica
- Tab "Controlli" → fattori comuni per G(s): frazioni, s², (1+s/ω), ecc.
- Puoi digitare direttamente LaTeX se preferisci (es. `\frac{s}{1+s}`)
- Nessuna sintassi Python richiesta: scrivi la matematica come la scriveresti su carta
