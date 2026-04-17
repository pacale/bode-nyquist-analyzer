# 📈 Analizzatore Interattivo Bode & Nyquist

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white) 
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

Un'applicazione avanzata per l'esplorazione e l'analisi interattiva delle funzioni di trasferimento $G(s)$ nel dominio della frequenza. Sviluppata per studenti di Automatica e Ingegneri, combina la potenza simbolica di **SymPy** con la precisione numerica di **python-control**.

---

## 🚀 Perché questo Analizzatore?

A differenza di molti simulatori standard, questo strumento è stato progettato per scopi didattici avanzati, seguendo rigorosamente il **metodo di Basile & Chiacchio**.

- **Accuratezza Asintotica**: Calcola i diagrammi approssimati (asintoti) gestendo correttamente i segni della costante di Bode ($K_b$) e le fasi iniziali.
- **Supporto Non-Minimum Phase**: Gestisce nativamente poli e zeri nel semipiano destro (RHP), applicando correttamente le inversioni di fase asintotiche.
- **Input Matematico Naturale**: Grazie all'integrazione con **MathLive**, puoi scrivere le formule come le scriveresti su carta, senza dover conoscere la sintassi Python/SymPy.

---

## ✨ Funzionalità

- **Diagrammi di Bode**: Modulo (dB) e Fase (gradi) con confronto immediato tra curva esatta (blu) e approssimata (arancione tratteggiata).
- **Diagramma di Nyquist**: Vista polare completa con evidenziazione del punto critico (-1, 0) e direzione di percorrenza.
- **Analisi Automatica**: Calcolo istantaneo di Tipo, Ordine, Zeri, Poli e parametri dei poli complessi ($\omega_n, \zeta$).
- **Tabella di Verifica**: Inserisci frequenze arbitrarie per ottenere i valori puntuali di modulo e fase ed eliminare ogni dubbio.
- **Interfaccia Dinamica**: Supporto completo Dark/Light mode con grafici Plotly che si adattano al tema scelto.

---

## 🛠 Installazione ed Esecuzione Locale

### 1. Clonazione
Scarica il progetto sul tuo computer:
```bash
git clone https://github.com/pacale/bode-nyquist-analyzer
cd bode-nyquist-analyzer
```

### 2. Preparazione Ambiente (Consigliato)
Crea un ambiente virtuale per mantenere il sistema pulito:
```bash
python -m venv venv
# Attivazione Windows:
venv\Scripts\activate
# Attivazione Mac/Linux:
source venv/bin/activate
```

### 3. Installazione Dipendenze
```bash
pip install -r requirements.txt
```

### 4. Avvio
```bash
streamlit run app.py
```

---

## ⌨️ Guida all'Inserimento Formule

L'app supporta l'inserimento tramite tastiera matematica virtuale:
- Puoi digitare `s^2 + 2s + 1` o usare i pulsanti per frazioni e radici.
- Il parser trasforma automaticamente il tuo LaTeX in espressioni simboliche elaborate.
- Esempio di funzione valida: `100 / (s*(1+s/10)*(s^2+s+1))`

---

## 📂 Struttura del Progetto

- `app.py`: Core logic, motore di calcolo asintotico e layout Streamlit.
- `requirements.txt`: Librerie dipendenti.
- `.streamlit/config.toml`: Ottimizzazioni per il server e configurazione porte.

---

## 📄 Licenza

Distribuito sotto Licenza MIT. Vedi `LICENSE` per maggiori informazioni.

---

*Sviluppato con ❤️ per il mondo dell'Ingegneria.*
