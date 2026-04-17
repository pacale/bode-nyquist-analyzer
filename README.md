# 📈 Analizzatore Interattivo Bode & Nyquist

Un'applicazione avanzata sviluppata con **Streamlit**, **SymPy** e **python-control** per l'esplorazione e l'analisi interattiva delle funzioni di trasferimento $G(s)$ nel dominio della frequenza. 

Questo strumento didattico e professionale confronta le risposte esatte in frequenza con le approssimazioni asintotiche (metodo di Basile & Chiacchio), consentendo un'analisi profonda dei margini di stabilità e del comportamento del sistema in anello aperto e chiuso.

## 🚀 Funzionalità Principali

- **Diagrammi di Bode**: Plot interattivi per modulo e fase, con tracciamento simultaneo delle curve esatte (Control Theory) e di quelle asintotiche approssimate. Risolti tutti i problemi nativi di sfasamento su zeri o poli nel semipiano destro (RHP).
- **Diagrammi polari di Nyquist**: Resoconto visivo completo ed evidenziazione delle frequenze $\omega \to 0$ in modo nativo.
- **Supporto alla Scrittura Simbolica**: Input testuale gestito intelligentemente grazie ad un parsing iterativo con SymPy. Puoi scrivere forme espanse (es: `s^2 + 2*s + 1`) o parzialmente raggruppate. 
- **Tabella di Interrogazione Frequenziale**: Confronto numerico (errore) tra risposta esatta e approssimazione per frequenze fornite dall'utente.
- **Pannello di Analisi Automatico**: Elenco dettagliato del Tipo, Ordine, Zeri, Poli (classificati se complessi in smorzamento e pulsazione naturale $\omega_n$), e margini di fase/guadagno.
- **Dark Mode Dedicata & UI/UX Pulita**: Interfaccia moderna customizzata in formato light/dark.

---

## 🛠 Come scaricarlo ed eseguirlo in locale

Per usare il progetto sul tuo computer, assicurati di avere installato **Python 3.10** o una versione superiore.

### 1. Clona o Scarica il progetto
Puoi usare Git per scaricare questa repository:
```bash
git clone https://github.com/pacale/bode-nyquist-analyzer
cd bode-nyquist-analyzer
```
*(In alternativa, puoi scaricare il pacchetto ZIP dal sito GitHub ed estrarlo).*

### 2. Crea un ambiente virtuale (consigliato ma opzionale)
```bash
python -m venv venv
```
Attiva l'ambiente virtuale:
- Su **Windows**: `venv\Scripts\activate`
- Su **Mac/Linux**: `source venv/bin/activate`

### 3. Installa le dipendenze
Il file `requirements.txt` contiene tutte le librerie necessarie (come `streamlit`, `sympy`, `control`, `plotly`).
```bash
pip install -r requirements.txt
```

### 4. Avvia l'applicazione Streamlit
Dalla console nel path principale del progetto esegui:
```bash
streamlit run app.py
```
L'applicazione si aprirà automaticamente nel tuo browser predefinito all'indirizzo http://localhost:8501.

---


### Struttura del Codice Base
- `app.py`: Racchiude tutto il motore logico e il backend che gestisce l'interfaccia. Integra librerie Plotly, Control e il CSS.
- `.streamlit/config.toml`: Configurazioni native per disattivare watcher errati ed imporre temi base per compatibilità.
- `requirements.txt`: Elenco delle librerie che servono al progetto.
