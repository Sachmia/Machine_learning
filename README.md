# MNIST – Streamlit-app (rita en siffra)

## Deployad app

Här är den deployade appen:

- **Streamlit Cloud:** https://sara-hedberg-mnist.streamlit.app

## Beskrivning

Appen låter användaren rita en siffra med musen eller ladda upp en bild. En ML-modell tränad på MNIST predikterar siffran 0–9.

## Innehåll i repot

- `streamlit_app_mnist.py` – Streamlit-appen
- `models/` – sparad modell (`.joblib`)
- `Sara_kunskapskontroll_2.ipynb` – notebook med körningar och analys
- `requirements.txt` – dependencies

## Köra lokalt (valfritt)

```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app_mnist.py
```
