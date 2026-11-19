# GMS Vaja 2 – Obdelava oblaka točk (Leica MS60 / Leica Infinity)

Spletna aplikacija v Pythonu in Streamlitu za branje, vizualizacijo in obdelavo oblakov točk, posnetih z elektronskim tahimetrom **Leica MS60** in izvoženih iz **Leica Infinity** (predmet *Geodetski merski sistemi – Vaja 2*).

## Struktura projekta

```text
3d-scann/
├── app.py                  # Glavna Streamlit aplikacija
├── requirements.txt        # Odvisnosti (združljive s Streamlit Community Cloud)
├── data/                   # Mapa za surove izvožene oblake točk (.pts, .txt, .csv, .las, .laz)
│   └── .gitkeep
└── src/
    ├── __init__.py
    ├── io_handler.py       # Branje in shranjevanje oblakov točk (np.float64)
    ├── processing.py       # Predobdelava in čiščenje oblakov točk (2. korak)
    └── adjustment.py       # Izravnava geometrijskih teles (3. korak)
```

## Lokalni zagon

1. Ustvarite in aktivirajte virtualno okolje ter namestite odvisnosti:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Zaženite Streamlit aplikacijo:

```bash
streamlit run app.py
```

3. Surove oblake točk (`.pts`, `.txt`, `.csv`, `.las`, `.laz`) lahko shranite v mapo `data/` ali pa jih naložite neposredno prek stranske vrstice v aplikaciji.

## Objava na Streamlit Community Cloud

Aplikacija uporablja izključno lahke Python knjižnice iz `requirements.txt` (brez `open3d`) in je pripravljena za neposredno objavo na Streamlit Community Cloud z glavno datoteko `app.py`.
