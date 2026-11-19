# GMS Vaja 2 – Obdelava in MNK izravnava oblakov točk (Leica MS60)

Spletna aplikacija v Pythonu in Streamlitu za branje, predobdelavo, 2.5D volumetrijo in geodetsko izravnavo oblakov točk po metodi najmanjših kvadratov (MNK), posnetih z elektronskim tahimetrom **Leica MS60** in izvoženih iz programske opreme **Leica Infinity** (*Geodetski merski sistemi – Vaja 2*).

## Funkcionalnosti aplikacije

Aplikacija vse koordinate `(X, Y, Z)` obdeluje v 64-bitni natančnosti (`np.float64`) za ohranjanje polne natančnosti velikih geodetskih koordinat (npr. D96/TM) in je razdeljena na tri zavihke:

1. **Pregled in predobdelava (CloudCompare):**
   - branje formatov `.pts`, `.txt`, `.csv`, `.las` in `.laz` (samodejno zaznavanje ločil in preskakovanje glav ter vrstic s številom točk),
   - grobi prostorski izrez (Bounding Box po X, Y, Z) s ponastavitvijo,
   - statistično odstranjevanje osamelcev (**SOR filter** prek `cKDTree`),
   - avtomatska izolacija največjega povezanega objekta (**DBSCAN** z varnim vokselnim združevanjem pri oblakih > 50.000 točk),
   - prostorsko enakomerno (**Voxel**) ali naključno (**Target count**) redčenje oblaka,
   - interaktivni 3D predogled (Plotly) in prenos očiščenega oblaka v `.txt` (6 decimalnih mest).
2. **Izračun volumna (2.5D):**
   - projekcija očiščenega oblaka na 2D rastrsko mrežo (`grid_step`),
   - izračun skupnega volumna (m³ in dm³/L), dodanega in odvzetega volumna glede na referenčno ravnino `ground_z`, površine tlorisa ter 2D toplotnega zemljevida višin (Heatmap).
3. **Izravnava po MNK (Gauss-Helmertov model `A·v + B·Δ = f`):**
   - akumulativno seštevanje normalnih enačb brez sestavljanja velikih matrik v pomnilniku (v lokalnem težiščnem sistemu),
   - podprti modeli: **Krogla**, **Ravnina** (brez singularnosti za navpične stene), **Pokončni valj** in **Pokončni stožec**,
   - opcijsko avtomatsko izločanje grobih napak (`|v| > 3·σ₀`) s ponovitvijo izravnave,
   - izpis aposteriornega standardnega odklona `σ₀`, RMSE, parametrov z ocenami `± σ` in kofaktorske matrike `Q_ΔΔ`,
   - 3D prikaz odstopanj (`RdBu_r`) s polprosojno izravnano ploskvijo (`go.Surface`) ter histogram odstopanj,
   - neposreden izvoz celotnega poročila (`.txt`) in odstopanj po točkah (`.csv`).

## Struktura projekta

```text
3d-scann/
├── app.py                  # Glavna Streamlit aplikacija (3 zavihki)
├── requirements.txt        # Lahke odvisnosti za Streamlit Community Cloud
├── data/                   # Mapa za surove izvožene oblake točk iz Leica Infinity
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── io_handler.py       # Branje in shranjevanje oblakov točk (np.float64)
│   ├── processing.py       # Predobdelava, filtriranje, segmentacija in 2.5D volumen
│   └── adjustment.py       # Geodetska izravnava po MNK (Gauss-Helmertov model)
└── tests/
    └── test_adjustment.py  # Enotski testi MNK izravnave
```

## Nalaganje datotek iz Leica Infinity

1. V **Leica Infinity** izvozite oblak točk v enem izmed podprtih formatov: **`.pts`**, **`.las`**, **`.laz`**, **`.txt`** ali **`.csv`** (priporočamo `.las` ali `.pts`).
2. Izvoženo datoteko lahko:
   - skopirate v mapo `data/` v tem repozitoriju (in jo po želji naložite na GitHub) – aplikacija jo bo samodejno zaznala v stranski vrstici pod možnostjo **Lokalna mapa data/**, ali
   - naložite neposredno v brskalniku prek možnosti **Naloži datoteko** v stranski vrstici.

## Lokalni zagon in testi

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Zagon enotskih testov MNK izravnave:

```bash
python -m unittest discover -s tests -v
```

## Objava na Streamlit Community Cloud

Aplikacija uporablja izključno lahke knjižnice iz [requirements.txt](requirements.txt) (brez `open3d`) in je pripravljena za neposredno objavo na **Streamlit Community Cloud** z glavno datoteko `app.py`.
