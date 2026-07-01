## How to Run

**1. Create and activate a conda environment** (recommended)
```bash
conda create -n age_gender python=3.11 -y
conda activate age_gender
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python -m streamlit run app.py
```

Upload a face image and view the prediction with full decision breakdown.

> **Note:** Using a conda environment is recommended to avoid dependency conflicts, since the project uses both TensorFlow and PyTorch. Using `python -m streamlit` ensures the app runs with the correct environment's packages.
