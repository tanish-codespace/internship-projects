**Task1**
# Long Hair Gender Identification

A gender detection system with a twist: for people aged **20–30**, gender is decided by **hair length** (long hair → Female, short hair → Male) regardless of biological gender. Outside that age range, the system predicts true biological gender. Built as Task 1 of my data science internship.

---

## Problem Statement

Build a model that:
- Detects a **long-haired person aged 20–30 as Female**, even if male
- Detects a **short-haired person aged 20–30 as Male**, even if female
- For anyone **outside 20–30**, predicts their actual gender regardless of hair
- Includes a working **GUI**

---

## Datasets

| Dataset | Use |
|---|---|
| **UTKFace** (~23k images) | Age + gender labels stored in filenames; used to train the age/gender model and as the image source for hair labeling |
| **Custom Hair Dataset** (405 images) | Manually labeled long/short hair on UTKFace faces aged 20–30 — no public hair-length dataset exists, so I built one by hand |

---

## Methodology

**Two-model pipeline:**

```
Input image
     │
     ├─► Age + Gender CNN (TensorFlow/Keras) → estimates age & biological gender
     │
     └─► Hair Length Model (PyTorch ResNet18) → long / short
     │
     ▼
Decision logic:
  Age 20–30  → use hair rule (long=Female, short=Male)
  Otherwise  → use biological gender
```

1. **Age/Gender model** — custom CNN trained on UTKFace (grayscale 128×128). Retrained with **Random Erasing** augmentation to reduce a hair-based shortcut bias.
2. **Hair model** — ResNet18 fine-tuned on 405 manually labeled images with heavy augmentation.
3. **Decision logic** — plain Python applying the 20–30 age rule.
4. **GUI** — Streamlit app to upload an image and view age, hair length, and final gender with a decision breakdown.

---

## Results

- **Hair length model:** 95% test accuracy
- **Age/Gender model:** ~90% gender accuracy, ~6.3 years age MAE
- Correctly handles all four target cases (long-haired male → Female, short-haired female → Male, etc.)

---

## Limitations

- No public hair-length dataset exists → limited to 405 hand-labeled images
- UTKFace is imbalanced (few long-haired males / short-haired females), so rare cases are less reliable
- Age model struggles on professional/studio photos due to domain shift

---

## Tech Stack

TensorFlow/Keras · PyTorch · OpenCV · Streamlit · scikit-learn

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Upload a face image and view the prediction with full decision breakdown.

