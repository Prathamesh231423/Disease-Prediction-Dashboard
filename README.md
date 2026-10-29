# ❤️ Heart Disease Prediction Dashboard

A professional, responsive machine learning dashboard built with **Python**, **Pandas**, **Scikit-learn**, and **Streamlit**.

## Project Structure

```text
disease-prediction-dashboard/
├── app.py            # Streamlit interactive dashboard
├── train.py          # ML pipeline training and evaluation script
├── model.pkl         # Saved Scikit-learn pipeline (generated)
├── data/
│   └── heart.csv     # Heart disease dataset
├── tests/
│   └── test_project.py # Pytest test suite
├── requirements.txt  # Python package dependencies
├── .gitignore        # Git ignore rules
└── README.md         # Documentation
```

## Setup & Installation

1. **Clone or navigate to the project directory:**
   ```bash
   cd disease-prediction-dashboard
   ```

2. **Install dependencies:**
   ```bash
   python -m pip install -r requirements.txt
   ```

3. **Train the model:**
   ```bash
   python train.py
   ```
   *This loads `data/heart.csv`, trains a Scaled Random Forest pipeline with stratified splitting, evaluates performance, and serializes the model to `model.pkl`.*

4. **Run Automated Tests:**
   ```bash
   pytest
   ```

5. **Launch the Streamlit Dashboard:**
   ```bash
   streamlit run app.py
   ```

## Disclaimer
This application is developed strictly for **educational and demonstration purposes**. Predictions are not medical diagnoses. Consult a licensed medical professional for healthcare advice.
