# Parkinson's Disease Detection 🧠

This project uses a Decision Tree Classifier to predict Parkinson's disease based on biomedical voice measurements. The dataset is sourced from the UCI Machine Learning Repository.

---

## 📂 Dataset

- **Source:** [UCI Parkinson’s Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data)
- **Target variable:** `status` (1 = Parkinson's, 0 = Healthy)
- **Features:** 22 voice-related attributes (e.g., MDVP:Fo(Hz), jitter, shimmer, etc.)

---

## ⚙️ How It Works

1. **Data Preprocessing:**
   - Drops non-numeric `name` column.
   - Scales features using `StandardScaler`.

2. **Model Training:**
   - Trains a `DecisionTreeClassifier`.
   - Evaluates model accuracy on a test split.

3. **Persistence:**
   - Saves model and scaler using `pickle` for reuse.

4. **Prediction:**
   - Accepts new input, scales it, and predicts the probability of Parkinson's.

---
