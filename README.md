# Customer Churn Prediction Dashboard

A modern, professional Streamlit application that deploys a pretrained Decision Tree Classifier pipeline to predict customer churn risk for telecommunications subscribers.

---

## 🚀 Key Features

* **Premium Visual Design**: Leverages curated custom fonts (Google Font *Outfit*), deep slate theme gradients, modern cards, and dynamic risk badges (green/red for safe vs high churn likelihood).
* **Automated Feature Engineering**: Calculates customer loyalty, service call ratios, and revenue metrics in real-time before sending the data to the inference pipeline.
* **Responsive Column Layout**: Systematically organizes the 19 input metrics across multiple functional categories (Account Profile, Usage Minutes, Event Frequencies, Billing Charges, Support Interaction details).
* **Data Persistence**: Includes a capability to download the full customer profile and prediction score record as a structured CSV.

---

## 🛠️ Installation & Setup

1. **Clone or Navigate to Project Directory**
   Ensure absolute paths resolve to the project location.

2. **Install Dependencies**
   Install the required core libraries using `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run streamlit**
   Launch the web server locally:
   ```bash
   streamlit run app.py
   ```
   Open the browser at `http://localhost:8501`.

---

## 📋 Dependency Requirements

The primary packages required to execute the pipeline are listed inside `requirements.txt`:
* **`streamlit`** - Frontend web application interface
* **`pandas` & `numpy`** - Feature engineering and calculations
* **`scikit-learn`** - Standard Scaling and Classifier inference pipeline execution
* **`joblib`** - Model serialization loader

---

## ⚙️ Model Information & Pipeline Preprocessing

* **Algorithm**: Decision Tree Classifier
* **Performance Metrics**:
  * Accuracy: **94.60%**
  * Recall: **88.66%**
  * F1 Score: **82.69%**
* **Model Inputs**: Expects 73 standard features consisting of raw variables, 5 engineered features, and 50 state one-hot variables (with state AK acting as the base reference category).
