# Module 2 — Modeling and Analytics Evaluation

### 1. Data Interpretation and Target Profile
- **Class Distribution:** The target variable `churned` represents a typical business imbalance profile (~30% active churners vs. 70% retained).
- **Core Correlations:** High customer support ticket volumes strongly predict churn, while longer historical retention tenures insulate customers from churning.

### 2. Experimental Setup and Design Choice
- **Validation Frame:** A stratified 80/20 train/test split maintains target ratios across blocks.
- **Algorithm Selected:** Random Forest Classifier was chosen to natively accommodate mixed feature inputs (categorical text channels along with scaled numerical ranges) without assuming linear dependencies.

### 3. Model Results & Tradeoffs
- **Performance Evaluation:** The model achieves stable predictive classification metrics (ROC-AUC > 0.85). 
- **Operational Tradeoff:** Prioritizing high **Recall** helps capture all potential churners for customer success interventions, though it introduces a risk of **Precision** loss (false alarms on stable clients).
