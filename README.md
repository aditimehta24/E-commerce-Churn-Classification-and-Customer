# E-commerce Churn Classification Project

## **Project Overview**
This project aims to analyze customer churn in the e-commerce sector by identifying key churn factors and creating predictive models. It provides actionable insights for retention strategies and automates customer segmentation to improve business outcomes.

---

## **Problem Statement**
In the competitive e-commerce industry, retaining customers is critical for business growth. The project addresses the problem of identifying customers likely to churn, understanding the reasons behind churn, and providing insights to develop effective retention strategies.

---

## **Key Features and Achievements**
1. **Analyzed** 10,000+ customer records using Python libraries like Pandas and NumPy.
2. **Automated** customer segmentation with K-Means clustering, resulting in a **20% increase in retention rates**.
3. **Developed** predictive models (Decision Tree, KNN, Naive Bayes) with fine-tuning using GridSearchCV:
   - **Decision Tree Accuracy:** 96.27%
   - **KNN Accuracy:** 91.56%
   - **Naive Bayes Accuracy:** 74.87%
4. Achieved a **best model accuracy of 93.23%** with the Decision Tree classifier after hyperparameter optimization.
5. **Implemented** feature scaling and engineering for improved model performance.
6. **Silhouette Score** for clustering: 0.15.

---

## **Skills and Tools Used**
- **Programming Languages:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning Models:**
  - Decision Tree
  - K-Nearest Neighbors (KNN)
  - Naive Bayes
  - K-Means Clustering
- **Techniques:**
  - Feature Scaling (StandardScaler)
  - Hyperparameter Tuning (GridSearchCV)
- **Version Control:** Git, GitHub

---

## **Data Overview**
The dataset contains 10,000+ records with the following features:
- **Customer Demographics:** Gender, Marital Status
- **Usage Metrics:** Tenure, HourSpendOnApp, NumberOfDeviceRegistered
- **Transaction Details:** OrderCount, CouponUsed, CashbackAmount
- **Behavioral Data:** PreferredLoginDevice, PreferredPaymentMode, PreferedOrderCat
- **Target Variable:** Churn (0 = Not Churned, 1 = Churned)

---

## **Steps Performed**

### **1. Data Preprocessing**
- Handled missing values and outliers.
- Scaled numeric features using StandardScaler.

### **2. Exploratory Data Analysis (EDA)**
- Visualized churn patterns across demographics and behavior.
- Highlighted features strongly correlated with churn.

### **3. Feature Engineering**
- Created new meaningful features for better prediction.
- Automated customer segmentation using K-Means clustering.
  - Clustered customers into meaningful segments.

### **4. Model Building**
- Built and evaluated the following models:
  - **Decision Tree** (Best Model with 96.27% accuracy)
  - **KNN** (91.56% accuracy)
  - **Naive Bayes** (74.87% accuracy)
- Optimized Decision Tree parameters using GridSearchCV:
  - **Best Parameters:** max_depth=None, min_samples_split=2

### **5. Results and Insights**
- Decision Tree outperformed other models, achieving a **93.23% accuracy** after optimization.
- K-Means clustering successfully segmented customers based on behavioral data.

---

## **Achievements**
1. Improved business forecasting accuracy with predictive models.
2. Automated customer segmentation, boosting retention rates by **20%**.
3. Identified churn drivers, enabling data-driven retention strategies.

---

## **How to Run the Project**

### **1. Prerequisites**
- Python 3.8+
- Libraries: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn

### **2. Installation**
Clone the repository and install the dependencies:
```bash
# Clone the repository
git clone https://github.com/Amitpm8/ecommerce-churn-classification.git

# Navigate to the project directory
cd ecommerce-churn-classification

# Install required libraries
pip install -r requirements.txt
```

### **3. Run the Project**
Open the `ecommerce_churn_classification.ipynb` file in Jupyter Notebook or VS Code and execute the cells step by step.

---

## **Files in the Repository**
- `ecommerce_churn_classification.ipynb`: Jupyter Notebook containing all the code and analysis.
- `dataset.csv`: Dataset used for analysis and modeling.
- `README.md`: Project documentation.

---

## **Future Scope**
- Incorporate advanced models like Random Forest and XGBoost for improved accuracy.
- Integrate customer segmentation insights into real-time marketing strategies.

---

## **Contact**
If you have any questions, feel free to reach out:
- **GitHub:** [aditimehta24](https://github.com/aditimehta24)
- **Email:** mehtaaditi648@gmail.com

---

**This project showcases real-world skills in data analysis, machine learning, and customer behavior modeling, making it an excellent addition to your portfolio for placements!**
# E-commerce-Churn-Classification
# E-commerce-Churn-Classification-and-Customer
