# Predictive Analytics Lab Work

This repository contains 9 practical programs implementing fundamental predictive analytics techniques, from data preprocessing and exploratory analysis to advanced time series forecasting, clustering, and optimization algorithms.

## Programs Overview

### Program 1: Retail Sales Data Analysis
- Analyzed Online Retail dataset with 541,909 transactions
- Performed data cleaning: removed missing CustomerID, duplicates, and negative quantities
- Identified top 10 best-selling products and calculated total revenue by product
- Generated monthly revenue trend visualizations
- Conducted customer behavior analysis: unique customers, repeat customer percentage, country-wise segmentation
- Created comprehensive sales performance dashboards with heatmaps and bar charts
  

**Screenshot**

<img width="514" height="273" alt="image" src="https://github.com/user-attachments/assets/2fac5de4-de12-42c1-a957-2e72b51d5bdd" />
<img width="543" height="433" alt="image" src="https://github.com/user-attachments/assets/94994e6b-89c5-4ce8-9314-e39fd9ad0873" />
<img width="535" height="375" alt="image" src="https://github.com/user-attachments/assets/a3795a7a-86ad-4a4a-b16e-53fa0a950cf9" />


---

### Program 2: California Housing Price Prediction with Regularization
- Implemented Linear Regression, Ridge, and Lasso regression on California Housing dataset
- Performed exploratory data analysis with correlation heatmaps and feature distributions
- Applied StandardScaler for data normalization
- Conducted hyperparameter tuning using GridSearchCV with 5-fold cross-validation
- Compared model performance using RMSE, R-squared, and MAE metrics
- Ridge Regression achieved best performance (RMSE: 0.7456, R²: 0.5758)

**Screenshot**

<img width="395" height="323" alt="image" src="https://github.com/user-attachments/assets/cd7c5755-f784-40b0-92dd-49fbc10a45d8" />
<img width="510" height="288" alt="image" src="https://github.com/user-attachments/assets/d18aae2c-bff5-4922-b946-ba036735af41" />
<img width="575" height="456" alt="image" src="https://github.com/user-attachments/assets/78f08dc2-e698-46bc-9881-56318945248b" />


---

### Program 3: Diabetes Prediction with Multiple Classification Algorithms
- Analyzed Pima Indians Diabetes dataset with SMOTE for class imbalance handling
- Implemented 6 classification algorithms: Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest
- Performed comprehensive data visualization: histograms, scatter matrices, pairplots, heatmaps
- Applied data preprocessing: missing value imputation, MinMaxScaler normalization
- Conducted hyperparameter optimization for KNN (n_neighbors=24 optimal)
- Random Forest achieved highest accuracy: 78.5% after SMOTE balancing

**Screenshot**


<img width="762" height="567" alt="image" src="https://github.com/user-attachments/assets/c49b1462-4609-4256-b887-b302d0feec53" />
<img width="634" height="544" alt="image" src="https://github.com/user-attachments/assets/66dd9543-7709-487c-b42e-260cfd6dede8" />
<img width="722" height="525" alt="image" src="https://github.com/user-attachments/assets/69768b70-a068-4690-9e6f-915179f7d64f" />
<img width="506" height="391" alt="image" src="https://github.com/user-attachments/assets/65d7e4e1-070f-4375-abb2-ad1c9c0c39f9" />
<img width="565" height="261" alt="image" src="https://github.com/user-attachments/assets/25bf014a-9c8d-439b-b755-8a73cc548c84" />


---

### Program 4: Diabetes Prediction with Hyperparameter Tuning
- Analyzed larger diabetes dataset (100,000 records) with 9 features
- Performed extensive data preprocessing: label encoding, StandardScaler normalization
- Conducted correlation analysis showing HbA1c_level and blood_glucose_level as strongest predictors
- Implemented Logistic Regression with GridSearchCV hyperparameter tuning
- Tested parameters: C values, penalty types, solvers, max_iter, class_weight
- Achieved 95.89% accuracy with 5-fold cross-validation mean score: 0.9601

**Screenshot**

<img width="580" height="520" alt="image" src="https://github.com/user-attachments/assets/56ae9670-8ad2-492c-9be6-e314cd2acc2c" />
<img width="556" height="477" alt="image" src="https://github.com/user-attachments/assets/54c326ba-b0db-4a69-9474-df7df0acd5c4" />
<img width="585" height="205" alt="image" src="https://github.com/user-attachments/assets/ec3b60e0-2b4d-4dd4-96a1-d2434569dd8d" />
<img width="563" height="206" alt="image" src="https://github.com/user-attachments/assets/37610c4e-9d3a-40c5-b5fa-daefd9677d89" />
<img width="518" height="210" alt="image" src="https://github.com/user-attachments/assets/a3a6124b-7541-4d5f-a1fe-69923d034c9e" />


---

### Program 5: Student Performance Clustering Analysis
- Implemented K-Means and DBSCAN clustering on student performance dataset
- Applied PCA for dimensionality reduction and improved visualization
- Used Elbow Method and Silhouette Score to determine optimal clusters (k=2)
- Evaluated clustering performance with Silhouette Score and Davies-Bouldin Index
- PCA significantly improved clustering quality (Silhouette Score from 0.05 to 0.30)
- Identified two distinct student clusters: high-performing and struggling students

**Screenshot**

<img width="510" height="323" alt="image" src="https://github.com/user-attachments/assets/5f348d0e-55e6-439b-8b44-f83a5f5d7837" />
<img width="537" height="315" alt="image" src="https://github.com/user-attachments/assets/a2726b71-fa71-4278-b9af-57394c647d92" />
<img width="511" height="373" alt="image" src="https://github.com/user-attachments/assets/4e0d8554-eb38-4d6f-942e-36d5b30d1acd" />
<img width="508" height="500" alt="image" src="https://github.com/user-attachments/assets/a1c51945-715c-48a7-b626-20903eddd406" />

---

### Program 6: Time Series Analysis with Classical vs Advanced Models
- Analyzed Airline Passengers dataset for monthly passenger forecasting
- Implemented both classical (Decision Tree) and advanced (LSTM) time series models
- Created lag features for trend analysis and seasonal pattern detection
- Applied StandardScaler normalization and TimeseriesGenerator for sequence preparation
- LSTM model (75% accuracy) significantly outperformed Decision Tree (55% accuracy)
- Demonstrated LSTM's superior ability to capture seasonal patterns and long-term dependencies

**Screenshot**

<img width="524" height="340" alt="image" src="https://github.com/user-attachments/assets/fdd946d6-7fa1-4215-b19e-d87506b523d5" />
<img width="542" height="348" alt="image" src="https://github.com/user-attachments/assets/d98ec7dd-1ac0-442b-b382-5eb1de2ce7a9" />
<img width="456" height="152" alt="image" src="https://github.com/user-attachments/assets/646ef6b5-a58f-421a-928a-4b0a9d5eb79a" />
<img width="438" height="163" alt="image" src="https://github.com/user-attachments/assets/07a55fc7-0813-4b8b-9361-3f64733a758d" />


---

### Program 7: Comprehensive Time Series Forecasting with ARIMA/SARIMA
- Conducted in-depth time series analysis on Airline Passengers dataset
- Performed seasonal decomposition, ACF/PACF analysis, and spectral density analysis
- Implemented K-Means clustering for time series segmentation (3 clusters identified)
- Applied ARIMA and SARIMA models for forecasting with performance comparison
- SARIMA significantly outperformed ARIMA (RMSE: 31.79 vs 80.66, MAE: 25.27 vs 67.38)
- Generated comprehensive visualizations including clustering results and forecast comparisons

**Screenshot**

<img width="535" height="294" alt="image" src="https://github.com/user-attachments/assets/59a9f2a5-5daf-4b55-9aff-9cf861f05b47" />
<img width="547" height="234" alt="image" src="https://github.com/user-attachments/assets/c2e89613-6f8d-4143-8a0a-45bd76619611" />
<img width="489" height="707" alt="image" src="https://github.com/user-attachments/assets/3f298816-fdf9-41e1-9b64-2a9b533bed59" />
<img width="532" height="378" alt="image" src="https://github.com/user-attachments/assets/cf99dce9-e295-47f2-b871-8fd525b518a2" />
<img width="499" height="377" alt="image" src="https://github.com/user-attachments/assets/712c7004-cfec-427e-ae6d-1bb22936775f" />
<img width="541" height="301" alt="image" src="https://github.com/user-attachments/assets/34ebc532-e6b7-44bd-aaca-3bb15d8c63d8" />

---

### Program 8: Traveling Salesman Problem with Genetic Algorithm
- Implemented Genetic Algorithm using PyGAD for TSP optimization
- Defined 5-city problem with Euclidean distance matrix calculation
- Configured GA parameters: 500 generations, 5 parents mating, swap mutation
- Visualized optimal route with city coordinates and directional arrows
- Achieved optimal route: [1→0→3→2→4] with shortest distance: 44.14 units
- Demonstrated evolutionary algorithm effectiveness for combinatorial optimization

**Screenshot**


<img width="659" height="489" alt="image" src="https://github.com/user-attachments/assets/4683f912-b641-481c-b5f2-a7c645350432" />

---

### Program 9: Temperature Prediction with Genetic Algorithm Optimization
- Analyzed London Weather and Australia Weather datasets
- Implemented Random Forest regression for temperature prediction
- Applied TPOT (Tree-based Pipeline Optimization Tool) for automated machine learning
- Compared baseline models with GA-optimized pipelines
- GA-optimized models showed improved performance (R²: 0.9768 vs 0.9758 for London dataset)
- Generated comparative visualizations showing tighter prediction clustering with GA optimization

**Screenshot**

<img width="520" height="364" alt="image" src="https://github.com/user-attachments/assets/7fe5ea7e-ece9-4198-81b2-d2f8dc9132dd" />
<img width="555" height="340" alt="image" src="https://github.com/user-attachments/assets/ebb94ad5-5ba4-448a-bfa8-2fda3f878ec8" />
<img width="549" height="223" alt="image" src="https://github.com/user-attachments/assets/f4d46f08-44cb-4199-a76c-8d7047c5bd68" />
<img width="594" height="356" alt="image" src="https://github.com/user-attachments/assets/4de33a93-a344-4a39-a6ee-8c4fa789cc49" />


```
