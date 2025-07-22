
<h1 align="center">CUSTOMER CHURN PREDICTION FOR SyriaTel TELECOM</h1>

![](./Images/telheader.jfif)

Image By: [Freepik](https://www.pinterest.com/pin/380765343511077199/)

## Project Structure and Navigation

- *`README.md` — Project documentation*
- *`student.ipynb`  — Jupyter notebook for analysis*
- *`Images/` — Saved plots and figures*
- *`Data/` — Dataset files*
- *`Presentation/` — Project summary slides PDF*

## Table of Contents
1. *Business Understanding*
2. *Data Understanding*
3. *Data UPreparation*
4. *Exploratory Data Analysis (EDA)*
4. *Modelling*
4. *Model Evaluation*
4. *Conclusion and Recommendations*
5. *Links and Resources* - *( Data Source)*

# 1.0 Business Understanding

### 1.1 Business Context
SyriaTel is a major tel-com company that just like its competitiors, it experiences customer churn which significantly impact profitability hindering growth and reducing market share. Its also noted that acquairing new customers is more expensive than retaining existing ones.

### 1.2 Problem Statement
The company want to understand patterns leading to churn as well as be able to identify customers likely to leave in the near future for proper actions of retention stategies 

### 1.3 Project Objectives
To address the problem statement, this project aims to 
1. Predict customer Churn -  (churn = True) or not (churn = False)
2. Get factors influencing churn
3. Advise on key retention strategies.

#### 1.3.1 Success Metrics:
From a business perspective, we care most about:

- *Recall*: We want to catch as many churners as possible (even if we sometimes flag non-churners)
- *Precision*: We want to avoid too many false positives (wasting retention offers)
- *F1 Score*: A balance between precision and recall
- *ROC-AUC*: To understand how well our model separates churn vs non-churn

# 2.0 Data Understanding
The dataset was obtained from Kaggle, with 3,333 customer records , each with 21 feautures describing customer demographics, usage patterns, service plans, and support interactions. The target variable is churn (True/False).

#### Observations from info()
- No missing values accross the columns, all have 3,333 non-null values hence no imputation required
- 4 Columns are of Object (categorical) type(state,phone number,international plan and voice mail plan )
- 8 Columns are float type
- 8 Columns are int type
- 1 column is a bool- churn (target variable)
- The data has 3,333 rows and 21 columns.

# 3.0 DATA Preparation

### 3.1 Correlation matrix for Numeric Variables 
- we check for highly correlated features and drop some to avoid redundancy , multicolianility and simplicity.

![](./images/corr_matrix.png)

#### Observations

These features were dropped because they are **highly correlated** with other numeric features. Including both in the model can introduce **multicollinearity**, which may distort the model's ability to estimate the true effect of each variable.

| Dropped Feature         | Highly Correlated With       |
|-------------------------|------------------------------|
| `total day charge`      | `total day minutes`          |
| `total eve charge`      | `total eve minutes`          |
| `total night charge`    | `total night minutes`        |
| `total intl charge`     | `total intl minutes`         |

**Justification**:  
To reduce redundancy and simplify the model:
- Only one from each pair is retained, typically the one **more strongly correlated with the target (`churn`)** or **easier to interpret**.
- Dropping the redundant one helps prevent overfitting and improves interpretability.

### 3.2 Statistical Feature selection (SelectKBest)
- We apply Chi-square and ANOVA F-test to identify the most relevant features for churn prediction by evaluatin association with out target. 
- Chi-square was used for categorical variables to test the independence between features and the target.we drop those below 10 score
- ANOVA F-test was used for numeric variables to check whether feature means differ significantly across churn categories.we drop those less than 2

![](./images/SelectKBest.png)

#### Observation from Feature selection using SelectKBest
- International plan showed the strongest association with churn in categorical features but Voice mail plan was also significant with 25
- customer service calls and total day minutes had the highest F-scores, indicating strong influence on churn.
- the selected columns captures both plan types and usage behavior which is good for modeling. 

# 4.0 Explolatory Data Analysis (EDA)
### 4.1 Target distribution to check balance
![](./images/Churndistribution.png)

#### Observation
- The Target is inbalanced with only 14% churners. this will be handled by balancing the weight on the regression model to avoid it learning only to predict "NO CHURN"

### 4.2.0 Feature Distributions
 #### 4.2.1 Numerical Columns Data Distribution
![](./images/numcolsdistribution.png)
 
#### Observations from Numerical Feature Distributions
- **Customer Service Calls**: Most customers made 1–3 service calls, Frequest customer service calls may signal disatsfaction.

- **Total Day Minutes**: Follows a normal distribution, around 180–200 minutes.

- **Total Evening Minutes**: Slight Right-skewed, peaks around 200 minutes.

- **Number Vmail Messages**: Highly right-skewed with most customers having 0 messages, potential low predictive measure.

- **Total International Minutes**: Normally distributed around 10 minutes.

- **Total International Calls**: Most customers made 3–5 international calls; skewed right. May indicate Niche usage patterns

- **Total Night Minutes**: Near-normal distribution centered around 200 minutes.

#### 4.2.2 Categorical Columns Data Distribution
![](./images/catcolsdistibution.png)

#### Observations from Categorical Feature Distributions

- Most customers have no international plans
- Churn customers are less

# 5.0 Modelling and Evaluation

- i will start with logistic regression model and compare with Decision Tree Model to ascertain churn predictions

## **5.1 (LOGISTIC REGRESSION)**
### 5.1.1 Build Pipeline with Preprocessing + SMOTE + RFECV + GridSearch

- i will create a pipeline that handles **column Transforming** for numericand and categorical features using appropriate **scaling and encoding** 
- Since we noted above that our Target **Churn** is **inbalance**, i will include **SMOTE(Synthetic Minority Oversampling Technique)** to balance the classes to help model understand the minority. 
- Finaly i use **RFECV** to automatically *select the most relevant features* after processing and balancing to ensure we **reduce overfitting**.

### 5.1.2 Predictions/Evaluations (Logistic Regression)
![](./images/logistic_reg_performance.png)

### 5.1.3 Confusion Mattrix (Logistic Regression)
![](./images/log_reg_confusion_matrix.png)

### 5.1.4 ROC Curve -  (Logistic Regression)
![](./images/logistic_reg_ROC_CURVE.png)

## Observations from Logistic Regression Predictions

**Precision**:
- Out of all the predicted non-churners (Class 0), 93% were actually non-churners — excellent precision.
- Out of all the predicted churners (Class 1), only 31% were actually churners — low precision, meaning the model frequently predicts churners incorrectly. (High False Positives)

**Recall**:
- The model correctly identified 73% of actual non-churners.
- The model correctly identified 70% of actual churners — this is relatively good recall, indicating it can detect churners but at the cost of precision.

**F1 Score**:
- For non-churners: 0.82 — solid performance with good balance between precision and recall.
- For churners: 0.43 — weaker score, indicating the model struggles to precisely identify churners.

**Accuracy** = 73%: The model correctly predicted 73% of total cases. However, since the data is imbalanced (more non-churners), accuracy alone is not enough to judge performance.

**ROC-AUC** = 0.797: This is quite good — the model has a strong ability to distinguish between churners and non-churners, though it’s not perfect.

## **LOGISTIC REGRESSION Model Performance Note and DECISION**

The model performs well in identifying non-churners with high Accuracy (73%) and good ROC-AUC 0.80,indicating strong separation capability between churners and non-churners.

However, it struggles with precision for churners (31%), meaning it often wrongly predicts churn. Further tuning or trying other models

- I will therefore tyr another tree based model *Decision tree* and compare the findings

## 5.2 **(DECISION TREE MODEL)**

- This model will split data into smaller subsets while at the same time developing an associated decision tree incrementally
### 5.2.1 Predictions/Evaluations (Decision Tree Model)
![](./images/DECISION_TREE_performance.png)

### 5.2.2 Confusion Mattrix (Decision Tree Model)
![](./images/DECISION_TREE_confusion_matrix.png)

### 5.2.3 ROC Curve -  (Decision Tree Model vs Logistic Regression)
![](./images/TREE_VS_logistic_reg_ROC_CURVE.png)

## **Comaparison and Observations from Decision Tree Predictions (Compared to Logistic Regression)**

#### **1. Precision**:
Out of all predicted non-churners (**Class 0**), **96%** were actually correct — slightly better than logistic regression (**93%**).
For churners (**Class 1**), **71%** were correctly identified as actual churners — **huge improvement** over logistic regression (**31%**), showing **far fewer false positives**.
**Improvement**:  
The decision tree is **more precise** in identifying true churners — meaning **fewer good customers are wrongly flagged**.
#### **2. Recall**:
The model correctly identified **95%** of actual non-churners — much higher than logistic’s **73%**.
It also identified **74%** of actual churners — a **slight improvement** over logistic’s **70%**.
**Improvement**:  
While recall for churners only improved slightly, the tree managed to maintain **high recall while also dramatically improving precision**.
#### **3. F1 Score**:
For non-churners: **0.95** — significantly better than logistic’s **0.82**, showing better overall classification.
For churners: **0.72** — much stronger than logistic’s **0.43**, reflecting a better balance between precision and recall for the minority class.
**Improvement**:  
The decision tree provides **more reliable identification of churners**, which is **crucial for targeted business action**.
#### **4. Accuracy** = **92%**
The model correctly predicted **92% of all cases** — a major increase from logistic’s **73%**.
**Improvement**:
This shows the decision tree handles both classes well, **despite the class imbalance**.
#### **5. ROC-AUC** = **0.856**
Indicates a **strong ability to separate** churners from non-churners.
Better than logistic regression's **0.797**, reflecting **better probability calibration and class separation**.
**Improvement**:  
A higher AUC means the decision tree is **more confident in its predictions**, which is **useful for ranking customers by churn risk**.
#### **6. ROC Curve** 
- The closer the ROC curve is to the top-left corner, the better the model's, The **Decision Tree Model** curve is above the Logistic Regression curve.
- **Decision Tree Model** shows a better trade-off between True Positive Rate (Sensitivity/Recall) and False Positive Rate, making it the superior model for churn prediction.
- **Logistic Regression**, is not that bad off but has a lower ability to correctly classify churn cases compared to Random Forest.

# 6.0 **Conclusion and Recommendations**

## **Conclusion**

- Two classification models were evaluated: Logistic Regression and a tuned Decision Tree classifier to try answer to our business problem and objectives. 
- While the logistic regression gave a decent baseline, the Decison tree significantly outperformed it across all key metrics, including F1 score, precision, recall, and ROC-AUC as shown above. 
- The results demonstrate the Decision Tree capability to correctly identify churn customers while maintaining high reliability for loyal customers. 

## **Recommendations**
Based on the performance of the models and insights derived, the following business actions are recommended:

**1. Deploy the Decision Tree Model in Production**
- use the model as the core engine to score and flag customers likely to churn

**2. Prioritize High-Risk Customers for Retention Campaigns**
- Focus campaign and retention resources to customers identify as high risk churners.

**3.  Integrate Churn Scores into CRM Tools**
- Embed predictions into dashboards to ease decison making in various segements as well as real monitoring and faster decisions. 

**4.  Monitor Heavy Daytime Users and International Plan Holders**
- Based on Feature importance, Customers with high total day minutes or on international plans are most at risk to leave, Offer custom bundles or cheaper rates to retain these high-usage or globally connected users.

**4. Prioritize Customers with High Call Volume to Support**
- Many customer service calls correlate with churn — flag these customers for priority support or follow-up calls.

## 5. LINKS AND RESOURCES

- **Dataset Source** :  [kaggle-churn-in-telecoms-dataset](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)



