# 🧠 Stacking Classifier on Diabetes Dataset

This project implements a basic **stacking ensemble model** to classify diabetes outcomes using the **PIMA Indian Diabetes dataset**. Two base models (K-Nearest Neighbors and Support Vector Machine) are used, followed by a **meta-classifier** (Random Forest) that learns from their combined predictions. Hyperparameter tuning is also applied using GridSearchCV.

## 📁 Dataset
- **Source**: `diabetes.csv`
- **Features**: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, `Age`
- **Target**: `Outcome` (0 = No Diabetes, 1 = Diabetes)

## 🔧 Tools & Libraries
- Python
- pandas, numpy
- matplotlib, seaborn (for visualization)
- scikit-learn (modeling and evaluation)

## 🧪 Models Used
1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Classifier (SVC)**
3. **Random Forest Classifier** (used as a meta-model in stacking)

## 🧬 Workflow
1. Load and preprocess the dataset.
2. Split into training, validation, and test sets.
3. Train KNN and SVC on the training set.
4. Predict on the validation set and use these predictions as features for stacking.
5. Train a Random Forest Classifier on the stacked predictions.
6. Evaluate on test set.
7. Use GridSearchCV to tune hyperparameters of the Random Forest Classifier.

## 📊 Results
| Model           | Accuracy |
|----------------|----------|
| KNN            | 66.23%   |
| SVC            | 76.62%   |
| Stacked Model  | 76.62%   |
| Stacked + Tuning | 76.62% |

## ⚙️ Best Hyperparameters (GridSearchCV)
```json
{
  "criterion": "gini",
  "max_features": "log2",
  "min_samples_leaf": 1,
  "min_samples_split": 4,
  "n_estimators": 90
}
```
## 📌 Conclusion
Stacking with KNN and SVC improves the robustness of the model and yields better or comparable accuracy. Grid search allows fine-tuning of the final ensemble model for optimal performance.

## 📝 License
This project is licensed under the MIT License.

