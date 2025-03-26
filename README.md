# Credit Card Fraud Detection System

A machine learning project to detect fraudulent credit card transactions using both Random Forest and Neural Network approaches. This project tackles the challenge of imbalanced datasets in fraud detection, where fraudulent transactions are rare but costly events.

## Project Overview

Credit card fraud is a significant concern for financial institutions, with an estimated 0.1% of transactions being fraudulent. This project implements and compares two powerful machine learning approaches to identify these fraudulent transactions effectively:

1. **Random Forest Classifier** with SMOTE balancing
2. **Neural Network** with class weights and regularization techniques

The system effectively identifies potential fraud while minimizing false positives, which is critical in real-world fraud detection systems.

## Dataset

The project uses an anonymized dataset of credit card transactions, each labeled as fraudulent or legitimate. Key characteristics:
- Over 280,000 transactions
- Only 0.17% are fraudulent transactions (highly imbalanced)
- 30 features (all anonymized for privacy)
- Contains 'Time', 'Amount', and 'Class' (target) fields

## Key Components

### Data Preprocessing
- Feature scaling with StandardScaler
- Specialized handling for 'Time' and 'Amount' features
- Stratified train-test split to maintain class distribution

### Model Implementation

#### Random Forest Approach
- Applied SMOTE to balance training classes
- Optimized hyperparameters: n_estimators=100, max_depth=10
- Parallel processing for improved performance
- Feature importance analysis for interpretability

#### Neural Network Approach
- Multi-layer architecture with dropout and batch normalization
- Class weighting to handle imbalanced data
- Early stopping to prevent overfitting
- Optimized with Adam optimizer

### Evaluation Framework
- Classification reports with precision, recall, and F1-score
- ROC curves and AUC scores
- Precision-Recall curves (critical for imbalanced datasets)
- Confusion matrices
- Feature importance visualization

## Results

Both models achieve high performance:

**Random Forest Model:**
- 99.8% overall accuracy
- 85% recall on fraudulent transactions
- 43% precision on fraudulent transactions
- ROC-AUC: 0.9805

**Neural Network Model:**
- 99.9% overall accuracy
- 90% recall on fraudulent transactions  
- 12% precision on fraudulent transactions
- ROC-AUC: 0.9758

The tradeoff between precision and recall shows that:
- Random Forest provides better balance between precision and recall
- Neural Network maximizes recall but produces more false positives

## Technical Implementation

The project is implemented in Python using:

- **Data Handling**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (RandomForestClassifier, train_test_split, metrics)
- **Deep Learning**: TensorFlow/Keras (Sequential, Dense, BatchNormalization, Dropout)
- **Imbalanced Learning**: SMOTE from imbalanced-learn
- **Visualization**: Matplotlib, Seaborn

## Key Files

- `credit_card_fraud_detection.ipynb`: Main Jupyter notebook with all code and analysis
- `requirements.txt`: List of dependencies
- `model/`: Directory containing saved models (not included in repository)

## Installation and Usage

1. Clone the repository:
```
git clone https://github.com/PeterAntoun/credit-card-fraud-detection.git
cd credit-card-fraud-detection
```

2. Install dependencies:
```
pip install -r requirements.txt
```

3. Download the dataset from Kaggle (not included due to size) and place in the project directory
   - Dataset available at: https://www.kaggle.com/mlg-ulb/creditcardfraud

4. Run the Jupyter notebook:
```
jupyter notebook credit_card_fraud_detection.ipynb
```

## Future Improvements

- Implement anomaly detection approaches (Isolation Forest, Autoencoders)
- Explore additional feature engineering techniques
- Investigate cost-sensitive learning approaches
- Deploy model as a real-time API service
- Add streaming data processing capability
- Fine-tune the neural network architecture
- Explore ensemble methods combining both approaches

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- The dataset is from the Kaggle Credit Card Fraud Detection competition
- This project was developed as part of a machine learning course assignment
- Special thanks to the instructors and peers for their valuable feedback
