import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import  accuracy_score
import joblib

def poison_data(X, poison_level):
    """
    Poison a given percentage of samples in X with random noise.
    """
    X_poisoned = X.copy()
    n_samples = X.shape[0]
    n_poison = int(poison_level * n_samples)

    # Random indices to poison
    poison_indices = np.random.choice(n_samples, n_poison, replace=False)

    # Inject noise
    for i in poison_indices:
        X_poisoned[i] = np.random.uniform(
            low=np.min(X, axis=0),
            high=np.max(X, axis=0),
            size=X.shape[1]
        )

    return X_poisoned

data = pd.read_csv('/home/g24ait093/week4/data/iris.csv')
data.head(5)

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)

results = []

for level in [0.00, 0.05, 0.10, 0.50]:
    X_train_poisoned = poison_data(X_train, poison_percentage)

    X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
    y_train = train.species
    X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test.species

    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(X_train,y_train)
    prediction=mod_dt.predict(X_test)
    acc= accuracy_score(prediction,y_test)

    print(f"Poison Level: {int(poison_percentage * 100)}% - Validation Accuracy: {acc:.4f}")
        # Save results

    results.append({
        "Poison_Percentage": int(poison_percentage * 100),
        "Validation_Accuracy": round(acc, 4)
    })

    joblib.dump(mod_dt, "model.weights.h5"+str(level))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("iris_poisoning_results.csv", index=False)

