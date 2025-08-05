import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas.plotting import parallel_coordinates
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import  accuracy_score
import joblib
import matplotlib.pyplot as plt

def poison_data(X, poison_level):
    """
    Poison a given percentage of samples in X with random noise.
    """
    
    X_poisoned = X.copy()
    n_samples = X.shape[0]
    n_poison = int(0.05 * n_samples)

    # Random indices to poison
    poison_indices = np.random.choice(X_poisoned.index, n_poison, replace=False)

    #print(X_poisoned)
    # Inject noise
    for i in poison_indices:
        #print(i,X_poisoned.loc[i])
        X_poisoned.loc[i] = np.random.uniform(
            low=np.min(X, axis=0),
            high=np.max(X, axis=0),
            size=X.shape[1]
        )

    return X_poisoned

data = pd.read_csv('week4/iris.csv')
data.head(5)

train, test = train_test_split(data, test_size = 0.4, stratify = data['species'], random_state = 42)

results = []

for level in np.linspace(0, 1, 21):
    level = round(level,2)
    X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
    X_train_poisoned = poison_data(X_train, level)
    y_train = train.species
    
    X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
    y_test = test.species

    mod_dt = DecisionTreeClassifier(max_depth = 3, random_state = 1)
    mod_dt.fit(X_train_poisoned,y_train)
    prediction=mod_dt.predict(X_test)
    acc= accuracy_score(prediction,y_test)

    print(f"Poison Level: {int(level * 100)}% - Validation Accuracy: {acc:.4f}")
        # Save results

    results.append({
        "Poison_Percentage": int(level * 100),
        "Validation_Accuracy": round(acc, 4)
    })

    joblib.dump(mod_dt, "model.weights.h5"+str(level))

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to CSV
results_df.to_csv("iris_poisoning_results.csv", index=False)



# Line chart: Poisoning level vs accuracy
plt.figure(figsize=(6, 4))
plt.plot(
    results_df["Poison_Percentage"],
    results_df["Validation_Accuracy"],
    marker='o', linestyle='-', linewidth=2
)

plt.xlabel("Poison Level (%)")
plt.ylabel("Validation Accuracy")
plt.title("Impact of Data Poisoning on Accuracy (Line Chart)")
plt.ylim(0.7, 1.0)  # <-- Set Y-axis from 0.7 to 1.0
plt.grid(True)

# Save the plot
plt.savefig("poisoning_plot.png")

