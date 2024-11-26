import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score


gc_air = pd.read_csv('air_sequences_with_features_updated.csv')
gc_soil = pd.read_csv('soil_sequences_with_features_updated.csv')
gc_human = pd.read_csv('human_sequences_with_features_updated.csv')
gc_dog = pd.read_csv('dog_sequences_with_features_updated.csv')

all_dfs = pd.concat([gc_air, gc_soil, gc_human, gc_dog], ignore_index = True)
# Example DataFrame (replace this with your actual dataset)
data = all_dfs  # Load your dataset

# Encode the target variable (Type)
label_encoder = LabelEncoder()
data['Type'] = label_encoder.fit_transform(data['Type'])

# Split features and target
X = data.drop(['Type', 'Sequence', 'k-mer (k=3)'], axis=1)  # Drop complex or unused columns
y = data['Type']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=1000,       # Number of boosting iterations
    learning_rate=0.1,     # Learning rate
    depth=6,               # Depth of each tree
    loss_function='MultiClass',  # Loss function for multi-class classification
    random_seed=42,
    verbose=100            # Show training progress every 100 iterations
)

# Train the model
cat_model.fit(X_train, y_train)

# Make predictions
y_pred = cat_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
