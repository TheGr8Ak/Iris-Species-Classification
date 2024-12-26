import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import pickle

# Load and preprocess data
data = pd.read_csv('C:\Projects\OpenCV\iris.csv')
data = data.drop(columns=['Id'])

# Display basic dataset information
print(data.head())
print(data.shape)
print(data.info())
print(data.describe())

# Plot data relationships
sns.pairplot(data, hue='Species')
plt.show()

# Encode species labels
label_encoder = LabelEncoder().fit(data['Species'])
data['Species'] = label_encoder.transform(data['Species'])

# Split data into features and target
X = data.drop(columns='Species')
Y = data['Species']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(x_train, y_train)

# Evaluate the model
print("Model accuracy:", model.score(x_test, y_test))

y_pred = model.predict(x_test)
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Hyperparameter tuning with GridSearchCV
param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(x_train, y_train)
print("Best Parameters:", grid_search.best_params_)

# Save the model to a file
with open('iris_model.pkl', 'wb') as file:
    pickle.dump(model, file)
print("\nModel saved as 'iris_model.pkl'.")

# Load the model from the file
with open('iris_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
print("\nModel loaded successfully.")