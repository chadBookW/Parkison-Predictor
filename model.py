import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import pickle

# Load and preprocess data
def load_and_preprocess_data(url=None):
    # Load dataset
    if url:
        df = pd.read_csv(url)
    else:
        df = pd.read_csv('parkinsons.data')  # If running locally, replace with local path
    
    # Drop the 'name' column and split data into features (X) and target (y)
    X = df.drop(columns=['name', 'status'])  # 'status' is the target
    y = df['status']  # Target variable
    
    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# Train the model
def train_model():
    # Load and preprocess data
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data'
    X, y, scaler = load_and_preprocess_data(url)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Decision Tree model
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    # Test model accuracy
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Save the model and the scaler for later use
    with open('random_forest_model.pkl', 'wb') as model_file:
        pickle.dump(clf, model_file)
    
    with open('scaler.pkl', 'wb') as scaler_file:
        pickle.dump(scaler, scaler_file)
 
# Function to make predictions with the trained model
def make_prediction(input_data):
    # Load the trained model and scaler
    with open('random_forest_model.pkl', 'rb') as model_file:
        clf = pickle.load(model_file)
    
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    
    # Scale the input data
    input_scaled = scaler.transform(input_data)
    
    # Make predictions
    prediction = clf.predict(input_scaled)
    prediction_proba = clf.predict_proba(input_scaled)
    
    return prediction, prediction_proba

if __name__ == "__main__":
    train_model() 