import pandas as pd

# Replace 'path_to_dataset/train.csv' with the actual file path
data = pd.read_csv(r'C:\Users\LENOVO\.spyder-py3\prodigy projects\train.csv')
print(data.head())  # Displays the first 5 rows to understand the data


print(data.info())        # Overview of data types and null values
print(data.describe())    # Summary statistics (mean, max, min, etc.)
print(data.columns)



features = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]  # Independent variables
target = data['SalePrice']  # Dependent variable (price)


print(data.isnull().sum())  # Shows how many missing values in each column

# Remove rows with missing values (if any)
data = data.dropna()



from sklearn.model_selection import train_test_split

# Split the data: 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)


from sklearn.linear_model import LinearRegression

model = LinearRegression()        # Create a linear regression model
model.fit(X_train, y_train)       # Train the model


predictions = model.predict(X_test)
print("Sample Predictions:", predictions[:5])  # Show first 5 predictions


from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Absolute Error: {mae}")
print(f"R-squared Score: {r2}")



import matplotlib.pyplot as plt

plt.scatter(y_test, predictions, alpha=0.5, color='blue')
plt.title("Actual vs Predicted Prices")
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.grid(True)
plt.show()


coefficients = model.coef_
features_list = features.columns

for feature, coef in zip(features_list, coefficients):
    print(f"{feature}: {coef:.2f}")
