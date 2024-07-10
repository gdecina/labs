from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

exec(open("data_import.py").read())

# Split data into features and labels
X = adults.drop("value", axis = 1).values
y = adults["value"].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Model training
lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
lda.score(X_test, y_test)

results = pd.DataFrame({"variable": adults.drop("value", axis = 1).columns, 
                        "coefficient": lda.coef_.flatten().round(3)})