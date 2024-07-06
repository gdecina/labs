import pandas as pd
from sklearn.preprocessing import StandardScaler

colnames = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
            "hours-per-week", "native-country", "value"]
adults = pd.read_csv("adult.data", sep = ",", names = colnames, index_col = False)

if adults.isna().values.any():
    adults.dropna(inplace = True)

# Processing the dataset
adults.replace(to_replace = {"value": {"<=50K": 0, ">50K": 1}}, regex = True, inplace = True)
adults.drop(["fnlwgt", "education"], axis = 1, inplace = True)
adults["native-country"] = adults["native-country"].apply(lambda x: 0 if x == "United-States" else 1)
adults = pd.get_dummies(adults, columns = ["workclass", "occupation", "race", "sex", "relationship", "marital-status"])
adults = adults * 1 # Converts booleans to binary values

# Scale continuous variables
scaler = StandardScaler()
adults[["age", "education-num"]] = scaler.fit_transform(adults[["age", "education-num"]])
# adults.convert_dtypes(float, convert_boolean = float)