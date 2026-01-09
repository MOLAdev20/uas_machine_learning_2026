# ================ UAS Machine Learning ================
# Nama : M Sabiilul Hikam Azzuhrie
# NIM  : 231011400198
# Kelas : 05TPLE004
# ======================================================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report

# Load data
df = pd.read_csv("titanic_seaborn.csv")

# Fitur & target
X = df[["pclass","sex","age","sibsp","parch","fare","embarked"]]
y = df["survived"]

# Preprocess: missing value + encoding
num = ["pclass","age","sibsp","parch","fare"]
cat = ["sex","embarked"]
pre = ColumnTransformer([
    ("num", SimpleImputer(strategy="median"), num),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat)
])

# Model
model = Pipeline([("pre", pre),
                  ("dt", DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42))])

# Split + train + evaluate
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

# Visualisasi tree (dibatasi biar kebaca)
dt = model.named_steps["dt"]
ohe = model.named_steps["pre"].named_transformers_["cat"].named_steps["oh"]
feat_names = num + list(ohe.get_feature_names_out(cat))

plt.figure(figsize=(16, 7))
plot_tree(dt, feature_names=feat_names, class_names=["0","1"], filled=True, max_depth=3)
plt.title("Decision Tree Titanic (depth<=3)")
plt.show()
