from pathlib import Path
import warnings
import joblib
import pandas as pd
from flask import Flask, render_template, request
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
BASE = Path(__file__).resolve().parent
DATA_PATH = BASE / "PS02_Retail_Customer_Purchase_Prediction(1).csv"
MODEL_PATH = BASE / "ps02_high_sales_order_model.joblib"
HISTORY_PATH = BASE / "prediction_history.csv"
app = Flask(__name__)

FIELDS = [
    {"name":"Region","label":"Region","options":["North","South","East","West","Central"]},
    {"name":"City","label":"City","default":"Delhi","type":"text"},
    {"name":"Category","label":"Category","options":["Books","Clothing","Electronics","Furniture","Groceries"]},
    {"name":"Product Name","label":"Product name","default":"Sample Product","type":"text"},
    {"name":"Quantity","label":"Quantity","default":"1","type":"number","min":"1","step":"1"},
    {"name":"Unit Price","label":"Unit price","default":"1000","type":"number","min":"0","step":"0.01"},
    {"name":"Discount","label":"Discount","default":"0","type":"number","min":"0","step":"0.1"},
    {"name":"Payment Mode","label":"Payment mode","options":["Credit Card","Debit Card","Cash","UPI","Net Banking"]},
]
TARGET = "High_Sales_Order"
DROP = [TARGET, "Sales", "Profit", "Purchase", "Order ID", "Customer Name", "Order Date"]

def train_model():
    if not DATA_PATH.exists():
        # Accept the common alternative dataset filename too.
        alternatives = list(BASE.glob("*.csv"))
        candidates = [p for p in alternatives if "Retail_Customer_Purchase_Prediction" in p.name or "Ecommerce_Sales_Data" in p.name]
        if not candidates:
            raise FileNotFoundError("Dataset CSV not found. Put the supplied ecommerce CSV beside main.py.")
        data_path = candidates[0]
    else:
        data_path = DATA_PATH
    df = pd.read_csv(data_path)
    if "Sales" not in df.columns:
        raise ValueError("The dataset must contain a Sales column.")
    median_sales = float(df["Sales"].median())
    df[TARGET] = (df["Sales"] >= median_sales).astype(int)
    X = df.drop(columns=[c for c in DROP if c in df.columns], errors="ignore")
    y = df[TARGET]
    categorical = X.select_dtypes(include=["object","category"]).columns.tolist()
    numeric = X.select_dtypes(exclude=["object","category"]).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler())]), numeric),
        ("cat", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("onehot",OneHotEncoder(handle_unknown="ignore"))]), categorical)
    ])
    Xtr, Xte, ytr, yte = train_test_split(X,y,test_size=.2,random_state=42,stratify=y)
    candidates = []
    for estimator in [
        LogisticRegression(max_iter=2000,class_weight="balanced"),
        RandomForestClassifier(n_estimators=300,min_samples_leaf=2,random_state=42,class_weight="balanced",n_jobs=-1)
    ]:
        pipe = Pipeline([("preprocess",prep),("model",estimator)])
        pipe.fit(Xtr,ytr)
        candidates.append((accuracy_score(yte,pipe.predict(Xte)),pipe))
    score, model = max(candidates,key=lambda item:item[0])
    joblib.dump(model,MODEL_PATH)
    return model, score

model, holdout_accuracy = train_model()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/contact", methods=["GET","POST"])
def contact():
    return render_template("contact.html")

@app.route("/history")
def history():
    rows = []
    if HISTORY_PATH.exists():
        hist = pd.read_csv(HISTORY_PATH).fillna("")
        rows = hist.to_dict(orient="records")
    return render_template("history.html", historical_data=rows)

@app.route("/project", methods=["GET","POST"])
def project_page():
    result = probability = None
    if request.method == "POST":
        payload = {}
        for field in FIELDS:
            name = field["name"]
            raw = request.form.get(name, field.get("default",""))
            if name in ("Quantity","Unit Price","Discount"):
                payload[name] = float(raw or 0)
            else:
                payload[name] = raw
        Xnew = pd.DataFrame([payload])
        # Fill any expected feature not present in the form with a neutral value.
        expected = list(model.feature_names_in_)
        for col in expected:
            if col not in Xnew.columns:
                Xnew[col] = 0
        Xnew = Xnew[expected]
        pred = int(model.predict(Xnew)[0])
        probability = round(float(model.predict_proba(Xnew)[0][1]) * 100, 2)
        result = "High-sales order (proxy class)" if pred else "Lower-sales order (proxy class)"
        record = {**payload, "prediction":result, "high_sales_probability_percent":probability}
        pd.DataFrame([record]).to_csv(HISTORY_PATH, mode="a", header=not HISTORY_PATH.exists(), index=False)
    return render_template("project.html", fields=FIELDS, result=result, probability=probability,
                           model_name=type(model.named_steps["model"]).__name__,
                           accuracy=round(holdout_accuracy*100,2))

if __name__ == "__main__":
    app.run(debug=True)
