# Telecom X – Parte 2: Prevendo Churn
# Pipeline: Preparação -> Correlação -> Seleção -> Modelos -> Avaliação -> Interpretação
import json, requests
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# =========================
# 1) EXTRAÇÃO E TRATAMENTO
# =========================
URL = "https://raw.githubusercontent.com/ingridcristh/challenge2-data-science/main/TelecomX_Data.json"
data = requests.get(URL, timeout=15).json()
df = pd.DataFrame(data)

# Flatten
df_full = pd.concat([
    df.drop(['customer', 'phone', 'internet', 'account'], axis=1),
    df['customer'].apply(pd.Series),
    df['phone'].apply(pd.Series),
    df['internet'].apply(pd.Series),
    df['account'].apply(pd.Series)
], axis=1)

# Normalização de campos de cobrança
charges = df_full['Charges'].apply(pd.Series)
df_full['Charges_Monthly'] = pd.to_numeric(charges.get('Monthly'), errors='coerce')
df_full['Charges_Total']   = pd.to_numeric(charges.get('Total'),   errors='coerce')
df_full.drop(columns=['Charges'], inplace=True, errors='ignore')

# Ajustes básicos
if 'SeniorCitizen' in df_full.columns:
    df_full['SeniorCitizen'] = pd.to_numeric(df_full['SeniorCitizen'], errors='coerce').fillna(0).astype(int)

# Target
df_full['churn_flag'] = df_full['Churn'].map({'Yes':1, 'No':0})
if df_full['churn_flag'].isna().any():
    df_full['churn_flag'] = df_full['Churn'].astype(str).str.strip().str.lower().map(
        {'yes':1,'y':1,'1':1,'true':1,'no':0,'n':0,'0':0,'false':0}
    ).astype('Int64')

# Remover linhas sem Total (coerentes com dataset telco)
df_full = df_full.dropna(subset=['Charges_Total'])
df_full = df_full.dropna(subset=['churn_flag'])

# X, y
y = df_full['churn_flag'].astype(int)
drop_cols = ['churn_flag','Churn']
X = df_full.drop(columns=[c for c in drop_cols if c in df_full.columns])

# Tipagem
numeric_cols = X.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object','category','bool']).columns.tolist()
for c in categorical_cols:
    X[c] = X[c].astype('category')

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# =========================
# 2) PRÉ-PROCESSAMENTO
# =========================
num_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
cat_pipe = Pipeline(steps=[
    ("imp", SimpleImputer(strategy="most_frequent")),
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ],
    remainder="drop",
    verbose_feature_names_out=False
)

# Ajuste do preprocessor para pegar nomes das features transformadas
preprocessor.fit(X_train)
feat_names_all = preprocessor.get_feature_names_out()

# =======================================
# 3) ANÁLISE DE CORRELAÇÃO (numéricas)
# =======================================
corr_report = []
if numeric_cols:
    num_train = pd.DataFrame(preprocessor.named_transformers_['num'][:-1].transform(
        X_train[numeric_cols]
    ) if hasattr(preprocessor.named_transformers_['num'], "steps") else
        preprocessor.named_transformers_['num'].transform(X_train[numeric_cols])
    )
    # Recalcular com escala original não é essencial para ranqueamento simples:
    num_corr = pd.concat([X_train[numeric_cols].reset_index(drop=True), y_train.reset_index(drop=True)], axis=1)
    c = num_corr.corr(numeric_only=True)['churn_flag'].drop('churn_flag').sort_values(key=np.abs, ascending=False)
    corr_report = c.head(10)

print("\n[Correlação com churn (numéricas) - Top 10 | magnitude]")
if len(corr_report):
    print(corr_report.to_string())
else:
    print("Sem variáveis numéricas suficientes para correlação.")

# =======================================
# 4) SELEÇÃO DE VARIÁVEIS (mutual_info)
# =======================================
# Transformar X_train -> matriz densa pós-encoding
Xtr_all = preprocessor.transform(X_train)
Xte_all = preprocessor.transform(X_test)

n_features = Xtr_all.shape[1]
k = min(25, n_features)  # top-K simples
selector = SelectKBest(score_func=mutual_info_classif, k=k)
selector.fit(Xtr_all, y_train)

support = selector.get_support()
feat_names_sel = np.array(feat_names_all)[support]
mi_scores = pd.Series(selector.scores_[support], index=feat_names_sel).sort_values(ascending=False)

print("\n[Seleção por Mutual Information - Top 15]")
print(mi_scores.head(15).to_string())

# Redução de dimensionalidade
Xtr_sel = selector.transform(Xtr_all)
Xte_sel = selector.transform(Xte_all)

# =========================
# 5) MODELAGEM (2 modelos)
# =========================
models = {
    "log_reg": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE),
    "rf": RandomForestClassifier(
        n_estimators=400, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
}

def eval_model(name, clf, Xtr, Xte):
    clf.fit(Xtr, y_train)
    proba = clf.predict_proba(Xte)[:,1]
    pred  = (proba >= 0.5).astype(int)
    metrics = {
        "model": name,
        "roc_auc": roc_auc_score(y_test, proba),
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred)
    }
    return clf, metrics, pred, proba

results = []
trained = {}

for name, clf in models.items():
    clf_tr, m, y_pred, y_proba = eval_model(name, clf, Xtr_sel, Xte_sel)
    trained[name] = {"clf": clf_tr, "pred": y_pred, "proba": y_proba}
    results.append(m)

res_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
print("\n[Métricas - threshold 0.5]")
print(res_df.to_string(index=False))

best_name = res_df.iloc[0]['model']
best_clf = trained[best_name]['clf']
y_pred_best = trained[best_name]['pred']
y_proba_best = trained[best_name]['proba']

print(f"\nMelhor modelo: {best_name}")
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred_best))
print("\nClassification report:")
print(classification_report(y_test, y_pred_best, zero_division=0))

# ==================================
# 6) INTERPRETAÇÃO / IMPORTÂNCIAS
# ==================================
# Importância por permutação (modelo vencedor)
perm = permutation_importance(
    best_clf, Xte_sel, y_test, n_repeats=10, random_state=RANDOM_STATE, scoring="roc_auc", n_jobs=-1
)
perm_imp = pd.DataFrame({
    "feature": feat_names_sel,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values("importance_mean", ascending=False)

print("\n[Permutation Importance - Top 15]")
print(perm_imp.head(15).to_string(index=False))

# Importância nativa (se houver)
if hasattr(best_clf, "feature_importances_"):
    native_imp = pd.DataFrame({
        "feature": feat_names_sel,
        "importance": best_clf.feature_importances_
    }).sort_values("importance", ascending=False)
    print("\n[Importância nativa (árvore) - Top 15]")
    print(native_imp.head(15).to_string(index=False))
elif hasattr(best_clf, "coef_"):
    coef = best_clf.coef_.ravel()
    native_imp = pd.DataFrame({
        "feature": feat_names_sel,
        "coef": coef,
        "abs_coef": np.abs(coef)
    }).sort_values("abs_coef", ascending=False)
    print("\n[Coeficientes (LogReg) - Top 15 | |coef|]")
    print(native_imp.head(15).to_string(index=False))

# =========================
# 7) CONCLUSÃO ESTRATÉGICA
# =========================
top_vars = perm_imp.head(8)['feature'].tolist()
print("\n=== Conclusão Estratégica (resumo) ===")
print("* Desempenho do melhor modelo:", best_name,
      f"| ROC AUC={roc_auc_score(y_test, y_proba_best):.3f}",
      f"| F1={f1_score(y_test, y_pred_best):.3f}",
      f"| Recall={recall_score(y_test, y_pred_best):.3f}",
      f"| Precision={precision_score(y_test, y_pred_best, zero_division=0):.3f}")

print("* Principais fatores associados ao churn (top features por permutation importance):")
for f in top_vars:
    print("  -", f)

print("* Direções estratégicas sugeridas (com base nessas variáveis):")
print("  1) Atuar em clientes com menor tenure (onboarding/benefícios iniciais).")
print("  2) Incentivar contratos de maior duração quando Month-to-month estiver entre as top features.")
print("  3) Migrar métodos de pagamento associados a maior churn para alternativas mais estáveis.")
print("  4) Revisar pacotes com alta cobrança mensal (ajuste/bundles) quando charges aparecerem entre as top features.")
