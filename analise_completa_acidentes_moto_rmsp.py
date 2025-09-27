import os
import joblib
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
# Usar backend não interativo para evitar erros de Tkinter em threads/processos
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.inspection import permutation_importance
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils import Bunch
from sklearn.preprocessing import StandardScaler

from imblearn.over_sampling import SMOTENC
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.model_selection import GridSearchCV

RND = 42
plt.style.use("seaborn-v0_8")
plt.rcParams["figure.figsize"] = (10, 6)

# Execução em modo rápido por padrão (pode desativar com FAST=0 no ambiente)
FAST = os.environ.get("FAST", "1") == "1"

# ----------------------------
# 0) Helpers
# ----------------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)
ensure_dir("outputs")
ensure_dir(os.path.join("outputs", "cache"))

def save_fig(fig, name, dpi=300):
    path = os.path.join("outputs", name)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

# ----------------------------
# 1) Carregamento e filtro
# ----------------------------
print("1) Carregando dados...")
frames = []
for fname in ["pessoas_2015-2021.csv", "pessoas_2022-2025.csv"]:
    if os.path.exists(fname):
        try:
            df = pd.read_csv(fname, encoding="latin-1", sep=";", low_memory=False)
            frames.append(df)
            print(f"  - {fname}: {len(df):,} linhas")
        except Exception as e:
            print(f"  - Erro lendo {fname}: {e}")

if not frames:
    raise FileNotFoundError("Nenhum dos arquivos esperados foi encontrado no diretório.")

df_raw = pd.concat(frames, ignore_index=True)
print(f"Total concatenado: {len(df_raw):,} registros")

# Filtrar para motocicletas e RMSP
mask_moto = df_raw["tipo_veiculo_vitima"].str.upper().str.contains("MOTOCICLETA", na=False)
mask_rms = df_raw["regiao_administrativa"].str.upper().str.contains("METROPOLITANA|GRANDE SÃO PAULO|METROPOLITANA DE SÃO PAULO", na=False)
df = df_raw[mask_moto & mask_rms].copy()
print(f"Após filtro (motocicleta + RMSP): {len(df):,} registros")

# Remover registros com target inválido
df = df[df["gravidade_lesao"].notna()]
df = df[~df["gravidade_lesao"].str.upper().str.contains("NAO|NÃO|DESCONHECIDO", na=False)]
print(f"Após remover 'não disponível' em gravidade: {len(df):,} registros")

# Normalizar strings uppercase for consistency
for col in ["gravidade_lesao", "sexo", "tipo_via", "tipo_de_vitima", "faixa_etaria_demografica", "faixa_etaria_legal", "municipio"]:
    if col in df.columns:
        df[col] = df[col].astype(str).str.upper().str.strip()

# Converter idade para numérica (se houver problemas)
df["idade"] = pd.to_numeric(df["idade"], errors="coerce")

# ----------------------------
# 2) EDA mínima e gráficos (salvos)
# ----------------------------
print("2) Gerando EDA básica (PNG)...")

# Histograma idade
fig = plt.figure()
sns.histplot(df["idade"].dropna(), bins=30, kde=True)
plt.title("Distribuição da Idade (motociclistas - RMSP)")
plt.xlabel("Idade")
save_fig(fig, "histograma_idade.png")

# Boxplot idade x gravidade
fig = plt.figure()
sns.boxplot(x="gravidade_lesao", y="idade", data=df)
plt.title("Idade por Gravidade")
save_fig(fig, "boxplot_idade_gravidade.png")

# Correlação idade vs gravidade (encode gravidade as ordinal for visualization only)
grav_map = {g: i for i, g in enumerate(sorted(df["gravidade_lesao"].unique()))}
df["gravidade_num_viz"] = df["gravidade_lesao"].map(grav_map)
fig = plt.figure()
sns.regplot(x="idade", y="gravidade_num_viz", data=df, scatter_kws={"alpha":0.15})
plt.title("Idade vs Gravidade (codificada)")
save_fig(fig, "correlacao_idade_gravidade.png")

# Show class distribution
class_counts = df["gravidade_lesao"].value_counts()
fig = plt.figure()
sns.barplot(x=class_counts.index, y=class_counts.values)
plt.title("Distribuição das classes (gravidade)")
plt.ylabel("Contagem")
save_fig(fig, "distribuicao_gravidade.png")

# ----------------------------
# 3) Seleção de features e split
# ----------------------------
print("3) Preparando features e split treino/teste...")

# Escolha de features que estariam disponíveis no momento do acidente
candidate_features = [
    "sexo", "idade", "faixa_etaria_demografica", "faixa_etaria_legal",
    "tipo_via", "tipo_de_vitima", "municipio", "mes_sinistro", "dia_sinistro", "ano_sinistro"
]
# manter apenas colunas existentes
features = [c for c in candidate_features if c in df.columns]
print("Features consideradas:", features)

X = df[features].copy()
y = df["gravidade_lesao"].copy()

# Remover linhas com target NA já feito; para features, iremos imputar no pipeline
# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RND, stratify=y
)
print(f"Split feito: treino {len(X_train):,}, teste {len(X_test):,}")

# ----------------------------
# 4) Preparar ColumnTransformer antes SMOTENC
#     - Numeric: SimpleImputer (median)
#     - Categorical: OrdinalEncoder (gera inteiros para SMOTENC)
# ----------------------------
print("4) Montando preprocessamento (ordinais para SMOTENC)...")
numeric_cols = [c for c in features if X[c].dtype.kind in "biufc" and c != "idade" or c == "idade"]
# Ensure numeric list has 'idade','mes_sinistro','dia_sinistro','ano_sinistro' etc
numeric_cols = [c for c in numeric_cols if c in features and X[c].dtype.kind in "biufc" or c == "idade"]
# but we'll explicitly pick numeric-like columns
numeric_cols = [c for c in features if c in ["idade", "mes_sinistro", "dia_sinistro", "ano_sinistro"] and c in features]
categorical_cols = [c for c in features if c not in numeric_cols]

print("  numeric_cols:", numeric_cols)
print("  categorical_cols:", categorical_cols)

# Preprocessor that yields array: [num_cols..., cat_cols_ord...]
pre_smote_transformers = []
if numeric_cols:
    pre_smote_transformers.append(("num_impute", SimpleImputer(strategy="median"), numeric_cols))
# OrdinalEncoder for categorical to feed SMOTENC
from sklearn.preprocessing import OrdinalEncoder
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
pre_smote_transformers.append(("cat_ord", ord_enc, categorical_cols))

pre_smote = ColumnTransformer(pre_smote_transformers, remainder="drop", sparse_threshold=0)

# Fit pre_smote on training only to determine categories
pre_smote.fit(X_train)

# Determine indices for categorical features in the pre_smote output
n_num = len(numeric_cols)
n_cat = len(categorical_cols)
cat_indices = list(range(n_num, n_num + n_cat))
print("  -> categorical indices for SMOTENC:", cat_indices)

# ----------------------------
# 5) Pipeline completo (pre_smote -> SMOTENC -> post processing -> clf)
#    After SMOTENC we will apply StandardScaler to numeric and OneHotEncoder to categorical.
# ----------------------------
print("5) Montando pipeline imblearn com SMOTENC e pós-processamento...")

# Post-smote column transformer: numeric scaler + onehot for categorical (indices relative to SMOTENC output)
post_smote_transformers = []
if numeric_cols:
    # numeric indices are 0..n_num-1
    post_smote_transformers.append(("num_scale", StandardScaler(), list(range(0, n_num))))
# OneHot for categorical indices after SMOTE
if n_cat > 0:
    post_smote_transformers.append(("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True), cat_indices))

post_smote = ColumnTransformer(post_smote_transformers, remainder="drop", sparse_threshold=0)

# Classifier
rf = RandomForestClassifier(random_state=RND, class_weight="balanced", n_jobs=-1)

# Create the imbalanced pipeline
smote = SMOTENC(categorical_features=cat_indices, random_state=RND)

pipeline = ImbPipeline(steps=[
    ("pre_smote", pre_smote),
    ("smote", smote),
    ("post_smote", post_smote),
    ("clf", rf)
])

# ----------------------------
# 6) Baseline: train RF without Grid but using the pipeline with SMOTENC (quick baseline)
# ----------------------------
print("6) Treinando baseline (pipeline)...")
if FAST:
    # BalancedRandomForest lida com desbalanceamento sem SMOTE; usar pipeline simplificado rápido
    print("  (FAST=1) Usando BalancedRandomForest sem SMOTENC para baseline rápido")
    pre_no_smote_fast = ColumnTransformer([
        ("num_impute", SimpleImputer(strategy="median"), numeric_cols),
        ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
    ], remainder="drop", sparse_threshold=0)
    fast_clf = BalancedRandomForestClassifier(random_state=RND, n_estimators=200, n_jobs=-1)
    baseline_fast = Pipeline([
        ("pre", pre_no_smote_fast),
        ("clf", fast_clf)
    ])
    baseline_fast.fit(X_train, y_train)
    y_pred_base = baseline_fast.predict(X_test)
    f1_base = f1_score(y_test, y_pred_base, average="macro")
    print(f"  F1-score (baseline BalancedRF FAST): {f1_base:.4f}")
    print(classification_report(y_test, y_pred_base))
else:
    pipeline.set_params(clf__n_estimators=100)
    pipeline.fit(X_train, y_train)
    y_pred_base = pipeline.predict(X_test)
    f1_base = f1_score(y_test, y_pred_base, average="macro")
    print(f"  F1-score (baseline pipeline+SMOTENC): {f1_base:.4f}")
    print(classification_report(y_test, y_pred_base))

# Salvar matriz de confusão baseline
cm = confusion_matrix(y_test, y_pred_base, labels=np.unique(y))
fig = plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
baseline_title = "Baseline (BalancedRF FAST)" if FAST else "Baseline (Pipeline + SMOTENC)"
plt.title(f"Matriz de Confusão - {baseline_title}")
baseline_fname = "matriz_confusao_baseline_fast.png" if FAST else "matriz_confusao_baseline_pipeline.png"
save_fig(fig, baseline_fname)

# ================================
# 7) GridSearchCV rápido (modo DEBUG)
# ================================
print("7) Rodando GridSearchCV (modo rápido de debug)...")

# Usar só 20% do treino para acelerar
X_train_small, _, y_train_small, _ = train_test_split(
    X_train, y_train, train_size=0.2, stratify=y_train, random_state=RND
)

# Pré-processamento (imputação + ordinal)
X_train_enc = pre_smote.transform(X_train_small)

# Aplicar SMOTENC uma vez
smote = SMOTENC(categorical_features=cat_indices, random_state=RND)
X_res, y_res = smote.fit_resample(X_train_enc, y_train_small)
print(f"  Dados balanceados (debug): {X_res.shape}, classes={np.bincount(pd.factorize(y_res)[0])}")

# Grade mínima
param_grid = {
    "n_estimators": [100],
    "max_depth": [None]
}

cv = StratifiedKFold(n_splits=2, shuffle=True, random_state=RND)

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=RND, class_weight="balanced", n_jobs=-1),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=2,
    refit=True
)

grid.fit(X_res, y_res)

print("  Melhor params (debug):", grid.best_params_)
print(f"  Melhor score CV (debug f1_macro): {grid.best_score_:.4f}")

best_rf_debug = grid.best_estimator_

# Avaliar no conjunto de teste
X_test_enc = pre_smote.transform(X_test)
y_pred_grid = best_rf_debug.predict(X_test_enc)

f1_grid = f1_score(y_test, y_pred_grid, average="macro")
print(f"  F1-score no conjunto de teste (debug): {f1_grid:.4f}")
print(classification_report(y_test, y_pred_grid))

cm = confusion_matrix(y_test, y_pred_grid, labels=np.unique(y))
fig = plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="coolwarm",
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Matriz de Confusão - GridSearch Debug")
save_fig(fig, "matriz_confusao_grid_debug.png")

# ----------------------------
# 8) GridSearchCV sem SMOTENC (apenas pre_smote but without SMOTE step)
#    -> pipeline_no_smote: pre_smote -> post_smote_without_onehot? Simpler: pre_no_smote: do numeric impute + onehot (no ordinals)
# ----------------------------
print("8) Rodando GridSearchCV sem SMOTE (para comparação)...")

# Preprocessor without SMOTE: impute numeric + OneHot encode categoricals directly (no ordinals)
pre_no_smote = ColumnTransformer([
    ("num_impute", SimpleImputer(strategy="median"), numeric_cols),
    ("cat_ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_cols)
], remainder="drop", sparse_threshold=1.0)

pipeline_no_smote = Pipeline([
    ("pre_no_smote", pre_no_smote),
    ("clf", RandomForestClassifier(random_state=RND, class_weight="balanced", n_jobs=-1))
])

param_grid_no_smote = {
    "clf__n_estimators": [100, 200],
    "clf__max_depth": [None, 10, 20],
    "clf__min_samples_split": [2, 5]
}

if FAST:
    print("  (FAST=1) Usando RandomizedSearchCV no pipeline SEM SMOTE")
    from scipy.stats import randint
    rand_params_no_smote = {
        "clf__n_estimators": randint(100, 300),
        "clf__max_depth": [None, 10, 20],
        "clf__min_samples_split": randint(2, 6)
    }
    grid_no_smote = RandomizedSearchCV(
        estimator=pipeline_no_smote,
        param_distributions=rand_params_no_smote,
        n_iter=8,
        scoring="f1_macro",
        cv=cv,
        n_jobs=max(1, os.cpu_count()//2),
        verbose=2,
        refit=True,
        random_state=RND
    )
else:
    grid_no_smote = GridSearchCV(
        estimator=pipeline_no_smote,
        param_grid=param_grid_no_smote,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=2,
        refit=True
    )
grid_no_smote.fit(X_train, y_train)
print("  Melhor params (no smote):", grid_no_smote.best_params_)
print(f"  Melhor score CV (no smote): {grid_no_smote.best_score_:.4f}")

best_no_smote = grid_no_smote.best_estimator_
y_pred_no_smote = best_no_smote.predict(X_test)
f1_no_smote = f1_score(y_test, y_pred_no_smote, average="macro")
print(f"  F1-score no teste (no smote): {f1_no_smote:.4f}")

# Matriz
cm = confusion_matrix(y_test, y_pred_no_smote, labels=np.unique(y))
fig = plt.figure()
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title("Matriz de Confusão - GridSearch (SEM SMOTE)")
save_fig(fig, "matriz_confusao_grid_no_smote.png")

# ----------------------------
# 9) Calibração do melhor modelo (a partir do melhor_pipeline)
# ----------------------------
if FAST:
    print("9) Calibração ignorada no modo FAST (defina FAST=0 para habilitar).")
    y_pred_cal = y_pred_grid
    f1_cal = f1_grid
else:
    print("9) Calibrando probabilidades do melhor pipeline (CalibratedClassifierCV) - usando pipeline SEM SMOTE...")
    calibrator = CalibratedClassifierCV(base_estimator=best_no_smote, cv=3, method="isotonic")
    calibrator.fit(X_train, y_train)
    y_pred_cal = calibrator.predict(X_test)
    f1_cal = f1_score(y_test, y_pred_cal, average="macro")
    print(f"  F1 (calibrated): {f1_cal:.4f}")
    print(classification_report(y_test, y_pred_cal))

    cm = confusion_matrix(y_test, y_pred_cal, labels=np.unique(y))
    fig = plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples", xticklabels=np.unique(y), yticklabels=np.unique(y))
    plt.title("Matriz de Confusão - Modelo Calibrado (No-SMOTE)")
    save_fig(fig, "matriz_confusao_calibrado.png")

# ----------------------------
# 10) Permutation importance (no conjunto de teste) para explicar modelo
# ----------------------------
if FAST:
    print("10) Permutation importance ignorado no modo FAST (defina FAST=0 para habilitar).")
    importances = pd.Series(dtype=float)
else:
    print("10) Calculando permutation importance (teste) no pipeline SEM SMOTE...")
    res = permutation_importance(best_no_smote, X_test, y_test, n_repeats=10, random_state=RND, n_jobs=-1)
    importances = pd.Series(res.importances_mean, index=X.columns).sort_values(ascending=False)

    fig = plt.figure(figsize=(10,6))
    sns.barplot(x=importances.values, y=importances.index)
    plt.title("Permutation Importance (média sobre 10 runs)")
    save_fig(fig, "permutation_importance.png")

# ----------------------------
# 11) Comparação resumida e salvar resultados
# ----------------------------
print("11) Resumo comparativo e salvamento de artefatos...")

results = {
    "model": ["baseline", "grid_pipeline_smote", "grid_no_smote", "calibrated_or_grid"],
    "f1_macro": [f1_base, f1_grid, f1_no_smote, f1_cal]
}
df_results = pd.DataFrame(results)
df_results.to_csv("outputs/comparacao_modelos.csv", index=False)
print("  comparacao_modelos.csv salvo em outputs/")

# Salvar modelos
if not FAST:
    # Opcionalmente salvar baseline com SMOTE se executado em modo completo
    joblib.dump(pipeline, "outputs/pipeline_baseline_smote.joblib")
    joblib.dump(best_no_smote, "outputs/pipeline_grid_no_smote_best.joblib")
    joblib.dump(calibrator, "outputs/pipeline_grid_smote_calibrado.joblib")
else:
    # Salvar somente o melhor pipeline sem SMOTE (pronto para produção) e, opcionalmente, baseline FAST
    joblib.dump(best_no_smote, "outputs/pipeline_grid_no_smote_best.joblib")
    try:
        joblib.dump(baseline_fast, "outputs/pipeline_baseline_balancedrf_fast.joblib")
    except Exception:
        pass
print("  Modelos salvos em outputs/")

# Salvar importances e classe distribution
if not importances.empty:
    importances.to_csv("outputs/permutation_importance.csv")
pd.Series(y_test).value_counts().to_csv("outputs/distribuicao_classes_teste.csv")

# ----------------------------
# 12) Finalização
# ----------------------------
print("\nFINAL: comparação de F1 (macro):")
print(df_results.to_string(index=False))

print("\nArquivos gerados em './outputs':")
for f in sorted(os.listdir("outputs")):
    print(" -", f)