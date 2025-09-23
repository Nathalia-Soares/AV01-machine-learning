import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report

# 1. Carregar dados
df = pd.read_csv("pessoas_2015-2021.csv", encoding='latin-1', sep=';')

# 2. Filtrar motocicletas na Grande SP
df = df[(df["tipo_veiculo_vitima"] == "MOTOCICLETA") & 
        (df["regiao_administrativa"] == "METROPOLITANA DE SÃO PAULO")]

# 3. Definir target (gravidade da lesão) e remover casos inválidos
df = df[df["gravidade_lesao"] != "NAO DISPONIVEL"]

# 4. Discretização manual
# Remove colunas que não são necessárias ou redundantes, e colunas com muitos valores ausentes
df = df.drop(columns=['data_sinistro', 'data_obito', 'ano_mes_sinistro', 'ano_mes_obito', 
                      'grau_de_instrucao', 'profissao', 'nacionalidade',
                      'ano_obito', 'mes_obito', 'dia_obito', 'local_obito', 'tempo_sinistro_obito'])

# Tratar valores ausentes na coluna sexo
df = df[df["sexo"] != "NAO DISPONIVEL"]

df["sexo"] = df["sexo"].map({"FEMININO": 0, "MASCULINO": 1})

municipios_unicos = sorted(df["municipio"].unique())
mapa_municipio = {mun: i for i, mun in enumerate(municipios_unicos)}
df["municipio"] = df["municipio"].map(mapa_municipio)

# Como já filtramos por regiao_administrativa e tipo_veiculo_vitima, podemos removê-las
df = df.drop(columns=['regiao_administrativa', 'tipo_veiculo_vitima'])

mapa_via = {via: i for i, via in enumerate(sorted(df["tipo_via"].unique()))}
df["tipo_via"] = df["tipo_via"].map(mapa_via)

mapa_vitima = {vit: i for i, vit in enumerate(sorted(df["tipo_de_vitima"].unique()))}
df["tipo_de_vitima"] = df["tipo_de_vitima"].map(mapa_vitima)

mapa_faixa = {fx: i for i, fx in enumerate(sorted(df["faixa_etaria_legal"].unique()))}
df["faixa_etaria_legal"] = df["faixa_etaria_legal"].map(mapa_faixa)

mapa_faixa_demo = {fx: i for i, fx in enumerate(sorted(df["faixa_etaria_demografica"].unique()))}
df["faixa_etaria_demografica"] = df["faixa_etaria_demografica"].map(mapa_faixa_demo)

mapa_gravidade = {grav: i for i, grav in enumerate(sorted(df["gravidade_lesao"].unique()))}
df["gravidade_lesao"] = df["gravidade_lesao"].map(mapa_gravidade)

# 5. Separar features e target
X = df.drop(columns=["gravidade_lesao", "id_sinistro", "id_veiculo"])
y = df["gravidade_lesao"]

# Preencher valores ausentes na idade com a mediana
X['idade'] = X['idade'].fillna(X['idade'].median())

# Remover linhas com valores ausentes restantes (deve ser muito poucos agora)
X = X.dropna()
y = y[X.index]

print(f"Amostras finais: {X.shape[0]}")
print(f"Distribuição das classes: {y.value_counts()}")

# 6. Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Treinamento Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
clf.fit(X_train, y_train)

# 8. Avaliação com F1-score
y_pred = clf.predict(X_test)

# F1-score macro (todas as classes com peso igual)
f1_macro = f1_score(y_test, y_pred, average="macro")
# F1-score ponderado (considera o tamanho de cada classe)
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"F1-score (macro): {f1_macro:.4f}")
print(f"F1-score (weighted): {f1_weighted:.4f}")

# Opcional: relatório detalhado por classe
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred))
