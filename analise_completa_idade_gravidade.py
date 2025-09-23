import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Configurações
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.style.use('default')

print("🚗 ANÁLISE COMPLETA: FAIXA ETÁRIA vs GRAVIDADE + MODELO PREDITIVO")
print("=" * 75)
print("Região Metropolitana de São Paulo - Acidentes de Motocicleta (2015-2021)")
print("=" * 75)

# ================================
# PARTE I: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# ================================

print("\n📊 PARTE I: CARREGAMENTO E PREPARAÇÃO DOS DADOS")
print("-" * 55)

df = pd.read_csv("pessoas_2015-2021.csv", encoding='latin-1', sep=';')
print(f"Dados originais: {df.shape[0]:,} registros")

# Aplicar filtros
df = df[(df["tipo_veiculo_vitima"] == "MOTOCICLETA") & 
        (df["regiao_administrativa"] == "METROPOLITANA DE SÃO PAULO")]
print(f"Após filtro (motocicletas Grande SP): {df.shape[0]:,} registros")

df = df[df["gravidade_lesao"] != "NAO DISPONIVEL"]
print(f"Após remover gravidade N/D: {df.shape[0]:,} registros")

df = df[df["faixa_etaria_legal"] != "NAO DISPONIVEL"]
print(f"Após remover faixa etária N/D: {df.shape[0]:,} registros")

# ================================
# PARTE II: ANÁLISE EXPLORATÓRIA DETALHADA
# ================================

print("\n📈 PARTE II: ANÁLISE EXPLORATÓRIA - FAIXA ETÁRIA vs GRAVIDADE")
print("-" * 65)

# 2.1 Distribuição Geral
print("\n2.1 DISTRIBUIÇÃO GERAL DOS DADOS:")
print(f"   • Total de acidentes analisados: {len(df):,}")
print(f"   • Período: 2015-2021")
print(f"   • Distribuição por gravidade:")

gravidade_dist = df['gravidade_lesao'].value_counts()
for gravidade, count in gravidade_dist.items():
    pct = count / len(df) * 100
    print(f"     - {gravidade}: {count:,} cases ({pct:.1f}%)")

# 2.2 Análise Cruzada Detalhada
print("\n2.2 ANÁLISE CRUZADA DETALHADA:")

# Tabela cruzada
crosstab = pd.crosstab(df['faixa_etaria_legal'], df['gravidade_lesao'], margins=True)
print("\n   TABELA DE CONTINGÊNCIA (valores absolutos):")
print(crosstab)

# Percentuais por faixa etária
crosstab_perc = pd.crosstab(df['faixa_etaria_legal'], df['gravidade_lesao'], normalize='index') * 100
print("\n   PERCENTUAIS POR FAIXA ETÁRIA:")
print(crosstab_perc.round(1))

# 2.3 Análise de Risco Detalhada
print("\n2.3 RANKING DE RISCO POR FAIXA ETÁRIA:")

risk_analysis = []
for idade in crosstab.index[:-1]:  # Excluir linha 'All'
    total = crosstab.loc[idade, 'All']
    fatal = crosstab.loc[idade, 'FATAL']
    grave = crosstab.loc[idade, 'GRAVE']
    leve = crosstab.loc[idade, 'LEVE']
    
    taxa_fatal = (fatal / total) * 100
    taxa_grave = (grave / total) * 100
    taxa_grave_fatal = ((grave + fatal) / total) * 100
    
    risk_analysis.append({
        'Faixa_Etaria': idade,
        'Total_Acidentes': total,
        'Taxa_Fatal_Pct': taxa_fatal,
        'Taxa_Grave_Pct': taxa_grave,
        'Taxa_Grave_Fatal_Pct': taxa_grave_fatal,
        'Casos_Fatais': fatal,
        'Casos_Graves': grave,
        'Casos_Leves': leve
    })

risk_df = pd.DataFrame(risk_analysis)
risk_df = risk_df.sort_values('Taxa_Grave_Fatal_Pct', ascending=False)

print(f"\n{'Faixa Etária':<12} {'Total':<7} {'Fatal%':<8} {'Grave%':<8} {'G+F%':<8} {'Fatal':<6} {'Grave':<6}")
print("-" * 70)
for _, row in risk_df.iterrows():
    print(f"{row['Faixa_Etaria']:<12} {row['Total_Acidentes']:<7} "
          f"{row['Taxa_Fatal_Pct']:<8.1f} {row['Taxa_Grave_Pct']:<8.1f} "
          f"{row['Taxa_Grave_Fatal_Pct']:<8.1f} {row['Casos_Fatais']:<6} {row['Casos_Graves']:<6}")

# 2.4 Principais Insights
print("\n2.4 PRINCIPAIS INSIGHTS EXPLORATÓRIOS:")
faixa_mais_risco = risk_df.iloc[0]
faixa_menos_risco = risk_df.iloc[-1]
faixa_mais_acidentes = risk_df.sort_values('Total_Acidentes', ascending=False).iloc[0]

print(f"   🚨 MAIOR RISCO: {faixa_mais_risco['Faixa_Etaria']} anos")
print(f"      - Taxa graves/fatais: {faixa_mais_risco['Taxa_Grave_Fatal_Pct']:.1f}%")
print(f"      - Total acidentes: {faixa_mais_risco['Total_Acidentes']:,}")

print(f"   ✅ MENOR RISCO: {faixa_menos_risco['Faixa_Etaria']} anos")
print(f"      - Taxa graves/fatais: {faixa_menos_risco['Taxa_Grave_Fatal_Pct']:.1f}%")
print(f"      - Total acidentes: {faixa_menos_risco['Total_Acidentes']:,}")

print(f"   📊 MAIOR VOLUME: {faixa_mais_acidentes['Faixa_Etaria']} anos")
print(f"      - Total: {faixa_mais_acidentes['Total_Acidentes']:,} acidentes ({faixa_mais_acidentes['Total_Acidentes']/len(df)*100:.1f}% do total)")

# ================================
# PARTE III: MODELO PREDITIVO OTIMIZADO
# ================================

print("\n🤖 PARTE III: MODELO PREDITIVO OTIMIZADO")
print("-" * 50)

# 3.1 Preparação dos dados para ML
print("\n3.1 PREPARAÇÃO DOS DADOS PARA MACHINE LEARNING:")

# Criar cópia para ML
df_ml = df.copy()
df_ml = df_ml[df_ml["sexo"] != "NAO DISPONIVEL"]

print("   Removendo variáveis desnecessárias:")
# Lista de colunas a remover
colunas_remover = [
    'id_sinistro',           # Identificador único
    'id_veiculo',            # Pouco valor preditivo
    'cod_ibge',              # Redundante com municipio
    'idade',                 # Redundante com faixa_etaria_legal
    'regiao_administrativa', # Já filtrado
    'tipo_veiculo_vitima',   # Já filtrado
    'data_sinistro',         # Redundante com ano/mes/dia
    'data_obito',            # Muitos ausentes
    'ano_mes_sinistro',      # Redundante
    'ano_mes_obito',         # Muitos ausentes
    'grau_de_instrucao',     # Muitos ausentes
    'profissao',             # Muitos ausentes
    'nacionalidade',         # Muitos ausentes
    'ano_obito',             # Muitos ausentes
    'mes_obito',             # Muitos ausentes  
    'dia_obito',             # Muitos ausentes
    'local_obito',           # Muitos ausentes
    'tempo_sinistro_obito'   # Muitos ausentes
]

colunas_existentes = [col for col in colunas_remover if col in df_ml.columns]
df_ml = df_ml.drop(columns=colunas_existentes)

for col in ['id_sinistro', 'id_veiculo', 'cod_ibge', 'idade']:
    if col in colunas_existentes:
        print(f"   ✅ {col} - Redundante ou sem valor preditivo")

print(f"   • Colunas restantes: {df_ml.shape[1]}")
print(f"   • Variáveis finais: {list(df_ml.columns)}")

# 3.2 Codificação de variáveis
print("\n3.2 CODIFICAÇÃO DAS VARIÁVEIS:")

# Sexo
df_ml["sexo"] = df_ml["sexo"].map({"FEMININO": 0, "MASCULINO": 1})

# Outras categóricas
categorical_columns = ['municipio', 'tipo_via', 'tipo_de_vitima', 
                      'faixa_etaria_legal', 'faixa_etaria_demografica']

label_encoders = {}
for col in categorical_columns:
    if col in df_ml.columns:
        le = LabelEncoder()
        df_ml = df_ml[df_ml[col] != "NAO DISPONIVEL"]  # Remove N/A
        df_ml[col] = le.fit_transform(df_ml[col].astype(str))
        label_encoders[col] = le
        print(f"   • {col}: {len(le.classes_)} categorias")

# Target
gravity_encoder = LabelEncoder()
df_ml["gravidade_lesao"] = gravity_encoder.fit_transform(df_ml["gravidade_lesao"])

# 3.3 Separação features/target
print(f"\n3.3 DATASET FINAL PARA MODELAGEM:")
X = df_ml.drop(columns=["gravidade_lesao"])
y = df_ml["gravidade_lesao"]

# Limpar valores ausentes restantes
X = X.dropna()
y = y[X.index]

print(f"   • Amostras finais: {X.shape[0]:,}")
print(f"   • Features: {X.shape[1]}")
print(f"   • Distribuição das classes:")
for i, classe in enumerate(gravity_encoder.classes_):
    count = (y == i).sum()
    pct = count / len(y) * 100
    print(f"     - {classe}: {count:,} ({pct:.1f}%)")

# ================================
# PARTE IV: TREINAMENTO E AVALIAÇÃO
# ================================

print("\n🎯 PARTE IV: TREINAMENTO E AVALIAÇÃO DO MODELO")
print("-" * 50)

# 4.1 Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"4.1 DIVISÃO DOS DADOS:")
print(f"   • Treino: {X_train.shape[0]:,} amostras")
print(f"   • Teste: {X_test.shape[0]:,} amostras")

# 4.2 Treinamento
print(f"\n4.2 TREINAMENTO DO MODELO:")
clf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    class_weight="balanced",
    max_depth=10,
    min_samples_split=100,
    min_samples_leaf=50
)

clf.fit(X_train, y_train)
print("   ✅ Modelo Random Forest treinado com sucesso!")

# 4.3 Predição e avaliação
print(f"\n4.3 AVALIAÇÃO DO MODELO:")
y_pred = clf.predict(X_test)

f1_macro = f1_score(y_test, y_pred, average="macro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")

print(f"   📊 MÉTRICAS DE PERFORMANCE:")
print(f"   • F1-score (macro): {f1_macro:.4f}")
print(f"   • F1-score (weighted): {f1_weighted:.4f}")

# 4.4 Importância das features
print(f"\n4.4 IMPORTÂNCIA DAS FEATURES:")
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': clf.feature_importances_
}).sort_values('importance', ascending=False)

for i, (_, row) in enumerate(feature_importance.iterrows(), 1):
    print(f"   {i:2d}. {row['feature']:<25}: {row['importance']:.4f} ({row['importance']:.1%})")

# 4.5 Análise específica da faixa etária
print(f"\n4.5 ANÁLISE DA FAIXA ETÁRIA NO MODELO:")
faixa_importance = feature_importance[feature_importance['feature'] == 'faixa_etaria_legal']

if not faixa_importance.empty:
    importancia = faixa_importance.iloc[0]['importance']
    posicao = feature_importance.reset_index(drop=True).index[feature_importance['feature'] == 'faixa_etaria_legal'][0] + 1
    
    print(f"   • Importância da faixa etária: {importancia:.4f} ({importancia:.1%})")
    print(f"   • Posição no ranking: {posicao}º lugar de {len(feature_importance)}")
    print(f"   • Status: {'✅ Entre as top features!' if posicao <= 5 else '⚠️ Importância moderada' if posicao <= 8 else '🔍 Baixa importância'}")

# ================================
# PARTE V: RELATÓRIO DETALHADO
# ================================

print(f"\n📋 PARTE V: RELATÓRIO DETALHADO DE CLASSIFICAÇÃO")
print("-" * 55)

target_names = gravity_encoder.classes_
print(classification_report(y_test, y_pred, target_names=target_names))

# ================================
# PARTE VI: CONCLUSÕES E RECOMENDAÇÕES
# ================================

print(f"\n💡 PARTE VI: CONCLUSÕES E RECOMENDAÇÕES")
print("-" * 45)

print("6.1 PRINCIPAIS ACHADOS:")
print(f"   🎯 ANÁLISE EXPLORATÓRIA:")
print(f"      • Faixa de maior risco: {faixa_mais_risco['Faixa_Etaria']} ({faixa_mais_risco['Taxa_Grave_Fatal_Pct']:.1f}% graves/fatais)")
print(f"      • Faixa de menor risco: {faixa_menos_risco['Faixa_Etaria']} ({faixa_menos_risco['Taxa_Grave_Fatal_Pct']:.1f}% graves/fatais)")
print(f"      • Maior volume de casos: {faixa_mais_acidentes['Faixa_Etaria']} ({faixa_mais_acidentes['Total_Acidentes']:,} acidentes)")

print(f"\n   🤖 MODELO PREDITIVO:")
print(f"      • F1-score balanceado: {f1_macro:.3f}")
print(f"      • Features otimizadas: {X.shape[1]} variáveis essenciais")
print(f"      • Faixa etária: {posicao}º lugar em importância")

print(f"\n6.2 RECOMENDAÇÕES DE POLÍTICAS PÚBLICAS:")
print("   🎯 FOCO PRIORITÁRIO:")

# Top 3 faixas de risco
top_3_risk = risk_df.head(3)
for i, (_, row) in enumerate(top_3_risk.iterrows(), 1):
    print(f"      {i}. Faixa {row['Faixa_Etaria']}: {row['Taxa_Grave_Fatal_Pct']:.1f}% de risco")
    if row['Faixa_Etaria'] == '0-17':
        print(f"         → Fiscalização de habilitação irregular")
    elif '18-24' in row['Faixa_Etaria']:
        print(f"         → Campanhas educativas para jovens")
    elif row['Faixa_Etaria'].startswith('6') or row['Faixa_Etaria'].startswith('7') or row['Faixa_Etaria'].startswith('8'):
        print(f"         → Avaliação médica periódica")

print(f"\n   📊 VOLUME DE IMPACTO:")
print(f"      • Faixa {faixa_mais_acidentes['Faixa_Etaria']}: {faixa_mais_acidentes['Total_Acidentes']:,} casos/ano em média")
print(f"      • Representa {faixa_mais_acidentes['Total_Acidentes']/len(df)*100:.1f}% de todos os acidentes")

print(f"\n✅ ANÁLISE COMPLETA CONCLUÍDA!")
print("=" * 75)