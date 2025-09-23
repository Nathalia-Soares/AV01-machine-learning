import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import font_manager

# Configuração de estilo
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 10

print("🎨 CRIANDO VISUALIZAÇÕES: FAIXA ETÁRIA vs GRAVIDADE")
print("=" * 55)

# Carregar e preparar dados
df = pd.read_csv("pessoas_2015-2021.csv", encoding='latin-1', sep=';')
df = df[(df["tipo_veiculo_vitima"] == "MOTOCICLETA") & 
        (df["regiao_administrativa"] == "METROPOLITANA DE SÃO PAULO")]
df = df[df["gravidade_lesao"] != "NAO DISPONIVEL"]
df = df[df["faixa_etaria_legal"] != "NAO DISPONIVEL"]

# Calcular dados para visualização
crosstab = pd.crosstab(df['faixa_etaria_legal'], df['gravidade_lesao'])
crosstab_perc = pd.crosstab(df['faixa_etaria_legal'], df['gravidade_lesao'], normalize='index') * 100

# Análise de risco
risk_analysis = []
for idade in crosstab.index:
    total = crosstab.loc[idade].sum()
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

# Criar figura com subplots
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('ANÁLISE DE ACIDENTES DE MOTOCICLETA: FAIXA ETÁRIA vs GRAVIDADE\nRegião Metropolitana de São Paulo (2015-2021)', 
             fontsize=16, fontweight='bold', y=0.98)

# Gráfico 1: Distribuição total por faixa etária
ax1 = axes[0, 0]
risk_df_sorted = risk_df.sort_values('Total_Acidentes', ascending=True)
bars1 = ax1.barh(range(len(risk_df_sorted)), risk_df_sorted['Total_Acidentes'], color='steelblue', alpha=0.8)
ax1.set_yticks(range(len(risk_df_sorted)))
ax1.set_yticklabels(risk_df_sorted['Faixa_Etaria'], fontsize=9)
ax1.set_xlabel('Número Total de Acidentes')
ax1.set_title('Distribuição de Acidentes por Faixa Etária', fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Adicionar valores nas barras
for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
             f'{int(width):,}', ha='left', va='center', fontsize=8)

# Gráfico 2: Taxa de risco (Grave + Fatal) por faixa etária
ax2 = axes[0, 1]
risk_df_risk = risk_df.sort_values('Taxa_Grave_Fatal_Pct', ascending=True)
colors = ['red' if x > 20 else 'orange' if x > 15 else 'green' for x in risk_df_risk['Taxa_Grave_Fatal_Pct']]
bars2 = ax2.barh(range(len(risk_df_risk)), risk_df_risk['Taxa_Grave_Fatal_Pct'], color=colors, alpha=0.8)
ax2.set_yticks(range(len(risk_df_risk)))
ax2.set_yticklabels(risk_df_risk['Faixa_Etaria'], fontsize=9)
ax2.set_xlabel('Taxa de Casos Graves/Fatais (%)')
ax2.set_title('Taxa de Risco por Faixa Etária', fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Adicionar valores nas barras
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', ha='left', va='center', fontsize=8)

# Gráfico 3: Heatmap da tabela cruzada (valores absolutos)
ax3 = axes[1, 0]
crosstab_display = crosstab.reindex(['0-17', '18-24', '25-29', '30-34', '35-39', '40-44', 
                                    '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 ou mais'])
sns.heatmap(crosstab_display.T, annot=True, fmt='d', cmap='YlOrRd', ax=ax3, 
            cbar_kws={'label': 'Número de Casos'})
ax3.set_title('Distribuição Absoluta: Faixa Etária x Gravidade', fontweight='bold')
ax3.set_ylabel('Gravidade da Lesão')
ax3.set_xlabel('Faixa Etária')
plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')

# Gráfico 4: Gráfico de barras empilhadas (percentual)
ax4 = axes[1, 1]
crosstab_perc_display = crosstab_perc.reindex(['0-17', '18-24', '25-29', '30-34', '35-39', '40-44', 
                                              '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80 ou mais'])

# Criar barras empilhadas
bottom_grave = np.zeros(len(crosstab_perc_display))
bottom_leve = crosstab_perc_display['FATAL'] + crosstab_perc_display['GRAVE']

bars_fatal = ax4.bar(range(len(crosstab_perc_display)), crosstab_perc_display['FATAL'], 
                    label='Fatal', color='darkred', alpha=0.8)
bars_grave = ax4.bar(range(len(crosstab_perc_display)), crosstab_perc_display['GRAVE'], 
                    bottom=crosstab_perc_display['FATAL'], label='Grave', color='orange', alpha=0.8)
bars_leve = ax4.bar(range(len(crosstab_perc_display)), crosstab_perc_display['LEVE'], 
                   bottom=bottom_leve, label='Leve', color='green', alpha=0.8)

ax4.set_xticks(range(len(crosstab_perc_display)))
ax4.set_xticklabels(crosstab_perc_display.index, rotation=45, ha='right')
ax4.set_ylabel('Percentual (%)')
ax4.set_title('Distribuição Percentual por Faixa Etária', fontweight='bold')
ax4.legend()
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('analise_faixa_etaria_gravidade.png', dpi=300, bbox_inches='tight')
print("📊 Gráficos salvos como 'analise_faixa_etaria_gravidade.png'")

# Criar segunda figura com análises adicionais
fig2, axes2 = plt.subplots(2, 2, figsize=(16, 10))
fig2.suptitle('ANÁLISES COMPLEMENTARES: PADRÕES E TENDÊNCIAS', fontsize=16, fontweight='bold', y=0.98)

# Gráfico 5: Evolução por ano
df['ano'] = df['ano_sinistro']
evolucao_ano = df.groupby(['ano', 'gravidade_lesao']).size().unstack(fill_value=0)
ax5 = axes2[0, 0]
evolucao_ano.plot(kind='line', ax=ax5, marker='o', linewidth=2)
ax5.set_title('Evolução Temporal dos Acidentes', fontweight='bold')
ax5.set_xlabel('Ano')
ax5.set_ylabel('Número de Casos')
ax5.legend(title='Gravidade')
ax5.grid(True, alpha=0.3)

# Gráfico 6: Distribuição por sexo e faixa etária (apenas top 6 faixas com mais acidentes)
ax6 = axes2[0, 1]
df_clean = df[df['sexo'] != 'NAO DISPONIVEL']
top_faixas = df['faixa_etaria_legal'].value_counts().head(6).index
df_top_faixas = df_clean[df_clean['faixa_etaria_legal'].isin(top_faixas)]

sexo_faixa = pd.crosstab(df_top_faixas['faixa_etaria_legal'], df_top_faixas['sexo'])
sexo_faixa_pct = sexo_faixa.div(sexo_faixa.sum(axis=1), axis=0) * 100

sexo_faixa_pct.plot(kind='bar', ax=ax6, color=['pink', 'lightblue'], alpha=0.8)
ax6.set_title('Distribuição por Sexo (Top 6 Faixas Etárias)', fontweight='bold')
ax6.set_xlabel('Faixa Etária')
ax6.set_ylabel('Percentual (%)')
ax6.legend(['Feminino', 'Masculino'])
plt.setp(ax6.get_xticklabels(), rotation=45, ha='right')
ax6.grid(axis='y', alpha=0.3)

# Gráfico 7: Comparação das 3 faixas de maior risco vs 3 de menor risco
ax7 = axes2[1, 0]
top_3_risk = risk_df.nlargest(3, 'Taxa_Grave_Fatal_Pct')
bottom_3_risk = risk_df.nsmallest(3, 'Taxa_Grave_Fatal_Pct')

labels = list(top_3_risk['Faixa_Etaria']) + list(bottom_3_risk['Faixa_Etaria'])
values = list(top_3_risk['Taxa_Grave_Fatal_Pct']) + list(bottom_3_risk['Taxa_Grave_Fatal_Pct'])
colors_risk = ['darkred'] * 3 + ['green'] * 3

bars7 = ax7.bar(range(len(labels)), values, color=colors_risk, alpha=0.8)
ax7.set_xticks(range(len(labels)))
ax7.set_xticklabels(labels, rotation=45, ha='right')
ax7.set_ylabel('Taxa de Casos Graves/Fatais (%)')
ax7.set_title('Maior vs Menor Risco por Faixa Etária', fontweight='bold')
ax7.axhline(y=risk_df['Taxa_Grave_Fatal_Pct'].mean(), color='orange', linestyle='--', 
           label=f'Média Geral: {risk_df["Taxa_Grave_Fatal_Pct"].mean():.1f}%')
ax7.legend()
ax7.grid(axis='y', alpha=0.3)

# Adicionar valores nas barras
for i, bar in enumerate(bars7):
    height = bar.get_height()
    ax7.text(bar.get_x() + bar.get_width()/2., height + 0.5,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Gráfico 8: Relação idade numérica vs gravidade
ax8 = axes2[1, 1]
df_idade = df[df['idade'].notna() & (df['idade'] > 0) & (df['idade'] <= 100)]

# Criar bins de idade
df_idade['idade_bin'] = pd.cut(df_idade['idade'], bins=[0, 18, 25, 35, 45, 55, 65, 100], 
                               labels=['0-17', '18-24', '25-34', '35-44', '45-54', '55-64', '65+'])

idade_gravidade = pd.crosstab(df_idade['idade_bin'], df_idade['gravidade_lesao'], normalize='index') * 100
idade_gravidade[['FATAL', 'GRAVE']].plot(kind='bar', ax=ax8, color=['darkred', 'orange'], alpha=0.8)
ax8.set_title('Taxa de Casos Graves/Fatais por Grupo de Idade', fontweight='bold')
ax8.set_xlabel('Grupo de Idade')
ax8.set_ylabel('Percentual (%)')
ax8.legend(['Fatal', 'Grave'])
plt.setp(ax8.get_xticklabels(), rotation=45, ha='right')
ax8.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.savefig('analise_complementar_acidentes.png', dpi=300, bbox_inches='tight')
print("📊 Gráficos complementares salvos como 'analise_complementar_acidentes.png'")

plt.show()

print("\n✅ VISUALIZAÇÕES CRIADAS COM SUCESSO!")
print("📁 Arquivos gerados:")
print("  • analise_faixa_etaria_gravidade.png")
print("  • analise_complementar_acidentes.png")