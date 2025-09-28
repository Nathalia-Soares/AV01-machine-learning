import os
import pandas as pd
import numpy as np
from datetime import datetime

def carregar_planilha(nome_arquivo, encoding='latin-1', sep=';'):
    if not os.path.exists(nome_arquivo):
        print(f"ERRO: Arquivo '{nome_arquivo}' não encontrado!")
        return None
        
    try:
        print(f"Carregando {nome_arquivo}...")
        df = pd.read_csv(nome_arquivo, encoding=encoding, sep=sep, low_memory=False)
        print(f"{nome_arquivo} carregado: {len(df):,} registros, {len(df.columns)} colunas")
        return df
    except Exception as e:
        print(f"ERRO ao carregar {nome_arquivo}: {e}")
        return None

def filtrar_por_anos(df, nome_arquivo, coluna_ano='ano_sinistro', anos_desejados=range(2020, 2026)):
    if df is None or df.empty:
        print(f"{nome_arquivo}: DataFrame vazio, pulando filtro")
        return None
        
    print(f"\nFiltrando {nome_arquivo} por {coluna_ano}...")
    
    # Verificar se a coluna existe
    if coluna_ano not in df.columns:
        print(f"ERRO: Coluna '{coluna_ano}' não encontrada em {nome_arquivo}!")
        print(f"Colunas disponíveis: {sorted(df.columns[:10])}...")
        return None
    
    # Mostrar distribuição original por ano
    print(f"Distribuição original por ano em {nome_arquivo}:")
    anos_originais = df[coluna_ano].value_counts().sort_index()
    for ano, count in anos_originais.head(10).items():
        print(f"   {ano}: {count:,} registros")
    
    # Converter para numérico se necessário
    df[coluna_ano] = pd.to_numeric(df[coluna_ano], errors='coerce')
    
    # Filtrar pelos anos desejados (2020-2025)
    anos_lista = list(anos_desejados)
    mask = df[coluna_ano].isin(anos_lista)
    df_filtrado = df[mask].copy()
    
    print(f"Filtro aplicado para anos {anos_lista[0]}-{anos_lista[-1]}")
    print(f"   Antes: {len(df):,} registros")
    print(f"   Depois: {len(df_filtrado):,} registros")
    
    # Mostrar distribuição filtrada por ano
    if not df_filtrado.empty:
        print("Distribuição filtrada por ano:")
        distribuicao = df_filtrado[coluna_ano].value_counts().sort_index()
        for ano, count in distribuicao.items():
            print(f"   {int(ano)}: {count:,} registros")
    
    return df_filtrado

def consolidar_dados(df1, df2, nome1="Planilha 1", nome2="Planilha 2"):
    """
    Consolida dois DataFrames em um único DataFrame.
    
    Args:
        df1 (pd.DataFrame): Primeiro DataFrame
        df2 (pd.DataFrame): Segundo DataFrame
        nome1 (str): Nome do primeiro DataFrame
        nome2 (str): Nome do segundo DataFrame
        
    Returns:
        pd.DataFrame: DataFrame consolidado
    """
    print(f"\nConsolidando dados...")
    
    dataframes_validos = []
    
    if df1 is not None and not df1.empty:
        dataframes_validos.append(df1)
        print(f"{nome1}: {len(df1):,} registros")
    else:
        print(f"{nome1}: DataFrame vazio ou inválido")
    
    if df2 is not None and not df2.empty:
        dataframes_validos.append(df2)
        print(f"{nome2}: {len(df2):,} registros")
    else:
        print(f"{nome2}: DataFrame vazio ou inválido")
    
    if not dataframes_validos:
        print("ERRO: Nenhum DataFrame válido para consolidar!")
        return None
    
    if len(dataframes_validos) == 1:
        print("Apenas um DataFrame válido, retornando ele mesmo")
        return dataframes_validos[0]
    
    # Verificar compatibilidade das colunas
    colunas1 = set(dataframes_validos[0].columns)
    colunas2 = set(dataframes_validos[1].columns)
    
    if colunas1 != colunas2:
        print("AVISO: Colunas diferentes entre os DataFrames")
        print(f"   Apenas em {nome1}: {colunas1 - colunas2}")
        print(f"   Apenas em {nome2}: {colunas2 - colunas1}")
        print("   Usando união das colunas (NaN onde não existe)")
    
    # Consolidar DataFrames
    df_consolidado = pd.concat(dataframes_validos, ignore_index=True, sort=False)
    print(f"Consolidação concluída: {len(df_consolidado):,} registros totais")
    
    return df_consolidado

def salvar_planilha(df, nome_arquivo='pessoas_2020-2025.csv', sep=';', encoding='latin-1'):
    """
    Salva o DataFrame em um arquivo CSV.
    
    Args:
        df (pd.DataFrame): DataFrame a ser salvo
        nome_arquivo (str): Nome do arquivo de saída
        sep (str): Separador a ser usado no CSV
        encoding (str): Codificação do arquivo
        
    Returns:
        bool: True se salvou com sucesso, False caso contrário
    """
    if df is None or df.empty:
        print("ERRO: DataFrame vazio, não é possível salvar!")
        return False
    
    try:
        print(f"\nSalvando dados em {nome_arquivo}...")
        df.to_csv(nome_arquivo, sep=sep, encoding=encoding, index=False)
        
        # Verificar se o arquivo foi criado
        if os.path.exists(nome_arquivo):
            tamanho_mb = os.path.getsize(nome_arquivo) / (1024 * 1024)
            print(f"Arquivo salvo com sucesso!")
            print(f"{nome_arquivo}")
            print(f"   {len(df):,} registros")
            print(f"   {len(df.columns)} colunas") 
            print(f"   {tamanho_mb:.2f} MB")
            print(f"   {os.path.abspath(nome_arquivo)}")
            return True
        else:
            print(f"ERRO: Arquivo {nome_arquivo} não foi criado!")
            return False
            
    except Exception as e:
        print(f"ERRO ao salvar {nome_arquivo}: {e}")
        return False

def mostrar_resumo_final(df, titulo="Resumo Final - pessoas_2020-2025.csv"):
    """
    Mostra um resumo completo do DataFrame final.
    
    Args:
        df (pd.DataFrame): DataFrame a ser analisado
        titulo (str): Título do resumo
    """
    if df is None or df.empty:
        print(f"{titulo}: DataFrame vazio")
        return
    
    print(f"\n{titulo}")
    print("=" * 60)
    print(f"Total de registros: {len(df):,}")
    print(f"Total de colunas: {len(df.columns)}")
    
    # Anos consolidados
    if 'ano_sinistro' in df.columns:
        print(f"\nDISTRIBUIÇÃO POR ANO (ano_sinistro):")
        anos_final = df['ano_sinistro'].value_counts().sort_index()
        total_anos = 0
        for ano, count in anos_final.items():
            print(f"   {int(ano)}: {count:,} registros ({count/len(df)*100:.1f}%)")
            total_anos += count
        print(f"   TOTAL: {total_anos:,} registros")
    
    # Outras estatísticas importantes
    colunas_interesse = ['gravidade_lesao', 'sexo', 'tipo_veiculo_vitima', 'regiao_administrativa']
    
    for col in colunas_interesse:
        if col in df.columns:
            print(f"\nTOP 5 - {col.upper()}:")
            top_values = df[col].value_counts().head(5)
            for valor, count in top_values.items():
                print(f"   {valor}: {count:,} ({count/len(df)*100:.1f}%)")

def main():
    """Função principal da aplicação."""
    print("GERADOR DE DADOS 2020-2025")
    print("=" * 50)
    print("Consolidando acidentes de trânsito para 2020-2025")
    print("Fonte: pessoas_2015-2021.csv + pessoas_2022-2025.csv")
    print("Filtro: coluna 'ano_sinistro' entre 2020-2025")
    print("Destino: pessoas_2020-2025.csv\n")
    
    # Definir arquivos e parâmetros
    arquivo1 = "pessoas_2015-2021.csv"
    arquivo2 = "pessoas_2022-2025.csv" 
    arquivo_saida = "pessoas_2020-2025.csv"
    anos_desejados = range(2020, 2026)  # 2020, 2021, 2022, 2023, 2024, 2025
    
    # 1. Carregar as planilhas
    print("CARREGAMENTO DAS PLANILHAS")
    print("-" * 40)
    df1 = carregar_planilha(arquivo1)
    df2 = carregar_planilha(arquivo2)
    
    # Verificar se pelo menos uma planilha foi carregada
    if df1 is None and df2 is None:
        print("ERRO CRÍTICO: Nenhuma planilha pôde ser carregada!")
        return 1
    
    # 2. Filtrar por anos (2020-2025)
    print(f"\nFILTRAGEM POR ANOS ({anos_desejados.start}-{anos_desejados.stop-1})")
    print("-" * 40)
    df1_filtrado = filtrar_por_anos(df1, arquivo1, anos_desejados=anos_desejados)
    df2_filtrado = filtrar_por_anos(df2, arquivo2, anos_desejados=anos_desejados)
    
    # 3. Consolidar dados filtrados
    print(f"\nCONSOLIDAÇÃO DOS DADOS FILTRADOS")
    print("-" * 40)
    df_final = consolidar_dados(df1_filtrado, df2_filtrado, arquivo1, arquivo2)
    
    if df_final is None or df_final.empty:
        print("ERRO: Não foi possível consolidar os dados!")
        return 1
    
    # 4. Salvar arquivo consolidado
    print(f"\nSALVAMENTO DO ARQUIVO FINAL")
    print("-" * 40)
    sucesso = salvar_planilha(df_final, arquivo_saida)
    
    if not sucesso:
        print("Falha ao salvar o arquivo final!")
        return 1
    
    # 5. Relatório final detalhado
    mostrar_resumo_final(df_final)
    
    # 6. Mensagem de sucesso
    print(f"\nPROCESSO CONCLUÍDO COM SUCESSO!")
    print("=" * 50)
    print(f"Arquivo criado: {arquivo_saida}")
    print(f"Período consolidado: 2020-2025")
    print(f"Total de registros: {len(df_final):,}")
    print(f"Localização: {os.path.abspath(arquivo_saida)}")
    print("\nPronto para uso em análises de Machine Learning!")
    
    return 0

if __name__ == "__main__":
    try:
        print(f"Início: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        exit_code = main()
        print(f"Fim: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        exit(exit_code)
    except KeyboardInterrupt:
        print("\nProcesso interrompido pelo usuário!")
        exit(1)
    except Exception as e:
        print(f"\nERRO INESPERADO: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
