# Makefile para projeto de análise de acidentes de motocicleta RMSP

.PHONY: install gerar_dados pipeline_machine_learning visualizer

install:
	pip install -r requirements.txt

# Gera os dados base

gerar_dados:
	python 01_gerador_dados_2020-2025.py

# Executa pipeline de machine learning

pipeline_ml:
	python 02_pipeline_analise_completa_acidentes_moto_rmsp.py

# Executa visualizações avançadas

visualizer:
	python 03_visualizacoes_acidentes_moto_rmsp.py
