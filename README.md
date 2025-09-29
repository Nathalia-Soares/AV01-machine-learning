<div align="center">

# AV1 - MACHINE LEARNING
## Análise de Acidentes de Motocicleta na RMSP

</div>

Este repositório contém um pipeline completo de análise de dados e machine learning aplicado ao conjunto de dados de acidentes de motocicleta na Região Metropolitana de São Paulo (RMSP) entre 2020 e 2025. O objetivo é explorar os dados, realizar pré-processamento, treinar modelos, avaliar desempenho e gerar conclusões.

Trabalho desenvolvido para a disciplina Aprendizagem de Máquina, do curso de Desenvolvimento de Software Multiplataforma da Fatec Cotia.

---

# Índice

1. [Estrutura do Repositório](#estrutura-do-repositório)
2. [Requisitos](#requisitos)
3. [Instalação e Execução](#instalação-e-execução)
4. [Observações](#observações)
5. [Referência](#referência)

---

### Estrutura do Repositório

- `01_gerador_dados_2020-2025.py`: Gera dados de 2020 a 2025 a partir dos arquivos `pessoas_2015-2021.csv` e `pessoas_2022-2025.csv`.
- `02_pipeline_analise_completa_acidentes_moto_rmsp.py`: Pipeline principal de análise e machine learning.
- `03_visualizacoes_acidentes_moto_rmsp.py`: Gera visualizações avançadas dos dados e resultados.
- `output/`: Diretório para resultados, gráficos e tabelas gerados.
- `Makefile`: Facilita a execução do pipeline com comandos simples.
- `requirements.txt`: Lista de dependências do projeto.

---

### Requisitos

- Python >= 3.11
- Bibliotecas Python:
   - pandas
   - numpy
   - matplotlib
   - seaborn
   - scikit-learn
   - imbalanced-learn
   - scipy
- Base de dados: arquivos CSV `pessoas_2015-2021.csv` e `pessoas_2022-2025.csv` disponíveis em:  
  https://infosiga.detran.sp.gov.br/rest/painel/download/file/dados_infosiga.zip

---

### Instalação e Execução

1. Clone este repositório:
   ```bash
   git clone https://github.com/Nathalia-Soares/AV01-machine-learning.git
   cd AV01-machine-learning
   ```

2. Instale as dependências:
   ```bash
   make install
   ```

3. Gere os dados consolidados:
   ```bash
   make gerar_dados
   ```

4. Execute o pipeline de machine learning:
   ```bash
   make pipeline_machine_learning
   ```

5. Gere as visualizações avançadas:
   ```bash
   make visualizer
   ```

---

### Observações

- Certifique-se de que as pastas de saída (`output/2_pipeline_ml`, `output/03_visualizacoes`) existem ou serão criadas automaticamente.
- Os arquivos CSV de dados devem estar no diretório raiz do projeto.
- O Makefile facilita a execução dos scripts, mas você pode rodar cada script manualmente com `python <nome_do_script.py>` se preferir.

---

### Referência

Base de dados original: [Infosiga DETRAN-SP](https://infosiga.detran.sp.gov.br/rest/painel/download/file/dados_infosiga.zip)


