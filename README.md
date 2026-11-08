# Mineração de Dados: Previsão de Inadimplência em Cartões de Crédito (Base UCI 350)
Repositório oficial do Trabalho Prático da disciplina de **Mineração de Dados (2026)** da **Universidade Federal do Paraná (UFPR)**, ministrada pela **Profª Drª Denise Fukumi Tsunoda**.

* **Equipe 11:** Guilherme Santos Paz e Ana Melina
* **Base de Dados:** Nº 22 — *Default of Credit Card Clients* (ID UCI: 350)
* **Categorias de Métodos:** *Functions* (`Logistic`) e *Rules* (`JRip`), comparados ao baseline `ZeroR`

---

## 1. Sobre o Projeto

O objetivo deste projeto é aplicar todas as etapas do processo de **Descoberta de Conhecimento em Bases de Dados (KDD)** para prever a probabilidade de inadimplência (*default*) de clientes de cartão de crédito no mês subsequente, com base em variáveis demográficas, limite de crédito e histórico de faturas e pagamentos de seis meses (abril a setembro de 2005) em Taiwan.

A **Base Nº 22** possui **Índice de Complexidade 4** (o mais elevado da especificação), reunindo quatro fatores críticos tratados neste repositório:
* **Fator M:** Atributos numéricos contínuos e qualitativos nominais misturados;
* **Fator D:** Mais de $20$ atributos preditores ($23$ preditores + $1$ atributo-classe);
* **Fator P:** Ajuste estrutural obrigatório (remoção do identificador `ID` e tratamento de códigos não documentados em `EDUCATION`, `MARRIAGE` e `PAY_0` a `PAY_6`);
* **Fator G:** Grande volume de dados ($30.000$ instâncias originais), tratado via amostragem estratificada reprodutível (`seed = 42`).

---

## 2. Estrutura do Repositório

~~~text
├── default of credit card clients.xls       # Base de dados original (UCI ID 350 - 30.000 instâncias)
├── Versao1_Python_Para_Weka_2.ipynb         # Notebook principal: KDD, limpeza, discretização e exportação (.arff e .xlsx)
├── Versao2_Full_Python_Sklearn.ipynb        # Notebook complementar: Pipeline 100% Python (Scikit-Learn + Wittgenstein/RIPPER)
├── requirements.txt                         # Dependências do ambiente virtual Python
├── Relatorio_Completo_Equipe11.txt          # Documentação teórica, etapas do KDD e análise detalhada dos resultados
├── README.md                                # Documentação do repositório
└── results/                                 # Artefatos gerados e saídas experimentais do Weka
    ├── base22_execucao1_continua.arff       # Base padronizada para a Execução 1 (14 contínuos + 10 nominais)
    ├── base22_execucao2_discretizada.arff   # Base padronizada para a Execução 2 (24 atributos nominais/discretizados)
    ├── Discretizacao_Execucao2_Excel.xlsx   # Planilha com os pontos de corte, base contínua e base discretizada
    ├── resultadoZeroR1.txt                  # Saída do Weka: Baseline ZeroR (10-fold CV)
    ├── resultadoLogistic1.txt               # Saída do Weka: Logistic - Execução 1 (Contínua)
    ├── resultadoLogistic2.txt               # Saída do Weka: Logistic - Execução 2 (Discretizada)
    ├── resultadoJRIP1.txt                   # Saída do Weka: JRip - Execução 1 (Contínua)
    └── resultadoJRIP2.txt                   # Saída do Weka: JRip - Execução 2 (Discretizada)
~~~

---

## 3. Pipeline KDD e Pré-processamento Realizado

Todo o tratamento foi construído de forma linear e auditável em Python (`Versao1_Python_Para_Weka_2.ipynb`):

1. **Remoção de Identificadores (Fator P):** Exclusão da coluna sequencial `ID` para evitar sobreajuste (*overfitting*) e renomeação do atributo-alvo para `CLASSE_INADIMPLENCIA`.
2. **Limpeza de Inconsistências de Domínio:**
   * `EDUCATION`: Os códigos não documentados `0` ($14$ casos), `5` ($280$ casos) e `6` ($51$ casos) foram consolidados na categoria residual `4` (`Outros`), que passou de $123$ para $468$ registros.
   * `MARRIAGE`: Os $54$ registros com código não documentado `0` foram agrupados na categoria `3` (`Outros`).
   * `PAY_0` a `PAY_6`: Os códigos não documentados `-2` (sem consumo no ciclo) e `-1` (pagamento integral em dia) foram unificados com `0` (pagamento mínimo de rotativo em dia) sob o rótulo nominal `Em_Dia` (ausência de mora ativa), mantendo os atrasos reais de `Atraso_1mes` até `Atraso_9mes_ou_mais`.
3. **Transformação de Escala (Fator M):** Conversão explícita dos códigos inteiros de variáveis qualitativas para strings nominais, impedindo que o Weka assumisse relações lineares ou distâncias métricas artificiais entre categorias.
4. **Amostragem Estratificada (Fator G):** Redução controlada de $30.000$ para $5.000$ instâncias (`random_state = 42`), preservando rigorosamente a distribuição original das classes: **$77,88\%$ `Adimplente` ($3.894$ clientes)** e **$22,12\%$ `Inadimplente` ($1.106$ clientes)**.
5. **Critérios de Discretização para a Execução 2 (Planilha Excel):**
   * `AGE` (*Hierarquia Conceitual Demográfica*): `Jovem_ate_29` ($\le 29$), `Adulto_30_45` ($30\text{--}45$), `Meia_Idade_46_60` ($46\text{--}60$) e `Senior_acima_60` ($> 60$).
   * `LIMIT_BAL` (*Frequência Igual / Quartis Estatísticos* $Q_1\text{--}Q_4$): Pontos de corte em $50.000$, $140.000$ e $240.000\text{ NT\$}$ ($25\%$ da amostra por faixa).
   * `BILL_AMT1..6` (*Particionamento Contábil-Quantílico*): Isolamento de saldo credor/zerado em `Sem_Saldo_ou_Negativo` ($\le 0\text{ NT\$}$) e faixas positivas em `Fatura_Baixa_ate_10k` ($1\text{--}10.000$), `Fatura_Media_10k_50k` ($10.001\text{--}50.000$) e `Fatura_Alta_acima_50k` ($> 50.000\text{ NT\$}$).
   * `PAY_AMT1..6` (*Particionamento Contábil-Quantílico*): Isolamento da ausência de pagamento em `Sem_Pagamento_0` ($= 0\text{ NT\$}$) e desembolsos positivos em `Pgto_Baixo_ate_2k` ($1\text{--}2.000$), `Pgto_Medio_2k_10k` ($2.001\text{--}10.000$) e `Pgto_Alto_acima_10k` ($> 10.000\text{ NT\$}$).

---

## 4. Resultados Experimentais no Weka

Todos os modelos foram avaliados sob **Validação Cruzada Estratificada de 10 Partições (*10-fold stratified cross-validation*)** sobre as $5.000$ instâncias.

### Tabela Comparativa de Desempenho

| Métrica de Avaliação | `ZeroR` (Baseline) | `JRip` — Exec. 1 (Contínua) | `JRip` — Exec. 2 (Discretizada) | `Logistic` — Exec. 1 (Contínua) | `Logistic` — Exec. 2 (Discretizada) |

### Principais Conclusões

1. **Superação Consistente do Baseline (`ZeroR`):** O `ZeroR` acerta $77,88\%$ apenas prevendo a classe majoritária (`Adimplente`), resultando em $\text{Kappa} = 0$ e $\text{Recall} = 0$ para inadimplentes. Tanto o `Logistic` quanto o `JRip` elevaram o Kappa para $\approx 0,36$ e alcançaram precisão de até $66,8\%$ na detecção de inadimplentes.
2. **Ganho Empírico com a Discretização (Execução 2 > Execução 1):**
   * No **`JRip`**, a discretização eliminou cortes numéricos ruidosos, reduziu o tempo de indução em **$84\%$** (de $0,31\text{ s}$ para $0,05\text{ s}$), elevou a acurácia para **$81,38\%$** e alcançou a **maior revocação de inadimplentes do estudo ($35,6\%$, com $394$ calotes detectados)**.
   * No **`Logistic`**, a discretização corrigiu a compressão de escala dos valores em Novos Dólares Taiwaneses (cujos coeficientes contínuos apareciam arredondados para `0` na 4ª casa decimal na Execução 1) e permitiu capturar efeitos não lineares por faixa de limite e pagamento, atingindo a **maior acurácia global ($81,76\%$)**, **maior Kappa ($0,3615$)** e **maior AUC-ROC ($0,752$)**.

---

## 5. Como Reproduzir o Projeto

### Pré-requisitos e Instalação (Python)

1. Clone este repositório:
~~~bash
git clone https://github.com/SEU-USUARIO/NOME-DO-REPOSITORIO.git
cd NOME-DO-REPOSITORIO
~~~

2. Crie um ambiente virtual e instale as dependências listadas no `requirements.txt`:
~~~bash
python -m venv .venv
# No Windows:
.venv\Scripts\activate
