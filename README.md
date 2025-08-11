# 📊 Telecom X – Parte 2: Prevendo Churn de Clientes

## 🎯 Propósito da Análise
Este projeto tem como objetivo principal **prever a probabilidade de churn (evasão) de clientes** da Telecom X, utilizando dados históricos e variáveis relevantes sobre perfil, serviços contratados, formas de pagamento e comportamento de consumo.  

A análise visa:
- **Identificar fatores determinantes** para o cancelamento dos serviços.
- **Treinar modelos preditivos** para classificar clientes com maior risco de churn.
- **Gerar insights estratégicos** para ações de retenção.

---

## 📂 Estrutura do Projeto
```
TelecomX_Parte2/
│
├── telecomx_parte2.py             # Script principal com pipeline completo de ML
├── dados_tratados.csv             # Base de dados já tratada (opcional)
├── README.md                      # Documentação do projeto
│
├── visualizacoes/                 # Pasta opcional para salvar gráficos
│   ├── correlacao_churn.png
│   ├── mutual_information.png
│   └── importancia_variaveis.png
│
└── artifacts_telecomx/            # Artefatos do modelo treinado
    ├── best_model_log_reg.joblib  # Modelo vencedor salvo
    └── best_threshold.txt         # Threshold ótimo encontrado
```

---

## 🔄 Processo de Preparação dos Dados

### 1. **Classificação das Variáveis**
- **Numéricas**: `tenure`, `Charges_Monthly`, `Charges_Total`, `SeniorCitizen`
- **Categóricas**: `Contract`, `InternetService`, `PaymentMethod`, `PaperlessBilling`, `PhoneService`, entre outras derivadas do JSON original.

### 2. **Tratamento e Limpeza**
- Flatten do JSON original para tabela única.
- Conversão de variáveis de cobrança para numérico (`Charges_Monthly` e `Charges_Total`).
- Conversão de `SeniorCitizen` para inteiro (`0` ou `1`).
- Remoção de linhas sem valor em `Charges_Total`.
- Criação da variável binária `churn_flag` (`1` para "Yes", `0` para "No").

### 3. **Codificação e Normalização**
- **Numéricas**: imputação de valores ausentes (mediana) + `StandardScaler`.
- **Categóricas**: imputação de valores ausentes (moda) + `OneHotEncoder` com `handle_unknown="ignore"`.

### 4. **Separação em Treino e Teste**
- **Treino**: 80% dos dados  
- **Teste**: 20% dos dados  
- Divisão **estratificada** para manter proporção de churn.

---

## 🧠 Modelagem e Justificativas

### **Modelos testados**
1. **Logistic Regression** (`log_reg`)
   - Interpretação simples dos coeficientes.
   - Bom baseline para classificação binária.
2. **Random Forest** (`rf`)
   - Modelo de árvore mais robusto a variáveis categóricas após OHE.
   - Capaz de capturar relações não lineares.

### **Seleção de Variáveis**
- **Mutual Information**: ranqueamento inicial das variáveis mais relevantes.
- Redução para **top 25** variáveis para diminuir dimensionalidade e ruído.

### **Métrica Principal**
- **ROC AUC** (capacidade de discriminação entre churn e não churn).
- **F1-score** analisado para balancear precisão e recall.

### **Resultado**
- **Melhor modelo**: `Logistic Regression`  
  - ROC AUC ≈ **0,820**  
  - Recall ≈ **0,704**  
  - Principais variáveis: `Charges_Total`, `tenure`, `Contract_Month-to-month`, `Charges_Monthly`, `InternetService_Fiber optic`, `PaymentMethod_Electronic check`.

---

## 📈 EDA – Análise Exploratória de Dados

Durante a EDA, foram gerados gráficos como:

1. **Correlação com Churn (variáveis numéricas)**
   - `tenure` com forte correlação negativa → clientes com pouco tempo de contrato têm maior chance de churn.

2. **Mutual Information – Top Variáveis**
   - `Contract_Month-to-month`, `InternetService_Fiber optic` e `PaymentMethod_Electronic check` entre os principais drivers.

3. **Importância das Variáveis (Permutation Importance)**
   - Confirma impacto de variáveis financeiras e tipo de contrato.

**Exemplo de gráfico de correlação:**

```plaintext
tenure           -0.351
Charges_Total    -0.197
Charges_Monthly  -0.156
SeniorCitizen     0.156
```

**Exemplo de insight:**  
> Clientes com contrato **mês-a-mês**, **internet de fibra** e **pagamento via electronic check** têm probabilidade significativamente maior de cancelar o serviço.

---

## 🚀 Próximos Passos
- Testar novos algoritmos (Gradient Boosting, XGBoost).
- Calibrar probabilidades para melhorar definição de threshold.
- Implementar rotina de scoring contínua e integração com CRM para disparo de campanhas de retenção.
