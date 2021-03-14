# MINI PROJETO 2 - ANÁLISE DE RISCO DE CRÉDITO
# Projeto desenvolvido como exercício do curso Formação Cientista de Dados, da DataScience Academy
# https://www.datascienceacademy.com.br/bundles?bundle_id=formacao-cientista-de-dados
# 

setwd("D:/CIENTISTA_DADOS/1_BIG_DATA_R_AZURE/CAP18")
getwd()


library(readr)
library(rvest)
library(ggplot2)
library(psych)
library(corrplot)
library(corrgram)
library(tidyverse)
library(lattice)

#install.packages("DMwR")
library(DMwR)
#install.packages("gmodels")
library(gmodels)
library(randomForest)
library(caTools)
library(ROCR) # Curva ROC
library(caret) # Confusion matrix
library(neuralnet) # Gerar modelo de rede neural
library(e1071) # Naive Bayes
library(kernlab) # Supor Vector Machine

# Carrega arquivo com funções para transformação de dados
#source("src/ClassTools.R")
source("src/plot_utils.R")

# Carregando o dataset
dataframe1 <- read.csv("credit_dataset.csv")

# Visualizando o dataframe completo.
# Aparentemente a primeira coluna (credit.rating) é a coluna que indica se o crédito foi concedido ou não
head(dataframe1)

# Dimensões do dataframe
dim(dataframe1) 

# Verificando o tipo das colunas.
glimpse(dataframe1)
# Aparentemente todas as colunas foram consideradas numéricas


## Criando funções para converter variáveis categóricas para tipo fator.
to.factors <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- as.factor(df[[variable]])
  }
  return(df)
}

## Como as variáveis numéricas estão em escalas diferentes, deve-se fazer uma escala e normalização
scale.features <- function(df, variables){
  for (variable in variables){
    df[[variable]] <- scale(df[[variable]], center=T, scale=T)
  }
  return(df)
}

# Normalizando as variáveis
numeric.vars <- c("credit.duration.months", "age", "credit.amount")
dataframe2 <- scale.features(dataframe1, numeric.vars)

# Variáveis do tipo fator
categorical.vars <- c('credit.rating', 'account.balance', 'previous.credit.payment.status',
                      'credit.purpose', 'savings', 'employment.duration', 'installment.rate',
                      'marital.status', 'guarantor', 'residence.duration', 'current.assets',
                      'other.credits', 'apartment.type', 'bank.credits', 'occupation', 
                      'dependents', 'telephone', 'foreign.worker')

dataframe2 <- to.factors(df = dataframe2, variables = categorical.vars)

#glimpse(dataframe2)
head(dataframe2)

# Cria um dataframe apenas com as colunas numéricas do dataset original
dataframe3 <- dataframe1[,c(3,6,14)]
head(dataframe3)

# Medidas de tendência central das colunas numéricas
summary(dataframe3)

# Observando a correlação entre as colunas numéricas

cols <- c("credit.duration.months", "credit.amount", "age")

# Vetor com os métodos de correlação
metodos <- c("pearson", "spearman")

cors <- lapply(metodos, function(method)
  (cor(dataframe2[,cols], method = method)))

plot.cors <- function(x, labs){
  diag(x) <- 0.0
  plot(levelplot(x,
                 main = paste("Plot de Correlação Usando Método", labs),
                 scales = list(x = list(rot = 90), cex = 1.0)))
}

# Mapa de correlação
Map(plot.cors, cors, metodos)

# Verificando a quantidade de crédito bom e ruim
table(dataframe2$credit.rating)

# É possível verificar que existem muito mais casos com crédito bom do que com crédito ruim.
# Então é necessário balancear para que haja quantidades parecidas e o modelo não fique tendencioso.

# Passar a coluna alvo para a última posição do dataframe
dataframe4 <- dataframe2$credit.rating
dataframe2$credit.rating <- NULL
dataframe2 <- cbind(dataframe2, dataframe4)
dataframe2 <- dataframe2 %>%
  rename(
    credit.rating = dataframe4
  )

# Faz o balanceamento
dataframe2 <- SMOTE(credit.rating ~ ., data = dataframe2, perc.over = 100)
table(dataframe2$credit.rating)

# Criar gráfico de barras para observar as variáveis
nomes <- colnames(dataframe2)
nomes <- nomes[1:20]
lapply(nomes, function(x){
  if(is.factor(dataframe2[,x])){
    ggplot(dataframe2, aes_string(x)) + 
      geom_bar(color = "green4", fill = "lightgreen") + 
      facet_grid(. ~ credit.rating)+
      ggtitle(paste("Total de Crédito Bom/Ruim por", x))
  }
})

# Função para seleção de variáveis
run.feature.selection <- function(num.iters=20, feature.vars, class.var){
  set.seed(10)
  variable.sizes <- 1:10
  control <- rfeControl(functions = rfFuncs, method = "cv", 
                        verbose = FALSE, returnResamp = "all", 
                        number = num.iters)
  results.rfe <- rfe(x = feature.vars, y = class.var, 
                     sizes = variable.sizes, 
                     rfeControl = control)
  return(results.rfe)
}

# Executando a função
rfe.results <- run.feature.selection(feature.vars = dataframe2[,-21], 
                                     class.var = dataframe2[,21])
# Visualizando os resultados
rfe.results
varImp((rfe.results))

# Utilização do modelo random forest para criação de um plot de importância das variáveis preditoras
modelo <- randomForest(credit.rating ~ .,
                       data = dataframe2, 
                       ntree = 100, nodesize = 10, importance = T)

varImpPlot(modelo)

# A princípio será montado um modelo com todas as variáveis e outro apenas com as 6 variáveis 
# mais importantes indicadas pelo random forest (métodos Mean Decrease Accuracy e Mean Decrease Gini).
# Isso será feito para comparação entre os modelos e como forma de validação para a retirada de 
# algumas variáveis.

# O primeiro passo é dividir os dados em treino e teste, de forma aleatória.
# Essa divisão será 70% do dataset para dados de treino e 30% para dados de teste.


amostra <- sample.split(dataframe2$credit.rating, SplitRatio = 0.70)

# ***** Treinamos nosso modelo nos dados de treino *****
# *****   Fazemos as predições nos dados de teste  *****

# Criando dados de treino - 70% dos dados
treino = subset(dataframe2, amostra == TRUE)

# Criando dados de teste - 30% dos dados
teste = subset(dataframe2, amostra == FALSE)


### Modelos
# Regressão Logística
# randomForest
# Suport Vector Machine
# Naive Bayes

# Construindo um modelo de regressão logística com todas as variáveis
formula.init <- "credit.rating ~ ."
formula.init <- as.formula(formula.init)
modelo_RL_1 <- glm(formula = formula.init, data = treino, family = "binomial")

# Visualizando o modelo
summary(modelo_RL_1)

# Testando o modelo nos dados de teste
prevendo_RL_1 <- predict(modelo_RL_1, teste, type="response")
prevendo_RL_1 <- round(prevendo_RL_1)

test.feature.vars <- teste[,-21]
test.class.var <- teste[,21]

CF_1 <- confusionMatrix(table(data = prevendo_RL_1, reference = test.class.var), positive = '1')
CF_1$table
CF_1$overall["Accuracy"]

# Visualizando os valores previstos e observados
resultados_RL_1 <- cbind(prevendo_RL_1, teste$credit.rating) 
colnames(resultados_RL_1) <- c('Previsto','Real')
resultados_RL_1 <- as.data.frame(resultados_RL_1)
View(resultados_RL_1)

## Feature selection - Observando o gráfico com as variáveis mais importantes para os modelos.
formula <- "credit.rating ~ ."
formula <- as.formula(formula)
control <- trainControl(method = "repeatedcv", number = 10, repeats = 2)
model <- train(formula, data = treino, method = "glm", trControl = control)
print(model)
importance <- varImp(model, scale = FALSE)
importance
plot(importance)

# Pode se ver que as 6 varáveis mais importantes são:
# account.balance
# credit.purpose
# credit.amount
# previous.credit.payment.status
# savings
# current.assets

# Construindo o modelo com as variáveis selecionadas
formula.new <- "credit.rating ~ account.balance + credit.purpose + previous.credit.payment.status + savings + current.assets + credit.amount"
formula.new <- as.formula(formula.new)
modelo_RL_2 <- glm(formula = formula.new, data = treino, family = "binomial")

# Visualizando o modelo
summary(modelo_RL_2)

# Testando o modelo nos dados de teste
prevendo_RL_2 <- predict(modelo_RL_2, teste, type = "response") 
prevendo_RL_2 <- round(prevendo_RL_2)

# Avaliando o modelo
CF_2 <- confusionMatrix(table(data = prevendo_RL_2, reference = test.class.var), positive = '1')
CF_2$table
CF_2$overall["Accuracy"]


accuracyVector <- c(CF_1$overall["Accuracy"],CF_2$overall["Accuracy"])

#lr.prediction.values <- predict(lr.model.best, test.feature.vars, type = "response")
#View(lr.prediction.values)
#View(test.class.var)
#class(lr.prediction.values)
#class(test.class.var)
#lr.prediction.values <- as.data.frame(lr.prediction.values)
#test.class.var <- as.data.frame(test.class.var)
#predictions <- prediction(lr.prediction.values, test.class.var)

# Modelo Random Forest com todas as variáveis
modelo_RF_1 <- randomForest(credit.rating ~ .,
                       data = treino,
                       ntree = 100,
                       nodesize = 10)


print(modelo_RF_1)

# Visualizando o Modelo e Fazendo Previsões

# Plot do Modelo
plot(modelo_RF_1)

# Fazendo as previsões
prevendo_RF_1 <- predict(modelo_RF_1, newdata = teste)
View(prevendo_RF_1)

# Visualizando os valores previstos e observados
resultados_RF_1 <- cbind(prevendo_RF_1, teste$credit.rating) 
colnames(resultados_RF_1) <- c('Previsto','Real')
resultados_RF_1 <- as.data.frame(resultados_RF_1)
View(resultados_RF_1)


# Gerar confusion matrix com o Caret
resultados_RF_1$Previsto <- factor(resultados_RF_1$Previsto)
resultados_RF_1$Real <- factor(resultados_RF_1$Real)

CF_3 <- confusionMatrix(resultados_RF_1$Real, resultados_RF_1$Previsto)
CF_3$table
CF_3$overall["Accuracy"]
accuracyVector <- c(accuracyVector,CF_3$overall["Accuracy"])

##### Modelo Random Forest com seis variáveis
modelo_RF_2 <- randomForest(credit.rating ~ account.balance
                            + previous.credit.payment.status
                            + credit.duration.months
                            + credit.amount
                            + current.assets
                            + savings,
                            data = treino,
                            ntree = 100,
                            nodesize = 10)
print(modelo_RF_2)

# Plot do Modelo
plot(modelo_RF_2)

# Fazendo as previsões
prevendo_RF_2 <- predict(modelo_RF_2, newdata = teste)
View(prevendo_RF_2)

# Visualizando os valores previstos e observados
resultados_RF_2 <- cbind(prevendo_RF_2, teste$credit.rating) 
colnames(resultados_RF_2) <- c('Previsto','Real')
resultados_RF_2 <- as.data.frame(resultados_RF_2)
View(resultados_RF_2)


# Gerar confusion matrix com o Caret
resultados_RF_2$Previsto <- factor(resultados_RF_2$Previsto)
resultados_RF_2$Real <- factor(resultados_RF_2$Real)

CF_4 <- confusionMatrix(resultados_RF_2$Real, resultados_RF_2$Previsto)
CF_4$table
CF_4$overall["Accuracy"]


# Criando um dataframe com a acurácia de cada modelo

accuracyVector <- c(accuracyVector,CF_4$overall["Accuracy"])

#######  Modelo Naive Bayes com todas as variáveis
modelo_NB_1 <- naiveBayes(credit.rating ~ .,treino)
print(modelo_NB_1)
prevendo_NB_1 <- predict(modelo_NB_1, teste[,-21])
head(prevendo_NB_1)
table(prevendo_NB_1, true = teste$credit.rating)

# Visualizando os valores previstos e observados
resultados_NB_1 <- cbind(prevendo_NB_1, teste$credit.rating) 
colnames(resultados_NB_1) <- c('Previsto','Real')
resultados_NB_1 <- as.data.frame(resultados_NB_1)
View(resultados_NB_1)

# Gerar confusion matrix com o Caret
resultados_NB_1$Previsto <- factor(resultados_NB_1$Previsto)
resultados_NB_1$Real <- factor(resultados_NB_1$Real)

CF_5 <- confusionMatrix(resultados_NB_1$Real, resultados_NB_1$Previsto)
CF_5$table
CF_5$overall["Accuracy"]

accuracyVector <- c(accuracyVector, CF_5$overall["Accuracy"])

#######  Modelo Naive Bayes com 6 variáveis
modelo_NB_2 <- naiveBayes(credit.rating ~ account.balance
                          + previous.credit.payment.status
                          + credit.duration.months
                          + credit.amount
                          + current.assets
                          + savings,
                          treino)
print(modelo_NB_2)
prevendo_NB_2 <- predict(modelo_NB_2, teste[,-21])
head(prevendo_NB_2)
table(prevendo_NB_2, true = teste$credit.rating)

# Visualizando os valores previstos e observados
resultados_NB_2 <- cbind(prevendo_NB_2, teste$credit.rating) 
colnames(resultados_NB_2) <- c('Previsto','Real')
resultados_NB_2 <- as.data.frame(resultados_NB_2)
head(resultados_NB_2)

# Gerar confusion matrix com o Caret
resultados_NB_2$Previsto <- factor(resultados_NB_2$Previsto)
resultados_NB_2$Real <- factor(resultados_NB_2$Real)

CF_6 <- confusionMatrix(resultados_NB_2$Real, resultados_NB_2$Previsto)
CF_6$table
CF_6$overall["Accuracy"]

accuracyVector <- c(accuracyVector, CF_6$overall["Accuracy"])


##### Modelo Suport Vector Machine com todas as variáveis- SVM
modelo_SVM_1 <- ksvm(credit.rating ~ .,data = treino, kernel="vanilladot" )
#summary(modelo_SVM_1)

prevendo_SVM_1 <- predict(modelo_SVM_1, teste)
View(prevendo_SVM_1)

# Visualizando os valores previstos e observados
resultados_SVM_1 <- cbind(prevendo_SVM_1, teste$credit.rating) 
colnames(resultados_SVM_1) <- c('Previsto','Real')
resultados_SVM_1 <- as.data.frame(resultados_SVM_1)
head(resultados_SVM_1)

# Gerar confusion matrix com o Caret
resultados_SVM_1$Previsto <- factor(resultados_SVM_1$Previsto)
resultados_SVM_1$Real <- factor(resultados_SVM_1$Real)

CF_7 <- confusionMatrix(resultados_SVM_1$Real, resultados_SVM_1$Previsto)
CF_7$table
CF_7$overall["Accuracy"]

accuracyVector <- c(accuracyVector, CF_7$overall["Accuracy"])


##### Modelo Suport Vector Machine com 6 variáveis- SVM
modelo_SVM_2 <- ksvm(credit.rating ~ account.balance
                     + previous.credit.payment.status
                     + credit.duration.months
                     + credit.amount
                     + current.assets
                     + savings,
                     data = treino, kernel="vanilladot" )
#summary(modelo_SVM_2)

prevendo_SVM_2 <- predict(modelo_SVM_2, teste)
View(prevendo_SVM_2)

# Visualizando os valores previstos e observados
resultados_SVM_2 <- cbind(prevendo_SVM_2, teste$credit.rating) 
colnames(resultados_SVM_2) <- c('Previsto','Real')
resultados_SVM_2 <- as.data.frame(resultados_SVM_2)
head(resultados_SVM_2)

# Gerar confusion matrix com o Caret
resultados_SVM_2$Previsto <- factor(resultados_SVM_2$Previsto)
resultados_SVM_2$Real <- factor(resultados_SVM_2$Real)

CF_8 <- confusionMatrix(resultados_SVM_2$Real, resultados_SVM_2$Previsto)
CF_8$table
CF_8$overall["Accuracy"]

accuracyVector <- c(accuracyVector, CF_8$overall["Accuracy"])


# Criando um dataframe com todas as acurácias conseguidas nos 6 modelos testados.
Modelos <- c("Regressao Logistica Todas", "Regressao Logistica 6","RandomForest Todas", "RandomForest 6", "Naive Bayes Todas", 
             "Naive Bayes 6", "SVM Todas", "SVM 6")
accuracyDataFrame <- data.frame(Modelos, accuracyVector)
colnames(accuracyDataFrame) <- c("Modelos", "Acurácia")
View(accuracyDataFrame)

######### OTIMIZAÇÃO DO MODELO




















