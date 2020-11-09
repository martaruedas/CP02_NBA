
#----------------------------------------------------------------------------------------------------
# REGRESIÓN Y SALARIOS NBA
# PREDICCIÓN (2020/2021)
# Marta Ruedas Burgos
# 7-11-2020
#----------------------------------------------------------------------------------------------------

#### NBA-Predicción ####

#----------------------------------------------------------------------------------------------------
# CP02: SALARIOS NBA (CV Y REGULACIÓN)
# Entregar un informe en formato html y dejar un enlace de acceso aL GitHub con el código.
# OBJETIVO: determinar el mejor modelo para predecir el salario de los jugadores de la NBA.
#----------------------------------------------------------------------------------------------------

# Estudiamos el modelo de predicción y simulación
# 1. Regresión lineal
# 2. Regresión no lineal
# 3. Series Temporales

# LIBRARIES AND FUNCTIONS

library(readr)
library(car) # Normalidad nba
library(tidyr) # Modelo lineal
library(tidyverse) # Modelo lineal
library(here) # Comentar
library(janitor) # Nombres limpios
library(skimr) # Vistosos Summarizes
library(magrittr) # Pipe operaciones
library(corrplot) # Correlaciones
library(ggcorrplot)  # Correlaciones
library(PerformanceAnalytics) # Correlaciones
library(leaps) # Modelo de seleccion
library(rsample)  # data splitting 
library(glmnet)   # implementing regularized regression approaches
library(dplyr)    # basic data manipulation procedures
library(ggplot2)  # plotting

# READ DATA

nba <- read_csv("nba.csv") 
View(nba)

#Tabla de la NBA compuesta por 485 jugadores y sus respectivas valoraciones.


# VARIABLES NAMES
nba %<>% clean_names()
colnames(nba)

#[1] "player"           "salary"           "nba_country"     
#[4] "nba_draft_number" "age"              "tm"              
#[7] "g"                "mp"               "per"             
#[10] "ts_percent"       "x3p_ar"           "f_tr"            
#[13] "orb_percent"      "drb_percent"      "trb_percent"     
#[16] "ast_percent"      "stl_percent"      "blk_percent"     
#[19] "tov_percent"      "usg_percent"      "ows"             
#[22] "dws"              "ws"               "ws_48"           
#[25] "obpm"             "dbpm"             "bpm"             
#[28] "vorp"  


# SUMMARIZE DATA

skim(nba)

# Nombre nba, columnas 28, filas 485.
# Tipos de columnas numeric (25) - character (3) PLAYER, NBA_COUNTRY AND TM.

# Al haber datos repetidos y varios NA usamos el distinct y el drop a continuación.

# DATA WRANGLING DATA

# Eliminar duplicados
# Eliminar todas las filas que estén duplicadas
nba %<>% distinct(player,.keep_all= TRUE)

# Eliminar NAs
nba %<>% drop_na()

# Summarise/Resumen
skim(nba)

#----------------------------------------------------------------------------------------------------
# EDA
#----------------------------------------------------------------------------------------------------

# LOG SALARY

# Pasa el salario a logaritmo 
log_nba <- nba %>% mutate(salary=log(salary))

skim(log_nba)

# ahora mi tabla se llama log_nba


#tabla de tres variables

vars <- c("player","nba_country","tm")

# Correlations

ggcorrplot(cor(log_nba %>% 
                 select_at(vars(-vars)), 
               use = "complete.obs"),
           hc.order = TRUE,
           type = "lower",  lab = TRUE)
# cuanto mayor sea el número mayor correlación. Mas cercano al 1 mayor correlación habrá entre las dos variables. 


#----------------------------------------------------------------------------------------------------
# VIF
#----------------------------------------------------------------------------------------------------

# Mide cuanto aumenta la varianza de un coeficiente de regresión estimado. Si sus predictores estan correlacionados. 
# Detecta problemas de multicolinealidad, que hace que nuestros modelos sean menos precisos.

nba_vif <- lm(salary~.-player-nba_country-tm, data=log_nba)

vif_values_nba <- car::vif(nba_vif)

#create horizontal bar chart to display each VIF value
barplot(vif_values_nba, main = "VIF Values NBA", horiz = TRUE, col = "darkorange")

#add vertical line at 5
abline(v = 9, lwd = 6, lty = 4)

# Tabla con los valores
knitr::kable(vif_values_nba)

# Si el VIF es mayor que 6 no es muy bueno. 


#----------------------------------------------------------------------------------------------------
# MODEL SELECTION
#----------------------------------------------------------------------------------------------------

# Coge todas las columnas menos las tres categóricas
nba <- log_nba %>% select_at(vars(-vars))

# Si ejecutamos set.seed(1234) nunca varian, siempre nos daría el mismo resultado aleatorio.

set.seed(1234)
num_data <- nrow(nba) # coge el número de filas
num_data_test <- 10 # el número de la muestra, tamaño
train = sample(num_data ,num_data-num_data_test) # donde probamos los modelos 

# Comprobación
data_train <- nba[train,]
data_test  <-  nba[-train,]

# Modelo seleccionado
modelo_seleccionado <- regsubsets(salary~. , data =data_train, method = "seqrep",nvmax=24) 

# 24 modelos va probando
# Si tiene * es que ha metido esa variable dentro del modelo

modelo_seleccionado_summary <- summary(modelo_seleccionado)

data.frame(
  Adj.R2 = (modelo_seleccionado_summary$adjr2), # buscar el valor maximo
  CP = (modelo_seleccionado_summary$cp),
  BIC = (modelo_seleccionado_summary$bic) # buscar el valor minimo
)

# 3 columnas, tres criterios distintos para elegir un modelo. 


# PLOT

plot(modelo_seleccionado, scale = "bic", main = "BIC NBA")


# DATAFRAME 
# coge el modelo con el maximo del R2 ajustado el valor minimo del CP y el valor minimo del BIC
data.frame(
  Adj.R2 = which.max(modelo_seleccionado_summary$adjr2),
  CP = which.min(modelo_seleccionado_summary$cp),
  BIC = which.min(modelo_seleccionado_summary$bic)
)

# De los 24 modelos que nos salian anteriormente por ejemplo el Adj.R2 el 14 es el que tiene el maximo.


# MODELO PARA EL R2 AJUSTADO 

coef(modelo_seleccionado,which.min(modelo_seleccionado_summary$adjr2))

# De todas las variables el mp los minutos por partidos es la variable mas importante para determinar el salario

# MODELO PARA EL CP

coef(modelo_seleccionado,which.min(modelo_seleccionado_summary$cp))

# variables necesarias para explicar el modelo segun el criterio del cp

# MODELO PARA EL BIC 
coef(modelo_seleccionado,which.min(modelo_seleccionado_summary$bic))

# variables necesarias para explicar el modelo segun el criterio del bic


# Anotacion: todos los modelos son erroneos, algunos modelos podrían servir.

# SET SEED

# Create training (70%) and test (30%) sets for the AmesHousing::make_ames() data.
# Use set.seed for reproducibility

set.seed(1234)
data_train <- nba[train,]
data_test  <-  nba[-train,]

# Create training and testing feature model matrices and response vectors.
# we use model.matrix(...)[, -1] to discard the intercept
nba_train_x <- model.matrix(salary ~ ., data_train)[, -1]
nba_train_y <- log(data_train$salary)

nba_test_x <- model.matrix(salary ~ ., data_test)[, -1]
nba_test_y <- log(data_test$salary)

# What is the dimension of of your feature matrix?
dim(nba_train_x)

# 471 filas y 24 columnas

# ELASTIC NET

lasso    <- glmnet(nba_train_x, nba_train_y, alpha = 1.0) 
elastic1 <- glmnet(nba_train_x, nba_train_y, alpha = 0.25) 
elastic2 <- glmnet(nba_train_x, nba_train_y, alpha = 0.75) 
ridge    <- glmnet(nba_train_x, nba_train_y, alpha = 0.0)

par(mfrow = c(2, 2), mar = c(6, 4, 6, 2) + 0.1)
plot(lasso, xvar = "lambda", main = "Lasso (Alpha = 1)\n\n\n")
plot(elastic1, xvar = "lambda", main = "Elastic Net (Alpha = .25)\n\n\n")
plot(elastic2, xvar = "lambda", main = "Elastic Net (Alpha = .75)\n\n\n")
plot(ridge, xvar = "lambda", main = "Ridge (Alpha = 0)\n\n\n")

# TUNING 
# va a coger muchos alphas de 0 a 1 para los distintos modelos.  

# maintain the same folds across all models
fold_id <- sample(1:10, size = length(nba_train_y), replace=TRUE)

# search across a range of alphas
tuning_grid <- tibble::tibble(
  alpha      = seq(0, 1, by = .1),
  mse_min    = NA,
  mse_1se    = NA,
  lambda_min = NA,
  lambda_1se = NA
)
tuning_grid


for(i in seq_along(tuning_grid$alpha)) {
  
  # fit CV model for each alpha value
  fit <- cv.glmnet(nba_train_x, nba_train_y, alpha = tuning_grid$alpha[i], foldid = fold_id)
  
  # extract MSE and lambda values
  tuning_grid$mse_min[i]    <- fit$cvm[fit$lambda == fit$lambda.min]
  tuning_grid$mse_1se[i]    <- fit$cvm[fit$lambda == fit$lambda.1se]
  tuning_grid$lambda_min[i] <- fit$lambda.min
  tuning_grid$lambda_1se[i] <- fit$lambda.1se
}

tuning_grid

#mse_1se buscar el que menor valor tenga, será el mejor modelo al tener mayor precisión. 
# Con el mejor alpha asociado a ese modelo.

# Ir cambiando variables de nuestro modelo y comprobar cual de ellos es el que menos errores tiene. 

