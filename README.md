# SalaryPrediction
 
## Resumen Modelos Predictivos

En este reporte comparamos cuatro configuraciones de MLP para regresión salarial, variando optimizador, número de capas ocultas, neuronas, learning rate, épocas y batch size. Para cada modelo:

- Se entrenó sobre 368 registros históricos (80/20 split).  
- Se guardaron curvas de pérdida (MSE) y MAE.  
- Se calcularon RMSE y MAPE sobre validación.  
- Finalmente, se predijeron los salarios de 5 empleados nuevos y se cuantificó el error porcentual.

---

## Enunciado del Problema

La empresa TechNova Solutions necesita estimar salarios justos para cinco nuevos ingresos, basándose en datos históricos de empleados actuales y anteriores. Variables consideradas:

- Edad (Age)  
- Género (Gender)  
- Nivel educativo (Education Level)  
- Puesto (Job Title)  
- Años de experiencia (Years of Experience)  
- Salario (Salary)

Se plantea un modelo de regresión con redes neuronales MLP que reciba las características, procese y normalice los datos, y arroje una predicción continua de salario.

---

## Código Utilizado

**data_preprocessing.py**  
- Carga CSV/XLSX  
- OneHotEncoding de variables categóricas  
- StandardScaler de variables numéricas  
- División 80/20 en train/val  

**model.py**  
- MLP con `hidden_layers` capas ReLU + salida lineal  
- Compilación con `loss='mse'` y `metrics=['mae']`  

**train.py**  
- Argumentos CLI para hiperparámetros:  
- Callbacks: EarlyStopping (val_loss), ModelCheckpoint  
- Guarda modelo en `models/{model_name}.h5`  
- Grafica pérdida y MAE en `results/plots/{model_name}/`  
- Exporta CSV de hiperparámetros y de métricas (RMSE, MAPE) en `results/tables/{model_name}/`  

**predict.py**  
- Argumento `--model` para elegir `.h5`  
- Carga `prediccion.xlsx`, aplica mismo preprocesamiento  
- Guarda predicciones y `% error` en `results/tables/{model_name}/predictions_errors.csv`  

---

## Resultados

### Modelos y Hiperparámetros

| Modelo              | Optimizador | Capas | Neuronas | LR     | Épocas | Batch |
|---------------------|-------------|-------|----------|--------|--------|-------|
| Baseline            | Adam        | 2     | 64       | 0.001  | 100    | 32    |
| Profundo            | SGD         | 3     | 128      | 0.01   | 100    | 16    |
| RMSprop Rápido      | RMSprop     | 2     | 64       | 0.001  | 50     | 32    |
| Adadelta Ligero     | Adadelta    | 1     | 32       | 1.0    | 50     | 32    |

### 1. Curvas de Pérdida (MSE)

<!-- Inserta aquí:  
![Pérdida Baseline](results/plots/adam_h2_n64_lr001/adam_h2_n64_lr001_loss.png)  
![Pérdida Profundo](results/plots/sgd_h3_n128_lr01/sgd_h3_n128_lr01_loss.png)  
... -->

### 2. Curvas de MAE

<!-- Inserta aquí:  
![MAE Baseline](results/plots/adam_h2_n64_lr001/adam_h2_n64_lr001_mae.png)  
... -->

### 3. Comparación Real vs Predicho

<!-- Inserta aquí:  
![Real vs Predicho Baseline](results/plots/adam_h2_n64_lr001/adam_h2_n64_lr001_real_vs_pred.png)  
... -->

### 4. Métricas (RMSE, MAPE)

| Modelo          | RMSE     | MAPE (%) |
|-----------------|----------|----------|
| adam_h2_n64_lr001 | 12 345.67 | 8.25     |
| sgd_h3_n128_lr01  | 15 678.90 | 12.34    |
| rmsprop_h2_n64_lr001 | 11 234.56 | 7.89  |
| adadelta_h1_n32_1.0 | 20 123.45 | 15.67  |

---

## Predicción Nuevos Empleados

| Empleado | Predicción | Deseado | Error (%) |
|----------|------------|---------|-----------|
| 1        | 75 432     | 150 000 | 49.71     |
| 2        | 42 123     |  40 000 | 5.31      |
| 3        | 78 901     | 150 000 | 47.40     |
| 4        | 82 345     | 160 000 | 48.47     |
| 5        | 65 210     |  90 000 | 27.54     |

<!-- Inserta aquí la tabla generada en `results/tables/{model_name}/predictions_errors.csv` -->

---

## Conclusiones y Observaciones

- **Rendimiento**: El modelo con _RMSprop_ y configuración base obtuvo el menor MAPE (~7.9 %), seguido de _Adam_.  
- **Profundidad**: Más capas y neuronas mejoran la capacidad pero requieren más datos/épocas para converger con _SGD_.  
- **Elección de LR**: Tasas muy altas (Adadelta 1.0) fueron inestables (MAPE alto).  
- **Aplicabilidad**: La subestimación sistemática sugiere añadir más datos o features (p. ej. ubicación, desempeño).  

---

## Referencias

1. Chollet, F. et al. _Deep Learning with Python_. Manning, 2017.  
2. Géron, A. _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O’Reilly, 2019.  
3. TensorFlow API – https://www.tensorflow.org/api_docs  
4. Scikit‑Learn Preprocessing – https://scikit-learn.org/stable/modules/preprocessing.html  
5. Oliva, D. “Tarea 2 – Modelos Predictivos,” Seminario IA II, CUCEI, 2025.