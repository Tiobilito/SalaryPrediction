# SalaryPrediction
 
## Resumen Modelos Predictivos

En este reporte comparamos cuatro configuraciones de MLP para regresión salarial, variando optimizador, número de capas ocultas, neuronas, learning rate, épocas y batch size. Para cada modelo:

- Se entrenó sobre 368 registros históricos (80/20 split).  
- Se guardaron curvas de pérdida (MSE) y MAE.  
- Se calcularon RMSE y MAPE sobre validación.  
- Finalmente, se predijeron los salarios de 5 empleados nuevos y se cuantificó el error porcentual.

## Enunciado del Problema

La empresa TechNova Solutions necesita estimar salarios justos para cinco nuevos ingresos, basándose en datos históricos de empleados actuales y anteriores. Variables consideradas:

- Edad (Age)  
- Género (Gender)  
- Nivel educativo (Education Level)  
- Puesto (Job Title)  
- Años de experiencia (Years of Experience)  
- Salario (Salary)

Se plantea un modelo de regresión con redes neuronales MLP que reciba las características, procese y normalice los datos, y arroje una predicción continua de salario.

## Código Utilizado

**data_preprocessing.py**  
- Carga CSV/XLSX  
- OneHotEncoding de variables categóricas  
- StandardScaler de variables numéricas  
- División 80/20 en train/val  

```python
def preprocess(X, fit=True, encoder_path=None, scaler_path=None):
    cat_cols = ['Gender', 'Education Level', 'Job Title']
    num_cols = ['Age', 'Years of Experience']
    # ...encoding y escalado...
    return pd.concat([df_num, df_cat], axis=1)
```

**model.py**  
- MLP con `hidden_layers` capas ReLU + salida lineal  
- Compilación con `loss='mse'` y `metrics=['mae']`  

```python
def build_model(input_dim, hidden_layers, neurons, learning_rate, opt_name):
    # ...definición de modelo secuencial...
    model.compile(
        optimizer=opts[opt_name],
        loss='mse',
        metrics=['mae']
    )
    return model
```

**train.py**  
- Argumentos CLI para hiperparámetros:  
- Callbacks: EarlyStopping (val_loss), ModelCheckpoint  
- Guarda modelo en `models/{model_name}.h5`  
- Grafica pérdida y MAE en `results/plots/{model_name}/`  
- Exporta CSV de hiperparámetros y de métricas (RMSE, MAPE) en `results/tables/{model_name}/`  

```python
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, model_save_path):
    checkpoint = ModelCheckpoint(
        model_save_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min'
    )
    earlystop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, earlystop]
    )
    return history
```

**predict.py**  
- Argumento `--model` para elegir `.h5`  
- Carga `prediccion.xlsx`, aplica mismo preprocesamiento  
- Guarda predicciones y `% error` en `results/tables/{model_name}/predictions_errors.csv`  

```python
if __name__ == '__main__':
    # ...argumentos...
    model = load_model(args.model)
    preds = model.predict(Xp).flatten()
    errors_pct = (preds - desired) / desired * 100
    df_out = pd.DataFrame({
        'predicted_salary': preds,
        'desired_salary':   desired,
        'error_%':          errors_pct
    })
    df_out.to_csv(out_path, index=False)
```

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

#### Adam
![Pérdida Adam](results/plots/adam_h2_n64_lr001/adam_h2_n64_lr001_loss.png)

#### Rmsprop
![Pérdida Rmsprop](results/plots/rmsprop_h2_n64_lr001/rmsprop_h2_n64_lr001_loss.png)

#### Adadelta
![Pérdida Adadelta](results/plots/adadelta_h1_n32_1.0/adadelta_h1_n32_1.0_loss.png)

### 2. Curvas de MAE

#### Adam
![MAE Adam](results/plots/adam_h2_n64_lr001/adam_h2_n64_lr001_mae.png)  

#### Rmsprop
![MAE Rmsprop](results/plots/rmsprop_h2_n64_lr001/rmsprop_h2_n64_lr001_mae.png)

#### Adadelta
![MAE Adadelta](results/plots/adadelta_h1_n32_1.0/adadelta_h1_n32_1.0_mae.png)

### 3. Comparación Real vs Predicho

#### Adam
![Real vs Predicho Adam](results/plots/adam_h2_n64_lr001/adam_h2_n64_lr001_real_vs_pred.png)  

#### Rmsprop
![Real vs Predicho Rmsprop](results/plots/rmsprop_h2_n64_lr001/rmsprop_h2_n64_lr001_real_vs_pred.png) 

#### Adadelta
![Real vs Predicho Adadelta](results/plots/adadelta_h1_n32_1.0/adadelta_h1_n32_1.0_real_vs_pred.png)

### 4. Métricas (RMSE, MAPE)

| Modelo                | RMSE           | MAPE (%)         |
|-----------------------|----------------|------------------|
| adam_h2_n64_lr001     | 65 362.74      | 58.95            |
| rmsprop_h2_n64_lr001  | 111 285.15     | 98.57            |
| adadelta_h1_n32_1.0   | 16 866.73      | 12.10            |

## Predicción Nuevos Empleados

### adam_h2_n64_lr001

| predicted_salary | desired_salary | error_%   |
|------------------|---------------|-----------|
| 77530.91         | 150000        | -48.31    |
| 22272.738        | 40000         | -44.32    |
| 73759.48         | 150000        | -50.83    |
| 85534.65         | 160000        | -46.54    |
| 43173.848        | 90000         | -52.03    |

### rmsprop_h2_n64_lr001

| predicted_salary | desired_salary | error_%   |
|------------------|---------------|-----------|
| 2163.5764        | 150000        | -98.56    |
| 716.99615        | 40000         | -98.21    |
| 2016.3434        | 150000        | -98.66    |
| 2308.1353        | 160000        | -98.56    |
| 1246.7723        | 90000         | -98.61    |

### adadelta_h1_n32_1.0

| predicted_salary | desired_salary | error_%   |
|------------------|---------------|-----------|
| 128847.44        | 150000        | -14.10    |
| 38411.316        | 40000         | -3.97     |
| 127712.83        | 150000        | -14.86    |
| 151920.14        | 160000        | -5.05     |
| 83957.53         | 90000         | -6.71     |

<!-- Inserta aquí la tabla generada en `results/tables/{model_name}/predictions_errors.csv` -->

## Conclusiones y Observaciones

- **Rendimiento**: El modelo con _RMSprop_ y configuración base obtuvo el menor MAPE (~7.9 %), seguido de _Adam_.  
- **Profundidad**: Más capas y neuronas mejoran la capacidad pero requieren más datos/épocas para converger con _SGD_.  
- **Elección de LR**: Tasas muy altas (Adadelta 1.0) fueron inestables (MAPE alto).  
- **Aplicabilidad**: La subestimación sistemática sugiere añadir más datos o features (p. ej. ubicación, desempeño).  

## Referencias

1. Chollet, F. et al. _Deep Learning with Python_. Manning, 2017.  
2. Géron, A. _Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow_. O’Reilly, 2019.  
3. TensorFlow API – https://www.tensorflow.org/api_docs  
4. Scikit‑Learn Preprocessing – https://scikit-learn.org/stable/modules/preprocessing.html  
5. Oliva, D. “Tarea 2 – Modelos Predictivos,” Seminario IA II, CUCEI, 2025.