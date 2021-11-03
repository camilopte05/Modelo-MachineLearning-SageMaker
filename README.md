# Modelo de predicción

Esta es la implementación de un tutorial de AWS para crear, entrenar e implementar un modelo de Machine Learning con Amazon SageMaker. Para más información visite el sito web oficial de [AWS](https://aws.amazon.com/es/getting-started/hands-on/build-train-deploy-machine-learning-model-sagemaker/).

## Objetivo

Crear, entrenar e implementar un modelo de aprendizaje automático sencillo usando SageMaker. Utilizando el conocido algoritmo de aprendizaje automático XGBoost.

## Que aprenderás

- Creará una instancia de bloc de notas
- Preparará los datos
- Entrenará el modelo para aprender de los datos
- Implementará el modelo
- Evaluará el rendimiento de su modelo de aprendizaje automático

## Guía

Para seguir esta guía paso a paso es necesario tener creada una cuenta en **AWS** [haga clic aquí](https://portal.aws.amazon.com/billing/signup).

## Paso 1: Abra la consola de Amazon SageMaker

[Haga clic aquí](https://portal.aws.amazon.com/billing/signup), para abrir una consola de administración de AWS, estando ahí utilice la barra de búsqueda y escriba **Amazon SageMaker**, y posteriormente seleccionarlo para abrir el servicio.


<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen1.png" /></p>

## Paso 2: Cree una instancia de bloc de notas de Amazon SageMaker

Cree una instancia de **Instancia de bloc de notas**.

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen2.png" /></p>

Durante la creación de la **Instancia de bloc de notas**, escriba el **Nombre de instancia del bloc de notas** este puede ser cualquier nombre, las demás opciones se pueden dejar por defecto. Se debe **Crear un nuevo rol** para Amazon SageMaker, para permitir que la instancia del bloc de notas acceda a Amazon S3 y pueda cargar datos de manera segura en este servicio.

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen3.png" /></p>

Estando en **Crear una función de IAM**, se debe seleccionar **Cualquier bucket de S3**, para permitir que los usuarios tengan acceso a su instancia del bloc de notas, a cualquier bucket y a su contenido en su cuenta. Posteriormente seleccioné **Crear función**.

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen4.png" /></p>

Para esta guía, se utilizarán las demás opciones como predeterminadas, seleccione **Crear instancia de bloc de notas**.

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen5.png" /></p>

Estando en el apartado **Instancias de bloc de notas**, se le mostrará su nueva instancia de bloc de notas con el estado **Pending**, esta debería pasar en unos minutos al estado de **InService**.

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen6.png" /></p>

## Paso 3: Prepare los datos

Cuando el estado de la instancia del bloc de notas este en **InService**, diríjase al menú desplegable **Acciones** y seleccione la opción **Abrir Jupyter** o en la columna **Acciones** que se encuentra al lado de **InService**, seleccione la opción **Abrir Jupyter**.

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen7.png" /></p>

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen8.png" /></p>

Estando en **Jupyter** en la pestaña **Files**, seleccione **New**, posteriormente **conda_python3**.

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen9.png" /></p>

Se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**. Con esto se importaron algunas bibliotecas y definieron algunas variables del entorno en su entorno de bloc de notas de **Jupyter**. Con esto se prepararon los datos, necesarios para entrenar el modelo de aprendizaje automático e implementarlo.

Durante la ejecución del código, aparecerá el símbolo * entre corchetes, Luego de completarse la ejecución del código, el símbolo * se reemplazará por el número **1**.

```python3
# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np                                
import pandas as pd                               
import matplotlib.pyplot as plt                   
from IPython.display import Image                 
from IPython.display import display               
from time import gmtime, strftime                 
from sagemaker.predictor import csv_serializer   

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
containers = {'us-west-2': '433757028032.dkr.ecr.us-west-2.amazonaws.com/xgboost:latest',
              'us-east-1': '811284229777.dkr.ecr.us-east-1.amazonaws.com/xgboost:latest',
              'us-east-2': '825641698319.dkr.ecr.us-east-2.amazonaws.com/xgboost:latest',
              'eu-west-1': '685385470294.dkr.ecr.eu-west-1.amazonaws.com/xgboost:latest'} # each region has its XGBoost container
my_region = boto3.session.Session().region_name # set the region of the instance
print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + containers[my_region] + " container for your SageMaker endpoint.")
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen10.png" /></p>

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen11.png" /></p>

Se debe copiar el siguiente código en una celda de **Jupyter**, cambiando el nombre del bucket de S3 (Este debe de ser único), posteriormente seleccioné **Run**. De no recibir un mensaje de **successfully**, deberás cambiar el nombre de bucket de S3 y volver a intentarlo.

```python3
bucket_name = 'compunube' # <--- CAMBIE ESTA VARIABLE POR UN NOMBRE ÚNICO PARA SU BUCKET
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen12.png" /></p>

Se procede a descargar los datos en su instancia de **Amazon SageMaker** y cargarlos en un marco de datos. Se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**.

```python3
try:
  urllib.request.urlretrieve ("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
  print('Success: downloaded bank_clean.csv.')
except Exception as e:
  print('Data load error: ',e)

try:
  model_data = pd.read_csv('./bank_clean.csv',index_col=0)
  print('Success: Data loaded into dataframe.')
except Exception as e:
    print('Data load error: ',e)
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen13.png" /></p>

Se procederá a mezclar los datos y los dividiremos en datos de entrenamiento y de prueba.

Se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**. Con esto se mezclarán y dividirán los datos.

```python3
train_data, test_data = np.split(model_data.sample(frac=1, random_state=1729), [int(0.7 * len(model_data))])
print(train_data.shape, test_data.shape)
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen14.png" /></p>

## Paso 4: Entrene el modelo con los datos

Se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**. Con esto se cambiará el formato y se cargarán los datos.

```python3
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen15.png" /></p>

Se procede a configurar la sesión de **Amazon SageMaker**, crear una instancia del modelo XGBoost (un estimador) y definir los hiperparámetros del modelo. Se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**.

```python3
sess = sagemaker.Session()
xgb = sagemaker.estimator.Estimator(containers[my_region],role, train_instance_count=1, train_instance_type='ml.m4.xlarge',output_path='s3://{}/{}/output'.format(bucket_name, prefix),sagemaker_session=sess)
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen16.png" /></p>

Ya con los datos cargados y el estimador XGBoost configurado, entrene el modelo. Se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**.

Pasado algunos minutos, se debería ver los registros de entrenamiento que se han generado.

```python3
xgb.fit({'train': s3_input_train})
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen17.png" /></p>

## Paso 5: Implemente el modelo

Para la implementación de modelo en un servidor y crear un punto de enlace al que pueda acceder, se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**.

```python3
xgb_predictor = xgb.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen18.png" /></p>

Para predecir si los clientes de los datos de prueba se inscribieron o no en el producto del banco, se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**.

```python3
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.serializer = csv_serializer # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen19.png" /></p>

## Paso 6. Evalúe el rendimiento del modelo

Se debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**. Con esto se comparan los valores reales con los valores predichos en una tabla denominada **matriz de confusión**.

Con base en los pronósticos, es factible anticipar que un cliente se inscribirá en un certificado de depósito para exactamente el 90% de los consumidores en los datos de prueba, con una precisión del 63% (278/429) para los registrados y del 90%. (10 785/11928) para aquellos que no están registrados.

```python3
cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))
```

<p align="center"><img src="https://github.com/camilopte05/Modelo-MachineLearning-SageMaker/blob/main/img/Imagen20.png" /></p>

## Paso 7: Termine los recursos

Se elimina el punto de enlace de Amazon SageMaker y los objetos de su bucket de S3, para esto debe copiar el siguiente código en una celda de **Jupyter**, posteriormente seleccioné **Run**.

```python3
sagemaker.Session().delete_endpoint(xgb_predictor.endpoint)
bucket_to_delete = boto3.resource('s3').Bucket(bucket_name)
bucket_to_delete.objects.all().delete()
```

