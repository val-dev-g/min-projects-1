from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, GBTClassifier,DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.tuning import TrainValidationSplit, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from pyspark.sql.functions import stddev, avg
from prometheus_client import Counter, Gauge, Histogram
from pyspark.metrics import MetricsSystem
from pyspark.prometheus import initialize_exporter, stop_exporter
from prometheus_client import CollectorRegistry
from time import sleep

spark = SparkSession.builder.appName('attrition').getOrCreate()
file_location = "HR-Employee-Attrition.csv"
file_type = "csv"
# CSV options
infer_schema = "false"
first_row_is_header = "true"
delimiter = ","
# The applied options are for CSV files. For other file types, these will be ignored.
df = spark.read.format(file_type) \
.option("inferSchema", infer_schema) \
.option("header", first_row_is_header) \
.option("sep", delimiter) \
.load(file_location)
df = df.drop("Over18")
df = df.drop("EmployeeCount")
df = df.drop("StandardHours")

registry = CollectorRegistry(auto_describe=True)
initialize_exporter(registry)

# gbt = GBTClassifier(featuresCol='features', labelCol='label')
# param_grid_gbt = ParamGridBuilder() \
#     .addGrid(gbt.maxDepth, [2, 4, 6]) \
#     .addGrid(gbt.maxBins, [20, 30]) \
#     .addGrid(gbt.maxIter, [10, 20, 30]) \
#     .build()

# rf = RandomForestClassifier(featuresCol='features', labelCol='label')
# param_grid_rf = ParamGridBuilder() \
#     .addGrid(rf.maxDepth, [2, 5, 10]) \
#     .addGrid(rf.maxBins, [10, 20, 30]) \
#     .addGrid(rf.numTrees, [10, 50, 100]) \
#     .addGrid(rf.impurity, ['gini', 'entropy']) \
#     .build()

# lr = LogisticRegression(featuresCol='features', labelCol='label')
# param_grid_lr = ParamGridBuilder() \
#     .addGrid(lr.regParam, [0.01, 0.05, 0.1]) \
#     .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
#     .build()

gbt = GBTClassifier(featuresCol='features', labelCol='label')
param_grid_gbt = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [2]) \
    .addGrid(gbt.maxBins, [20]) \
    .addGrid(gbt.maxIter, [10]) \
    .build()

rf = RandomForestClassifier(featuresCol='features', labelCol='label')
param_grid_rf = ParamGridBuilder() \
    .addGrid(rf.maxDepth, [2]) \
    .addGrid(rf.maxBins, [10]) \
    .addGrid(rf.numTrees, [10]) \
    .addGrid(rf.impurity, ['gini']) \
    .build()

lr = LogisticRegression(featuresCol='features', labelCol='label')
param_grid_lr = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01]) \
    .addGrid(lr.elasticNetParam, [0.0]) \
    .build()


def convert_to_int(df):
    # convertir les colonnes string en double quand cela est possible
    for column in df.columns:
        try:
            if int(df.select(column).first()[0]):
                df = df.withColumn(column, col(column).cast("double"))
        except:
            pass
    return df
        
def remove_outliers(df):

    # Boucle pour enlever les valeurs aberrantes de chaque colonne
    for col in df.columns:
        # Calcul des statistiques pour la colonne
        stats = df.select(avg(col), stddev(col)).first()
        mean = stats[0]
        std = stats[1]  
        if mean is not None and std is not None:
            # Calcul du seuil pour déterminer les valeurs aberrantes
            threshold = 3 * std + mean     
            # Suppression des valeurs aberrantes pour la colonne
            df = df.filter(df[col] <= threshold)
    return df


def assembler_for_pipeline(df, label):
    categoricalColumns = [col for (col, dtype) in df.dtypes if dtype == "string" and col != label]
    stages = []

    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]
    
    label_stringIdx = StringIndexer(inputCol=label, outputCol="label")
    stages += [label_stringIdx]

    numericCols = [col for (col, dtype) in df.dtypes if dtype.startswith('double') and col != "label"]
    assemblerInputs = [c + "classVec" for c in categoricalColumns] + numericCols
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    return assembler, stages

def model_for_pipeline(classifier, param_grid):
    evaluator = BinaryClassificationEvaluator(labelCol='label')
    cv = CrossValidator(estimator=classifier, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3).setSeed(1234)
    return cv

def view_feature_importances(df):
    
    assembler, stages = assembler_for_pipeline(df, 'Attrition')
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")
    paramGrid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20]) \
        .addGrid(rf.maxDepth, [5, 10]) \
        .build()
    
    evaluator = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")
    train, test = df.randomSplit([0.8, 0.2])
    stages += [assembler, rf]
    pipeline = Pipeline(stages=stages)
    
    pipelineModel = pipeline.fit(df)
    predictions = pipelineModel.transform(df)
    auc = evaluator.evaluate(predictions)
    print("AUC-ROC = %g" % auc)
    
    importances = pipelineModel.stages[-1].featureImportances.toArray()
    cols = df.columns
    selectedcols = ["label", "features"] + cols
    cols_importances = list(zip(selectedcols, importances))
    sorted_cols_importances = sorted(cols_importances, key=lambda x: x[1], reverse=True)
    print("Colonnes triées par ordre d'importance décroissante :")
    for col, importance in sorted_cols_importances:
        print(col, "=", importance)
        
    return sorted_cols_importances
        
def pipelinedata(df, assembler, stages, model):
    
    stages += [assembler, model]
    train, test = df.randomSplit([0.8, 0.2])
    pipeline = Pipeline(stages=stages)
    pipelineModel = pipeline.fit(train)
    predictions = pipelineModel.transform(test)
    predictions.take(1)
    selected = predictions.select("label", "prediction", "rawPrediction", "probability")
    return model, predictions, selected

def metrics(predictions, model):

    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC")
    auc_roc = evaluator.evaluate(predictions)
    evaluator = MulticlassClassificationEvaluator(predictionCol="prediction", labelCol="label", metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
    rmse = evaluator.evaluate(predictions)

    print(model + ": courbe ROC = %g" % auc_roc)
    print(model + ": précision = %g" % accuracy)
    print(model + ": RMSE = %g" % rmse)

    return accuracy
    

def filter_most_important_columns(df, sorted_cols_importances, label):
    selected_cols = [col for col, importance in sorted_cols_importances if importance > 0.01]
    if label not in selected_cols:
        selected_cols.append(label)
    if "features" in selected_cols:
        selected_cols.remove("features")
    if "label" in selected_cols:
        selected_cols.remove("label")
    df = df.select(selected_cols)
    return df


def select_most_efficient_model(df):

    best_model = None
    best_accuracy = 0
    models = {}
    
    df = convert_to_int(df)
    df = remove_outliers(df)
    # je selectionne les colonnes de manières brut pour qu'elles ne se modifie pas
    df_most_important = df.select(['MonthlyRate', 'NumCompaniesWorked', 'MonthlyIncome', 'JobLevel', 'Attrition', 'Age', 'Gender'])

    df_gbt = df_most_important
    df_rf = df_most_important
    df_lr = df_most_important

    # model random forest
    assembler, stages = assembler_for_pipeline(df_rf, 'Attrition')
    model_rf = model_for_pipeline(rf, param_grid_rf)
    model_rf, predictions_rf, selected_rf = pipelinedata(df_rf, assembler, stages, model_rf)
    accuracy = metrics(predictions_rf, 'rf')
    models["rf"] = {"model" : model_rf, "accuracy": accuracy}


    # model gradient boosting
    assembler, stages = assembler_for_pipeline(df_gbt, 'Attrition')
    model_gbt = model_for_pipeline(gbt, param_grid_gbt)
    model_gbt, predictions_gbt, selected_gbt = pipelinedata(df_gbt, assembler, stages, model_gbt)
    accuracy = metrics(predictions_gbt, 'gbt')
    models["gbt"] = {"model" : model_gbt, "accuracy": accuracy}

    # model logistic regression
    assembler, stages = assembler_for_pipeline(df_lr, 'Attrition')
    model_lr = model_for_pipeline(lr, param_grid_lr)
    model_lr, predictions_lr, selected_lr = pipelinedata(df_lr, assembler, stages, model_lr)
    accuracy = metrics(predictions_lr, 'lr')
    models["lr"] = {"model" : model_lr, "accuracy": accuracy}

    for model_name, model_info in models.items():
        accuracy = model_info['accuracy']
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = model_info['model']
            best_model_name = model_name

    return best_model_name, best_model, best_accuracy


best_model_name, model, accuracy = select_most_efficient_model(df)

print(best_model_name)
print(accuracy)
# suivre l'accuracy de notre modèle avec Promotheus
prediction_train = Gauge('train_prediction', 'suivre la prédiction du train', ['model_name', 'dataset_name'])
prediction_train.labels(best_model_name, 'HR-Employee-Attrition').set(accuracy)

print("sent data to prometheus")

model.write().overwrite().save("Model")

sleep(800000)

MetricsSystem.stop()
stop_exporter()
