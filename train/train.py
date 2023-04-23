import findspark
findspark.init()

from pyspark.sql.functions import corr
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql.types import DoubleType
from pyspark.sql import SparkSession
from pyspark.sql.functions import col,sum
from pyspark.sql.functions import stddev, avg
import numpy as np
from pyspark.ml.classification import GBTClassifier

def train_model():

    spark = SparkSession.builder.appName('attrition').getOrCreate()
    # File location and type
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


    def get_convert_to_numerical(df):
        num_col = []
        for column in df.columns:
            try:
                if int(df.select(column).first()[0]):
                    num_col += [column]
                    df = df.withColumn(column, df[column].cast(DoubleType()))
            except:
                pass
        return df, num_col


    def remove_columns_with_one_value(df):
        try:
            columns = df.columns
            cols_to_remove = []
            for col_name in columns:
                n_distinct = df.select(col(col_name)).distinct().count()
                if n_distinct < 2:
                    cols_to_remove.append(col_name)
            df = df.drop(*cols_to_remove)
            return df
        except:
            return df


    def remove_outliers(df):
        for col in df.columns:
            # Calcul des statistiques pour la colonne
            stats = df.select(avg(col), stddev(col)).first()
            mean = stats[0]
            std = stats[1]
            if mean is not None and std is not None:
            # Calcul du seuil pour déterminer les valeurs aberrantes
                threshold = 3 * std + mean
                df = df.filter(df[col] <= threshold)
        return df


    df, num_col = get_convert_to_numerical(df)
    # Convertir en DataFrame Pandas
    df_pandas = df.toPandas()
    df_pandas.corr(numeric_only=True)

    # Calculer la matrice de corrélation
    corr_matrix = df_pandas.corr(numeric_only=True).abs()
    # Exclure la diagonale   
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    mask[np.triu_indices_from(mask, k=1)] = False
    # Enlever les colonnes qui ont une corrélation trop hautes basses ou null
    numerical_columns_to_drop = [c for c in corr_matrix.columns if any(mask[:, corr_matrix.columns.get_loc(c)]) and (any(corr_matrix.loc[mask[:, corr_matrix.columns.get_loc(c)], c] > 0.8) or any((corr_matrix[c] < 0.2) & (corr_matrix[c] > 0)))]
    df_pandas = df_pandas.drop(numerical_columns_to_drop, axis=1)
    # reconvertir en df spark
    df_spark = spark.createDataFrame(df_pandas)
    df_spark = remove_columns_with_one_value(df_spark)
    # Comptage du nombre de valeurs nulles pour chaque colonne
    null_counts = df_spark.agg(*[sum(col(c).isNull().cast("int")).alias(c) for c in df_spark.columns])
    remove_outliers(df_spark)
    df_spark, num_col = get_convert_to_numerical(df_spark)
    
    categoricalColumns = [col for (col, dtype) in df_spark.dtypes if dtype == "string" and col != 'Attrition']
    
    stages = []
    for categoricalCol in categoricalColumns:
        stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')
        encoder = OneHotEncoder(inputCols=[stringIndexer.getOutputCol()], outputCols=[categoricalCol + "classVec"])
        stages += [stringIndexer, encoder]
    label_stringIdx = StringIndexer(inputCol = 'Attrition', outputCol = 'label')
    stages += [label_stringIdx]


    # Définition du modèle Gradient Boosting
    gbt = GBTClassifier(featuresCol='features', labelCol='label')

    param_grid = ParamGridBuilder() \
        .addGrid(gbt.maxDepth, [2, 4, 6]) \
        .addGrid(gbt.maxBins, [20, 30]) \
        .addGrid(gbt.maxIter, [10, 20, 30]) \
        .build()
    
    # Définition de l'évaluateur
    evaluator = BinaryClassificationEvaluator(labelCol='label')

    # Définition du validateur croisé
    cv = CrossValidator(estimator=gbt, estimatorParamMaps=param_grid, evaluator=evaluator, numFolds=3)

    assemblerInputs = [c + "classVec" for c in categoricalColumns] + num_col
    assembler = VectorAssembler(inputCols=assemblerInputs, outputCol="features")
    stages += [assembler, gbt]
    pipeline = Pipeline(stages=stages)
    (train_data, test_data) = df_spark.randomSplit([0.7, 0.3], seed=42)
    gbtModel = pipeline.fit(train_data)
    predictions = gbtModel.transform(test_data)
    accuracy = evaluator.evaluate(predictions)
    print(accuracy)
    gbtModel.write().overwrite().save("Model")

train_model()