import findspark
findspark.init()
from flask import Flask, request, jsonify, render_template
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
import os
from pyspark.sql.types import DoubleType
import pandas as pd

spark = SparkSession.builder.appName('predict-attrition').getOrCreate()
model = PipelineModel.load("Model")
app = Flask(__name__, template_folder=os.path.abspath('templates'))

@app.route('/')
def generate_html():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    json_data = request.get_json()
    data = json_data['data']
    data = pd.DataFrame.from_dict(data, orient='index').transpose()
    spark_df = spark.createDataFrame(data)

    spark_df = spark_df.withColumn("MonthlyRate", spark_df["MonthlyRate"].cast(DoubleType()))
    spark_df = spark_df.withColumn("NumCompaniesWorked", spark_df["NumCompaniesWorked"].cast(DoubleType()))
    spark_df = spark_df.withColumn("MonthlyIncome", spark_df["MonthlyIncome"].cast(DoubleType()))
    spark_df = spark_df.withColumn("JobLevel", spark_df["JobLevel"].cast(DoubleType()))
    spark_df = spark_df.withColumn("Age", spark_df["JobLevel"].cast(DoubleType()))

    prediction = model.transform(spark_df).head()

    app.logger.info('%s logged in successfully', prediction)

    return jsonify(prediction=float(prediction.prediction), probability=float(prediction.probability[1]))

if __name__ == '__main__':
    # Démarrer le serveur Prometheus HTTP sur le port 8000

    app.run(host='0.0.0.0', port=5000)
