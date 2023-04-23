import findspark
findspark.init()
import json
from flask import Flask, request, jsonify, render_template,Response
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression 
from pyspark.ml import Pipeline
import os
import time
from pyspark.sql.types import StructType, StructField, DoubleType
import pandas as pd
from metrics import predictions_counter, prediction_duration_histogram, prediction_latency_gauge
from prometheus import init_prometheus
from prometheus_client import Counter
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

spark = SparkSession.builder.appName('predict-attrition').getOrCreate()
model = PipelineModel.load("/app/Model")
app = Flask(__name__, template_folder=os.path.abspath('templates'))

#metrics = init_prometheus(app)

metrics = make_wsgi_app()
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {'/metrics': metrics})

@app.route('/')
def generate_html():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()
    predictions_counter.inc()

    json_data = request.get_json()
    data = json_data['data']
    data = pd.DataFrame.from_dict(data, orient='index').transpose()
    spark_df = spark.createDataFrame(data)
    prediction = model.transform(spark_df).head()

    duration = time.time() - start_time
    prediction_duration_histogram.observe(duration)
    prediction_latency_gauge.labels(version='v1.0').set(duration)

    app.logger.info('%s logged in successfully', prediction)

    return jsonify(prediction=float(prediction.prediction), probability=float(prediction.probability[1]))

if __name__ == '__main__':
    # Démarrer le serveur Prometheus HTTP sur le port 8000

    app.run(host='0.0.0.0', port=5000)
