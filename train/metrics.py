from prometheus_client import Counter, Gauge, Histogram

# Créer les métriques pour suivre vos prédictions
predictions_counter = Counter('predictions_total', 'Total number of predictions made')
prediction_duration_histogram = Histogram('prediction_duration_seconds', 'Prediction duration in seconds')
prediction_latency_gauge = Gauge('prediction_latency_seconds', 'Time between prediction request and response', ['version'])
