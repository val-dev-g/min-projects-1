from prometheus_flask_exporter import PrometheusMetrics

def init_prometheus(app):
    metrics = PrometheusMetrics(app)
    return metrics