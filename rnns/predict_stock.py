from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
from lstm_hb import predict_stock
from prometheus_flask_exporter import PrometheusMetrics
from prometheus_client import start_http_server, Counter, Histogram

# Inicializar a aplicação Flask e a API Flask-RESTPlus
app = Flask(__name__)

metrics = PrometheusMetrics(app)

api = Api(app, version='1.0', title='Stock Price Prediction API',
          description='API para previsão de preço de ações usando um modelo LSTM treinado')



# Definir o modelo de entrada para o Swagger
predict_model = api.model('PredictModel', {
    'symbol': fields.String(required=True, description='Símbolo da ação'),
    'reference_date': fields.String(required=True, description='Data de referência (formato: yyyy-mm-dd)')
})

PREDICTION_REQUESTS = Counter('prediction_requests', 'Number of prediction requests')

REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency in seconds', ['method', 'endpoint'])


@api.route('/predict')
class PredictPrice(Resource):
    @api.doc(description="Previsão de preço de ação para um símbolo e data de referência específicos")
    @api.expect(predict_model)  # Expecting the 'predict_model' as input
    @REQUEST_LATENCY.time()  # Decorator para medir a latência
    def post(self):
        PREDICTION_REQUESTS.inc()
        
        # Receber os dados de entrada
        data = request.get_json()
        symbol = data.get('symbol')
        reference_date = data.get('reference_date')

        # Validação de dados de entrada
        if not symbol or not reference_date:
            return jsonify({"error": "Missing symbol or reference_date"}), 400

        try:
            prediction = predict_stock(symbol, reference_date)

            # Retornar o resultado
            #return jsonify(prediction)
            return {"reference_date": reference_date, "prediction": prediction}
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    start_http_server(8000)
    app.run(debug=True, port=5000)