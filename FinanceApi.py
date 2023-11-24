import json

from flask import Flask, jsonify, request
import LSTM_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def predict_future1(start_date, end_date, tickers, future_time_steps, time_steps):
    # LSTM modelini kullanarak gelecekteki değerleri tahmin et
    result = LSTM_model.predict_future(start_date, end_date, tickers, future_time_steps, time_steps)

    return result

# Endpoint: Veriyi sağla
@app.route('/api/get_data', methods=['POST'])
def get_data():
    request_data = request.data.decode('utf-8')
    request_json = json.loads(request_data)

    future_time_steps = int(request_json.get('futureTimeSteps'))
    time_steps = int(request_json.get('timeSteps'))
    start_date = request_json.get('startDate')
    end_date = request_json.get('endDate')
    tickers = request_json.get('tickers')

    # Değerleri kontrol et ve gerekirse varsayılan değer atayarak işle
    if not start_date:
        start_date = '2022-01-01'  # Varsayılan başlangıç tarihi
    if not end_date:
        end_date = '2022-12-31'  # Varsayılan bitiş tarihi
    if not tickers:
        tickers = 'AAPL'  # Varsayılan hisse senedi simgesi
        # LSTM_model.predict_future fonksiyonunun yerine predict_future fonksiyonunu kullan
    predictions = predict_future1(start_date, end_date, tickers, future_time_steps, time_steps)

    return jsonify({'predictions': predictions})

if __name__ == '__main__':
    app.run(debug=True)
