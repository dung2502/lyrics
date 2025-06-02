from flask import Flask, jsonify
import requests
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

@app.route('/get-json', methods=['GET'])
def get_json():
    url = "https://drive.google.com/uc?id=1iM20Mu14O1d6_Oql22HRb_H7PBTwEPLY&export=download"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Kiểm tra lỗi HTTP
        data = response.json()  # Chuyển đổi dữ liệu thành JSON
        return jsonify(data)
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Lỗi khi fetch dữ liệu: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
