from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
import main

app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        drug = main.main(file_path)
        return drug
    return None


if __name__ == '__main__':
    app.run(host='127.0.0.2',  debug=True)
