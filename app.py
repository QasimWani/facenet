import flask
import predict

# Use pickle to load in the pre-trained model.

app = flask.Flask(__name__, template_folder='templates')

@app.route('/', methods=['GET', 'POST'])
def main():
    if flask.request.method == 'GET':
        return(flask.render_template('main.html'))
    if flask.request.method == 'POST':
        frame = flask.request.json["data"]
        embeddings = predict.run(predict.load_model(), frame)
        return flask.jsonify(embeddings)

if __name__ == '__main__':
    app.run()