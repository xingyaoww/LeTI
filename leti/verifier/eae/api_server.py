import flask
from leti.verifier.eae.utils import construct_pred_set
from leti.verifier.eae.nlp_utils import nlp

app = flask.Flask(__name__)

@app.route('/api', methods=['POST'])
def handle_construct_pred_set():
    request = flask.request.get_json()
    predicted_args = request['predicted_args']
    cur_event = request['cur_event']
    tokens = request['tokens']

    predicted_set, not_matched_pred_args = construct_pred_set(
        predicted_args,
        cur_event,
        tokens,
        nlp(" ".join(tokens)),
        head_only=True,
        nlp=nlp
    )
    return flask.jsonify({
        'predicted_set': str(predicted_set),
        'not_matched_pred_args': str(not_matched_pred_args)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
