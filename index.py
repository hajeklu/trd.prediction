from bottle import run, get, post, default_app, HTTP_CODES, request, response
from classify import call
from candle import arrayToCandle
import json
app = default_app()

for http_code in HTTP_CODES:
    @app.error(http_code)
    def json_error(error):
        response.default_content_type = "application/json"
        return json.dumps(dict(error=error.status_code, message=error.status_line))


@get('/')
def getIndex():
    return "Success"


@post('/predict')
def predict():
    candles = request.body.read()
    response.content_type = 'application/json; charset=UTF8'
    candleDirection = call(json.loads(candles))
    direction = json.dumps({'direction': candleDirection.tolist()[0][0]})
    return direction


run(port=8080)
