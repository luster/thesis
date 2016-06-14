import requests
import json

url = "https://hooks.slack.com/services/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"
requests.post(url, data=json.dumps({
    'text': 'FUCKKKKK'
}))
