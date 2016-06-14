import requests
import json

def post_slack(msg):
    url = "https://hooks.slack.com/services/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"
    requests.post(url, data=json.dumps({
        'text': msg, 'channel': '#general',
    }))
