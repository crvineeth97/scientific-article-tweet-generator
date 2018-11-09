from googlesearch import search
import requests


def google(string):
    for j in search(string, tld='com', lang='en', num=10, stop=1, pause=2.0):
        break
    print('URL: ' + j)
    return j


def shorten(string):
    uri = google(string)
    query_params = {
        'access_token': "9de58831f29edb8eb48790f3007f338b962e6407",
        'longUrl': uri
    }

    endpoint = 'https://api-ssl.bitly.com/v3/shorten'
    response = requests.get(endpoint, params=query_params, verify=True)

    data = response.json()

    if not data['status_code'] == 200:
        print('lol')

    shortened = data['data']['url']
    if "http://" in shortened:
        shortened = shortened[7:]
    elif "https://" in shortened:
        shortened = shortened[8:]
    if "www." in shortened:
        shortened = shortened[4:]
    return shortened


# EXAMPLE
# print(shorten("improving surveillance using cooperative target observation"))
# print(shorten("Markets are efficient if and only if P = NP"))
