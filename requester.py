import requests


def upload_data(id, header, state):
    auth = '54dsnv8s7hghdf4i7bdgsuASDLHs'
    url = 'https://compupandilla.castelancarpinteyro.com/web-repo/'#?name=Dante&last_names=Castelan%20Carpinteyro'

    # Hacer una solicitud POST
    response = requests.post(url, data={'auth': auth,'id': id, 'header': header, 'state': state})
    #response = requests.get(url)


    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        print('Solicitud exitosa')
        print('Contenido:', response.text)
    else:
        print('Error en la solicitud:', response.status_code)
