import requests

url = 'https://compupandilla.castelancarpinteyro.com/web-repo/?name=Dante&last_names=Castelan%20Carpinteyro'

# Hacer una solicitud GET
response = requests.get(url)

# Verificar si la solicitud fue exitosa
if response.status_code == 200:
    print('Solicitud exitosa')
    print('Contenido:', response.text)
else:
    print('Error en la solicitud:', response.status_code)
