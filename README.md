# api_docker

## Machine Learning en Producción

Integrantes: Felipe Bastarrica (158687), Aldo Gioda (285961), Esteban Maestro (150882)

Repositorio que contiene el Docker File para la ejecución del Endpoint FastAPI. 

Creación y ejecución de imagen de docker para la FastAPI:

* docker build -t api:latest .

* docker run --restart always -d -p 8080:80 api:latest
