# Create an ssh tunnel to the server for postgres

```shell
ssh -L 5432:localhost:5432 -N server.02
```

```sh
streamlit run app.py
```

## Build Docker Container

```sh
docker build -t streamlit-patent-landscaping .
```

## Run Docker Container

```sh
docker run -d -p 8501:8501 streamlit-patent-landscaping
```

## Build Docker image locally and store it as a tar file

```sh
docker save -o streamlit-patent-landscaping.tar registry.erhardt.net/streamlit-patent-landscaping
```

## Transfer Docker image to server

```sh
scp streamlit-patent-landscaping.tar server.02:/home/docker-compose/patent-landscaping
```

## Load Docker image on server

```sh
docker load -i streamlit-patent-landscaping.tar
```

## Run Docker container on server using docker-compose

```sh

```