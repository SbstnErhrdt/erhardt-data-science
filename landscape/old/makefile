run-local:
	source ./.env
	uvicorn app.main:app --reload

lint:
	pylint ./src/**
install-requirements:
	pip3 install -r requirements.txt
generate-requirements:
	pip3 list --format=freeze > requirements.txt

build-docker:
	docker build -t patent-portfolio-app .

start-docker:
	docker run -d -p 8000:8000 --env-file=.env patent-portfolio-app

build-deploy:
	sh build.sh