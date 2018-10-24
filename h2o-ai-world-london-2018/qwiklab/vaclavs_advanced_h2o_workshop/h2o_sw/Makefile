
build:
	docker build -t opsh2oai/nyc-workshop-2018 -f Dockerfile .

run:
	docker run --init --rm -u h2o:h2o -p 4040:4040 -p 8888:8888 -p 54321-54399:54321-54399 opsh2oai/nyc-workshop-2018

save:
	docker save opsh2oai/nyc-workshop-2018 | gzip -c > nyc-workshop-2018.gz

