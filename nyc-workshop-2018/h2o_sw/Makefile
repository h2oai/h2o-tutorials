
build:
	docker build -t opsh2oai/h2o-world-2017 -f Dockerfile .

run:
	docker run --init --rm -u h2o:h2o -p 4040:4040 -p 8787:8787 -p 8888:8888 -p 54321-54399:54321-54399 opsh2oai/h2o-world-2017

save:
	docker save opsh2oai/h2o-world-2017 | gzip -c > h2o-world-2017.gz

