
default: build

fetch:
	mkdir -p s3
	s3cmd sync s3://h2o-tutorials/ s3/

build: fetch
	docker build -t opsh2oai/h2o-training -f Dockerfile .

run:
	docker run --init --rm -u h2o:h2o -p 4040:4040 -p 8787:8787 -p 8888:8888 -p 54321-54399:54321-54399 opsh2oai/h2o-training

save:
	docker save opsh2oai/h2o-training | pigz -c > h2o-training.gz

