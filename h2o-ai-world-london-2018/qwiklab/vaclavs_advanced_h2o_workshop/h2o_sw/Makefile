
build:
	docker build -t opsh2oai/london-workshop-2018 -f Dockerfile .

run:
	docker run --init --rm -u h2o:h2o -p 8787:8787 -p 4040:4040 -p 8888:8888 -p 54321-54399:54321-54399 opsh2oai/london-workshop-2018

save:
	docker save opsh2oai/london-workshop-2018 | gzip -c > london-workshop-2018.gz

