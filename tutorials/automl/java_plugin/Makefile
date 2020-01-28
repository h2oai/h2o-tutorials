
PLUGIN := h2oautoml_plugin.jar
H2O_JAR := "$(shell pip show h2o | grep Location | awk '{print $$2}')/h2o/backend/bin/h2o.jar"

SRC := $(shell find ./src -name "*.java")

clean:
	rm -Rf ./build ./dist

build: clean
	@echo "sources = $(SRC)"
	mkdir -p build
	javac $(SRC) -cp $(H2O_JAR) -d ./build
	cp -R ./src/META-INF ./build

dist: build
	mkdir -p dist
	jar cvf ./dist/$(PLUGIN) -C ./build .

run:
	java -ea -agentlib:jdwp=transport=dt_socket,server=y,suspend=n,address=5005 -Dsys.ai.h2o.disable.cors=true -XX:OnOutOfMemoryError="kill -9 %p" -cp $(H2O_JAR):dist/$(PLUGIN) water.H2OApp -nthreads 2 -log_level INFO > ~/tmp/out.log 2>&1 &

kill:
	jps | grep H2OApp | awk '{print $$1}' | xargs kill
