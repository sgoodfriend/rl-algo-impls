rm -rf bin rai.jar
javac -cp "lib/*:src" -d bin $(find . -name "*.java")
cd bin
find ../lib -name "gson-2.10.1.jar" | xargs -n 1 jar xvf
rm -rf META-INF
jar cvf rai.jar $(find . -name '*.class' -type f)
cd ..