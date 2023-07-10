rm -rf bin rai.jar RAISocketAI.jar
javac -cp "src:$(find lib -name "*.jar" | tr '\n' ':')" -d bin $(find . -name "*.java") --release 11
cd bin
find ../lib -name "gson-2.10.1.jar" | xargs -n 1 jar xvf
find ../lib -name "commons-cli-1.5.0.jar" | xargs -n 1 jar xvf
rm -rf META-INF
jar cvf RAISocketAI.jar $(find . -name '*.class' -type f)
mv RAISocketAI.jar ../RAISocketAI.jar
cd ..