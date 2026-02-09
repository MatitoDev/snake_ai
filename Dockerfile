FROM eclipse-temurin:23-jre

WORKDIR /app
COPY snake_ai-1.0-SNAPSHOT.jar /app/snake_ai.jar

RUN mkdir -p /data
WORKDIR /data

COPY config.properties /data/config.properties
VOLUME ["/data"]

EXPOSE 8080

ENV JAVA_OPTS=""

ENTRYPOINT ["sh","-c","exec java $JAVA_OPTS -jar /app/snake_ai.jar"]
