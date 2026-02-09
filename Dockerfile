FROM eclipse-temurin:23-jre

WORKDIR /app
COPY snake_ai-1.0-SNAPSHOT.jar /app/snake_ai.jar

# Runtime working directory for config and output files
RUN mkdir -p /data
WORKDIR /data

# Default config inside the image (will be overwritten if you mount your folder to /data)
COPY config.properties /data/config.properties

# Hint: /data is meant to be mounted from host
VOLUME ["/data"]

ENV JAVA_OPTS=""

ENTRYPOINT ["sh","-c","exec java $JAVA_OPTS -jar /app/snake_ai.jar"]
