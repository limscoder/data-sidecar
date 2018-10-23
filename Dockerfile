# base
FROM ubuntu:18.04
RUN apt-get update && apt-get install -y curl gzip golang-1.10-go

# install tensorflow
RUN mkdir -p /tf-build
WORKDIR /tf-build
RUN curl https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-linux-x86_64-1.11.0.tar.gz --output libtensorflow.tar.gz
RUN tar -zxf libtensorflow.tar.gz -C /usr/local
RUN ldconfig

# build binary
COPY . /usr/lib/go-1.10/src/github.com/open-fresh/data-sidecar/
WORKDIR /usr/lib/go-1.10/src/github.com/open-fresh/data-sidecar
RUN mkdir -p /data-sidecar/bin
RUN /usr/lib/go-1.10/bin/go build -o /data-sidecar/bin/data-sidecar

# config models
COPY scoring/tf/models /data-sidecar/models

# service
ENTRYPOINT ["/data-sidecar/bin/data-sidecar", "--prom=http://prom.predictatron.net:9090", "--resolution=60", "--lookback=60", "--tfpath=/data-sidecar/models/model-btc_usd-5m:/data-sidecar/models/model-btc_usd-15m:/data-sidecar/models/model-btc_usd-60m:/data-sidecar/models/model-eth_usd-5m:/data-sidecar/models/model-eth_usd-15m:/data-sidecar/models/model-eth_usd-60m:/data-sidecar/models/model-bch_usd-5m:/data-sidecar/models/model-bch_usd-15m:/data-sidecar/models/model-bch_usd-60m"]