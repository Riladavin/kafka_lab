# Kafka installation

## Install Apple Silicon Rosetta
```bash
softwareupdate --install-rosetta
```

## Install Docker
1. Install docker from [Docker](https://www.docker.com)

## Get Kafka Docker Image
```bash
docker pull bitnami/kafka:latest
```

## Streamlit
For better performance, install the Watchdog module:
```bash
xcode-select --install
pip install watchdog
```