# Required to make env_* scripts work
.EXPORT_ALL_VARIABLES:

all: dev

PROJECT=oneformer

DOCKER_DOMAIN ?= us.gcr.io
# Allows us to use a different registry in CI builds for caching
DOCKER_URL ?= ${DOCKER_DOMAIN}/ridecell-1

TAG ?= dev

# For scripts/env_*
# Generic is for development and CI testing
GENERIC_KEY ?= alias/microservices_dev
# Owner is per microservice key for Kubernetes configs
OWNER_KEY ?= ${GENERIC_KEY}

FORCE_AUTO_RELOAD ?= 1

build:
	docker-compose build

clean:
    # If they aren't found, don't error out
	-docker-compose down
	-docker-compose rm -f

depends:
	docker-compose up --build -d --remove-orphans ${COMPOSE_BUILD_OPT}

dev:
	docker-compose -f docker-compose.yml up --build -d --remove-orphans ${COMPOSE_BUILD_OPT}
	make shell

shell:
	docker-compose exec oneformer_app bash

docker_up:
	docker-compose -f docker-compose.yml -f docker-compose-dev.yml up -d
