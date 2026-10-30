FROM node:24-bookworm-slim

ARG T3_VERSION=0.1.5
ARG CODEX_VERSION=0.130.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl git openssh-client procps ripgrep \
    && rm -rf /var/lib/apt/lists/* \
    && npm install --global "t3@${T3_VERSION}" "@openai/codex@${CODEX_VERSION}" \
    && mkdir -p /home/node/workspace \
    && chown node:node /home/node/workspace

USER node
WORKDIR /home/node/workspace
ENV HOME=/home/node

EXPOSE 3773
CMD ["t3", "serve", "--host", "0.0.0.0", "--port", "3773"]
