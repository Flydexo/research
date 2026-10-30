# T3 Code on Dokploy

This is a headless T3 Code server with the Codex CLI, running as a non-root user. T3 Code, provider credentials, Git configuration, and projects under `/home/node` persist in a Docker named volume. The image pins T3 Code 0.1.5 and Codex CLI 0.130.0; change the Dockerfile versions and rebuild to update. T3's npm launcher downloads its matching server release from GitHub on first start, so the server needs outbound network access.

## Deploy

1. In Dokploy, create a **Compose** service using Docker Compose (not Docker Stack). Select this repository and branch, and set the Compose path to `t3code-dokploy/compose.yaml`. The Dockerfile is built from the Compose file's directory. Enable isolated deployments if available.
2. Point a DNS record for your chosen domain to the Dokploy server. Under the Compose service's **Domains**, add the domain for service `t3code`, container port `3773`, path `/`, and enable HTTPS with a valid certificate. Deploy or redeploy after configuring the domain. Dokploy adds the proxy labels and network automatically. The Compose file does not publish a host port.
3. Open a shell as the container's `node` user using the Dokploy terminal, or on the Docker host use `docker exec -it <container-id> sh`. Run `t3 pair`. The command prints a short-lived one-time pairing URL and token. Its address may be the container's internal address: keep the complete `/pair#token=...` portion and replace the origin with your HTTPS domain, for example `https://t3.example.com/pair#token=...`. Open that URL in your browser or enter it in the T3 Code client. Generate a fresh link for each device; treat it as a password.
4. Configure a provider in Settings. Codex is included; to sign in interactively, run `codex login` in the container and follow its instructions, or use T3 Code's provider settings. Other providers must be installed in the image and authenticated in the container. For Git over SSH, configure a key and known hosts inside the persistent home; do not bake credentials into the image or commit them.
5. Clone projects into `/home/node/workspace` or mount another persistent volume at a directory owned by UID/GID 1000. Container-only paths outside the home volume are lost on replacement.

The paired client can access files and run agent commands as the container user. Use HTTPS, protect pairing links and account credentials, and restrict access to trusted users. Back up the named volume securely: it contains project files, T3 state, and potentially credentials. Don't attach the Docker socket or mount the host filesystem unless you intend to grant that access.

## Local smoke test

On a machine with Docker Compose, from this directory run:

```sh
docker compose build
docker compose -f compose.yaml -f compose.local.yaml up -d
docker compose exec t3code t3 pair
```

Open the printed pairing link, replacing its origin with `http://127.0.0.1:3773` if necessary. Stop with `docker compose -f compose.yaml -f compose.local.yaml down`; do not add `-v` if you want to keep the volume. The local override must not be used for the Dokploy deployment.

T3's optional remote browser needs additional Linux dependencies and sandbox configuration; this image does not set up that feature. Check the upstream remote-access guide before enabling it. If a newer client requires a newer server, update the pinned T3 version, rebuild and redeploy; back up the volume first because upgrades may migrate its database.

References: [T3 installation](https://github.com/pingdotgg/t3code/blob/main/docs/user/install.md), [remote access and pairing](https://github.com/pingdotgg/t3code/blob/main/docs/user/remote-access.md), [updating](https://github.com/pingdotgg/t3code/blob/main/docs/user/updating.md), [Dokploy Compose domains](https://docs.dokploy.com/docs/core/docker-compose/domains).
