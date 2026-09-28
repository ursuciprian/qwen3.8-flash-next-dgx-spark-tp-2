Harden this Dockerfile for production. The app is a Node.js 20 service: `npm run build` compiles TypeScript to `dist/`, and the runtime command is `node dist/server.js` on port 3000. Installing dependencies needs a private npm token, which must not end up in any image layer or in the image history; pass it as a BuildKit secret with the id `npm_token`. Use a multi-stage build, a pinned (non-`latest`) base image, production-only dependencies in the final image, and a non-root user. The container needs no `curl`.

List the problems in the current file (one line each), then return the new Dockerfile in one ```dockerfile fenced block. It must pass `hadolint` with no warnings or errors.

```dockerfile
FROM node:latest
ADD https://example.com/config.tar.gz /app/
ENV NPM_TOKEN=npm_abc123secret
WORKDIR /app
COPY . .
RUN apt-get update && apt-get install -y curl
RUN npm install
RUN npm run build
EXPOSE 3000
CMD npm start
```
