This Deployment fails to apply on a Kubernetes 1.30 cluster, and after the first error is fixed the pods still never become Ready and the scheduler complains about the resources. List every problem (one line each), then return the corrected manifest in one ```yaml fenced block. The app listens on port 8080 and serves its health check at `/healthz`.

```yaml
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: payments-api
  namespace: payments
spec:
  replicas: 3
  selector:
    matchLabels:
      app: payments
  template:
    metadata:
      labels:
        app: payments-api
    spec:
      containers:
        - name: api
          image: ghcr.io/example/payments-api:1.8.2
          ports:
            - containerPort: "8080"
          env:
            - name: DB_PORT
              value: 5432
          readinessProbe:
            httpGet:
              path: /healthz
              port: 80
          resources:
            requests:
              cpu: 500m
              memory: 1Gi
            limits:
              cpu: 250m
              memory: 512Mi
```
