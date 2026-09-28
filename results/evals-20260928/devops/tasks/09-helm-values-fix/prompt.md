The `webapp` Helm chart below renders broken manifests with this `values.yaml`: the deployed image tag is wrong (it must be exactly `2.40`), the Service and the resources are invalid, and the Ingress never routes traffic to `webapp.example.com`. The templates are correct; fix only the values. List each problem in one line, then return the corrected `values.yaml` in one ```yaml fenced block. The rendered output must pass `kubeconform -strict`. The Service should listen on port 80, CPU and memory limits should be 500m and 256Mi, and requests should be 250m and 128Mi.

`values.yaml`:
```yaml
replicaCount: 3
image:
  repository: ghcr.io/acme/webapp
  tag: 2.40
service:
  port: http
  targetPort: 8080
resources:
  limits:
  cpu: 500m
  memory: 256Mi
ingress:
  enabled: true
  className: nginx
  hosts:
    host: webapp.example.com
    paths:
      - path: /
        pathType: prefix
```

`templates/deployment.yaml`:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {{ .Release.Name }}
spec:
  replicas: {{ .Values.replicaCount }}
  selector:
    matchLabels:
      app: {{ .Release.Name }}
  template:
    metadata:
      labels:
        app: {{ .Release.Name }}
    spec:
      containers:
        - name: web
          image: "{{ .Values.image.repository }}:{{ .Values.image.tag }}"
          ports:
            - name: http
              containerPort: {{ .Values.service.targetPort }}
          resources:
            {{- toYaml .Values.resources | nindent 12 }}
```

`templates/service.yaml`:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: {{ .Release.Name }}
spec:
  selector:
    app: {{ .Release.Name }}
  ports:
    - name: http
      port: {{ .Values.service.port }}
      targetPort: {{ .Values.service.targetPort }}
```

`templates/ingress.yaml`:
```yaml
{{- if .Values.ingress.enabled }}
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: {{ .Release.Name }}
spec:
  ingressClassName: {{ .Values.ingress.className }}
  rules:
    {{- range .Values.ingress.hosts }}
    - host: {{ .host }}
      http:
        paths:
          {{- range .paths }}
          - path: {{ .path }}
            pathType: {{ .pathType }}
            backend:
              service:
                name: {{ $.Release.Name }}
                port:
                  number: {{ $.Values.service.port }}
          {{- end }}
    {{- end }}
{{- end }}
```
