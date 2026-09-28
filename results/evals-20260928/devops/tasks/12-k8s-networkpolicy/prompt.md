Write Kubernetes NetworkPolicies (networking.k8s.io/v1) for the namespace `shop`:

1. Default deny: no pod in `shop` may receive or send any traffic unless another policy allows it.
2. Pods labelled `app=checkout` accept ingress on TCP 8080 only from pods in the namespace `ingress-nginx`.
3. Pods labelled `app=checkout` may connect to TCP 5432 only on pods labelled `app=postgres` in the namespace `data` (only those pods, not every pod in `data`, and not `app=postgres` pods in other namespaces).
4. Pods labelled `app=checkout` may resolve DNS through the kube-dns pods (`k8s-app=kube-dns`) in `kube-system`, on both UDP and TCP 53.

Select namespaces by the automatic `kubernetes.io/metadata.name` label. Return all policies in one ```yaml fenced block, as separate YAML documents separated by `---`. They must pass `kubeconform -strict`.
