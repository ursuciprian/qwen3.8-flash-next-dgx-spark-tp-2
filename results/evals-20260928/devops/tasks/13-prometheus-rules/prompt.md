Write a Prometheus rules file for these needs:

- A recording rule `service:http_requests_5xx:ratio_rate5m`: for each `service`, the fraction of requests in the last 5 minutes that returned a 5xx status. The metric is the counter `http_requests_total` with labels `service`, `status` (e.g. "200", "503"), `method`, `instance`.
- An alert `HighErrorRate`: fires when that ratio stays above 5% for 10 minutes, with severity `page` and a summary annotation that names the service and shows the ratio as a percentage.
- An alert `PodCrashLooping`: fires when a container restarted more than 3 times in the last 15 minutes (metric `kube_pod_container_status_restarts_total` from kube-state-metrics, labels `namespace`, `pod`, `container`), severity `ticket`.

Return the file in one ```yaml fenced block. It must pass `promtool check rules`.
