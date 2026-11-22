#!/usr/bin/env bash
# Report one deploy job to LGTM over OTLP/HTTP JSON: one Loki log record per Deployment and one Tempo span `deploy`.
#
# Taken from evilflowers-analyzer-service's scripts/otlp-deploy.sh (the unified worker-kit nightly,
# EvilFlowersCatalog/evilflowers-worker#13) in place of evilflowers-actions' script, which sends no evilflowers.*
# attributes yet. Contract: knowledge base, Proposals/CI-Deploy-Unification §"What it sends" (service.name=github-actions,
# event.name=deployment, vcs.* / cicd.* / oci.* attributes). Local additions, in the evilflowers.* namespace:
# evilflowers.deploy.trigger (push / nightly / manual), evilflowers.worker.revision (the evilflowers-worker commit in
# the image), evilflowers.worker.ref (the kit ref it was resolved from: master or a release tag,
# scripts/resolve-worker.sh), evilflowers.deployments.
#
# The receiver is the node's alloy-node, $ELVIRA_OTLP_HTTP_ENDPOINT on the runner pod (roles/github_runner). Unset or
# unreachable: a warning, exit 0. Telemetry never fails a deploy. curl only: the runner image has no OTel SDK.
#
# Input (env): NAMESPACE ENVIRONMENT DEPLOYMENTS DIGEST DEPLOY_STATUS (succeeded|failed) DEPLOY_START_NS
#              WORKER_REVISION WORKER_REF TRIGGER WORKFLOW_NAME, and GitHub's GITHUB_* variables.
set -uo pipefail

warn() { echo "::warning::deploy event not sent: $1"; exit 0; }

endpoint="${ELVIRA_OTLP_HTTP_ENDPOINT:-}"
[ -n "$endpoint" ] || warn "ELVIRA_OTLP_HTTP_ENDPOINT is not set on this runner (playbooks/github-runner.yml not applied?)"
endpoint="${endpoint%/}"

status="${DEPLOY_STATUS:-failed}"
now_ns=$(date +%s%N)
start_ns="${DEPLOY_START_NS:-$now_ns}"
trace_id=$(od -An -N16 -tx1 /dev/urandom | tr -d ' \n')
span_id=$(od -An -N8 -tx1 /dev/urandom | tr -d ' \n')
image="ghcr.io/$(printf '%s' "$GITHUB_REPOSITORY" | tr '[:upper:]' '[:lower:]')"
run_url="${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}/actions/runs/${GITHUB_RUN_ID}/attempts/${GITHUB_RUN_ATTEMPT:-1}"

# JSON string literal (the values are names, URLs and hex; escape \ and " anyway).
js() { local v="${1//\\/\\\\}"; printf '"%s"' "${v//\"/\\\"}"; }
attr() { printf '{"key":%s,"value":{"stringValue":%s}}' "$(js "$1")" "$(js "$2")"; }

resource=$(printf '{"attributes":[%s,%s,%s,%s]}' \
  "$(attr service.name github-actions)" \
  "$(attr service.namespace ci)" \
  "$(attr deployment.environment.name "${ENVIRONMENT:-}")" \
  "$(attr k8s.namespace.name "${NAMESPACE:-}")")
scope='{"name":"evilflowers-equation-service/deploy"}'

common=$(printf '%s,' \
  "$(attr deployment.status "$status")" \
  "$(attr vcs.repository.url.full "${GITHUB_SERVER_URL}/${GITHUB_REPOSITORY}")" \
  "$(attr vcs.ref.head.name "${GITHUB_REF_NAME}")" \
  "$(attr vcs.ref.head.revision "${GITHUB_SHA}")" \
  "$(attr cicd.pipeline.name "${WORKFLOW_NAME:-${GITHUB_WORKFLOW:-}}")" \
  "$(attr cicd.pipeline.run.id "${GITHUB_RUN_ID}")" \
  "$(attr cicd.pipeline.run.url.full "$run_url")" \
  "$(attr oci.manifest.digest "${DIGEST:-}")" \
  "$(attr container.image.name "$image")" \
  "$(attr evilflowers.deploy.trigger "${TRIGGER:-${GITHUB_EVENT_NAME}}")" \
  "$(attr evilflowers.worker.revision "${WORKER_REVISION:-}")" \
  "$(attr evilflowers.worker.ref "${WORKER_REF:-}")")
common="${common%,}"

if [ "$status" = "failed" ]; then severity=17; level=error; span_status='{"code":2,"message":"rollout or digest check failed"}'
else severity=9; level=info; span_status='{}'; fi

records=""
for deployment in ${DEPLOYMENTS:-}; do
  records+=$(printf '{"timeUnixNano":"%s","observedTimeUnixNano":"%s","severityNumber":%d,"severityText":"%s","body":{"stringValue":%s},"attributes":[%s,%s,%s],"traceId":"%s","spanId":"%s"},' \
    "$now_ns" "$now_ns" "$severity" "$level" "$(js "deployment $deployment $status")" \
    "$(attr event.name deployment)" "$(attr k8s.deployment.name "$deployment")" "$common" "$trace_id" "$span_id")
done
records="${records%,}"

logs=$(printf '{"resourceLogs":[{"resource":%s,"scopeLogs":[{"scope":%s,"logRecords":[%s]}]}]}' "$resource" "$scope" "$records")
spans=$(printf '{"resourceSpans":[{"resource":%s,"scopeSpans":[{"scope":%s,"spans":[{"traceId":"%s","spanId":"%s","name":"deploy","kind":1,"startTimeUnixNano":"%s","endTimeUnixNano":"%s","attributes":[%s,%s],"status":%s}]}]}]}' \
  "$resource" "$scope" "$trace_id" "$span_id" "$start_ns" "$now_ns" "$common" "$(attr evilflowers.deployments "${DEPLOYMENTS:-}")" "$span_status")

post() {
  curl -fsS --max-time 10 -H 'Content-Type: application/json' --data-binary @- "$endpoint/$1" > /dev/null
}
printf '%s' "$logs" | post v1/logs || warn "POST $endpoint/v1/logs failed"
printf '%s' "$spans" | post v1/traces || warn "POST $endpoint/v1/traces failed"

echo "deploy event sent: status=$status trace_id=$trace_id"
echo "DEPLOY_TRACE_ID=$trace_id" >> "${GITHUB_ENV:-/dev/null}"
