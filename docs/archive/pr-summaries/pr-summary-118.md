# PR Summary — Issue #118: Semgrep container image pinned by tag and digest

## Summary

Closes #118.

`.github/workflows/semgrep.yml` pinned `semgrep/semgrep` by bare digest, so
Renovate and Dependabot had no tag to key version bumps on. The image is now
`semgrep/semgrep:1.86.0@sha256:a9ea2d5621c29d815d90c2a3b2f9571da8972ef4ff855c9e4902681730240e35`.
Docker pulls by the digest, so the scanner build is unchanged. The comments and
bump protocol now say the tag moves in lockstep with the digest.

## Evidence

- The Docker Hub tag API (`hub.docker.com/v2/repositories/semgrep/semgrep/tags/1.86.0`)
  returns digest `sha256:a9ea2d56…0e35` for `1.86.0`.
- The registry manifest `HEAD` (`registry-1.docker.io/v2/semgrep/semgrep/manifests/1.86.0`)
  returns `docker-content-digest: sha256:a9ea2d56…0e35`.
- Both lookups ran in this run and match the existing pin, so this change adds
  the tag without altering the digest.

## Test Plan

- [x] Confirmed the `1.86.0` tag resolves to the pinned digest via both Docker
      Hub endpoints.
- [x] `./quality.sh` passes.
- [ ] The Semgrep workflow on this PR pulls the image and scans as before.
