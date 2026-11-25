# Releasing to the Hermes plugin catalog

The catalog entry lives in `NousResearch/hermes-agent` at
`plugin-catalog/neural-mesh.yaml`. It pins **this** repo by commit:

```yaml
sha: <commit in D0xedDevi0/hermes-neural-mesh>
version: "<must equal plugin.yaml version>"
image: https://raw.githubusercontent.com/D0xedDevi0/hermes-neural-mesh/<sha>/…
```

Three identifiers have to agree or the entry is wrong in a way that looks fine:
the catalogue `sha`, `plugin.yaml`'s `version`, and
`neural_mesh/__version__` (a test asserts the last two match). Screenshot URLs
are pinned to the same `sha`, so they 404 silently if you bump one and not the
other.

## Steps

1. `python3 scripts/revendor.py <upstream-ref>` — never hand-copy `neural_mesh/`.
2. Bump `plugin.yaml` + `neural_mesh/__version__`; update `PROVENANCE.md` (commit,
   version, and the "three names for one state" paragraph).
3. `python3 scripts/trust_check.py` and the full test suite.
4. Push. Then point `plugin-catalog/neural-mesh.yaml` at the new `sha` and
   `version`, refresh every raw URL, and add any new tool to
   `capabilities.provides_tools` — a tool declared there but not registered, or
   registered but not declared, is a review rejection.
5. Assert the declaration against the code before opening the PR:

   ```python
   tools = {t["name"] for t in Provider().get_tool_schemas()}
   assert tools == set(catalog["capabilities"]["provides_tools"])
   ```

6. Open the PR against `NousResearch/hermes-agent:main` from a branch on the fork.

## Credentials — the part that actually blocks releases

The token in `.secrets/github-d0xeddev.env` is a **fine-grained** PAT. It can
push to, and open PRs on, repositories it is scoped to (verified: creating a PR
on `D0xedDevi0/hermes-agent` returns 201). It cannot open a PR against
`NousResearch/hermes-agent` unless *that* repo is in the token's selected
repositories with **Pull requests: write**.

The symptom is exact and easy to misread as "the token is read-only":

```json
403 {"message":"Resource not accessible by personal access token"}
```

`GET /repos/NousResearch/hermes-agent` still returns 200 with
`"pull": true` — public read is granted regardless — so a read test proves
nothing about the write path. Test the write path itself:

```bash
# capability probe that leaves no trace: create and immediately close a draft PR
POST /repos/D0xedDevi0/hermes-agent/pulls   # 201 => Pull requests: write is present
```

If the fork probe is 201 and the upstream call is 403, the only fix is token
scope (repo selection), not a code or branch problem. Do not work around it by
committing a second credential or by disabling a protection.

**Fallback that needs one click, not a token change:** push the branch to the
fork and open

```
https://github.com/NousResearch/hermes-agent/compare/main...D0xedDevi0:<branch>?expand=1
```

with the prepared body. The branch and body are the deliverable; the click is a
permission the token does not have.

## Never

- Never commit a token, a `.env`, or a credential file to either repo.
- Never store the PR body's secrets — the body is public prose, not a log.
