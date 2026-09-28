This GitHub Actions workflow (`.github/workflows/ci.yml`) is broken: it does not even parse, and after that it has several logic and security bugs. List each problem in one line, then return the corrected workflow in one ```yaml fenced block. It must pass `actionlint` with no findings. Tests should run on Python 3.10, 3.11 and 3.12; deploy runs only on pushes to `main`, and only when the test job passed.

```yaml
name: ci
on:
  push:
    branches: main
  pull_request
jobs:
  test:
    strategy:
      matrix:
        python: [3.10, 3.11, 3.12]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - run: pip install -r requirements.txt && pytest
      - name: version
        id: ver
        run: echo "::set-output name=version::$(cat VERSION)"
      - run: echo "Version is ${{ steps.ver.outputs.ver }}"
  deploy:
    needs: test
    if: github.ref == 'refs/heads/main' && secrets.DEPLOY_TOKEN != ''
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: ./deploy.sh ${{ secrets.DEPLOY_TOKEN }}
```
