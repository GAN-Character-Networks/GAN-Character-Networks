#!/bin/bash
set -e -u -o pipefail

git ls-files vroom | grep -e "\.py$" | xargs pytest --doctest-modules