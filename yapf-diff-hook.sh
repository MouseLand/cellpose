#!/bin/bash
git diff -U0 --no-color --relative HEAD^ | yapf-diff -i --verbose --style "google"
