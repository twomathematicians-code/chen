#!/bin/sh
# CHEN container entrypoint — wraps the `chen` CLI.
# Passes through all arguments to the CLI.

set -e

# If the first arg is `serve`, run the API server.
# Otherwise, pass through to the CLI.
case "$1" in
  serve)
    shift
    exec chen serve "$@"
    ;;
  *)
    exec chen "$@"
    ;;
esac
