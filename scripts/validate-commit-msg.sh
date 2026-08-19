#!/bin/sh
set -eu

MODULES='ingest|retrieval|serving|runtime|storage|provenance|evaluation|profiling|docs|tooling'
TYPES='feat|fix|perf|refactor|docs|test|build|ci|chore'
PATTERN='^\[('"$MODULES"')\] ('"$TYPES"')(!)?: [^[:space:]].*$'

usage() {
    printf '%s\n' \
        'Usage:' \
        '  validate-commit-msg.sh --stdin' \
        '  validate-commit-msg.sh --file PATH'
}

check_one() {
    msg=$1
    if ! printf '%s\n' "$msg" | grep -Eq "$PATTERN"; then
        printf '%s\n' "Invalid commit message: $msg" >&2
        printf '%s\n' 'Expected: [<module>] <type>: <description>' >&2
        return 1
    fi
}

status=0

case "${1-}" in
    --stdin)
        if [ "$#" -ne 1 ]; then
            usage >&2
            exit 2
        fi

        while IFS= read -r line || [ -n "$line" ]; do
            [ -z "$line" ] && continue

            if ! check_one "$line"; then
                status=1
            fi
        done
        ;;

    --file)
        if [ "$#" -ne 2 ]; then
            usage >&2
            exit 2
        fi

        if [ ! -r "$2" ]; then
            printf '%s\n' "Cannot read commit message file: $2" >&2
            exit 2
        fi

        first_line=$(sed -n '1p' "$2")

        if ! check_one "$first_line"; then
            status=1
        fi
        ;;

    --help)
        usage
        exit 0
        ;;

    *)
        usage >&2

        exit 2
        ;;
esac

exit "$status"
