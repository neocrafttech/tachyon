#!/usr/bin/env bash
set -e

check() {
    echo "[INFO] Running cargo check..."
    cargo check
    echo "[OK] Cargo check passed!"

    echo "[INFO] Checking code formatting..."
    cargo fmt -- --check
    echo "[OK] Code is properly formatted!"

    echo "[INFO] Running clippy lints..."
    cargo clippy -- -D warnings
    echo "[OK] Clippy checks passed!"
}

build() {
    echo "[INFO] Building..."
    cargo build --release
    echo "[OK] Build completed!"
}

test() {
    echo "[INFO] Testing workspace..."
    cargo nextest run
    echo "[OK] Testing completed!"
}

help() {
    echo "Usage: $0 [check|build|deploy|all|help]"
    echo
    echo "Commands:"
    echo "  check   - Run cargo check, fmt, and clippy"
    echo "  build   - Only build the workspace (runs check first)"
    echo "  test    - Only run tests"
    echo "  all     - Run check, build, and test"
    echo "  help    - Show this help message"
}

main() {
    cmd="$1"
    case "$cmd" in
        check)
            check
            ;;
        build)
            build
            ;;
        test)
            test
            ;;
        all)
            check
            build
            deploy
            ;;
        help|""|*)
            help
            ;;
    esac
}

main "$@"
