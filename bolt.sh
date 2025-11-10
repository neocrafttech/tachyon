#!/usr/bin/env bash
set -e

setup_rust(){
    echo "[INFO] Checking Rust installation..."
    if command -v rustc >/dev/null 2>&1; then
        CURRENT_VERSION=$(rustc --version | awk '{print $2}')
        echo "[INFO] Found Rust version $CURRENT_VERSION"
        if [ "$CURRENT_VERSION" != "1.91.0" ]; then
            echo "[INFO] Updating Rust to 1.91.0..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.91.0
        else
            echo "[OK] Rust is already 1.91.0"
        fi
    else
        echo "[INFO] Rust not found. Installing Rust 1.91.0..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.91.0
    fi

    export PATH="$HOME/.cargo/bin:$PATH"
    rustc --version
    cargo --version

    echo "[INFO] Installing cargo-nextest if missing..."
    if ! cargo nextest --version >/dev/null 2>&1; then
        cargo install cargo-nextest
    fi
    cargo nextest --version
}

setup_cuda(){
    sudo apt install clang-format
	wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
	sudo dpkg -i cuda-keyring_1.1-1_all.deb
	sudo apt-get update
	sudo apt-get -y install cuda-toolkit-13-0
	sudo apt-get install -y cuda-drivers
}
setup() {
    setup_rust
    setup_cuda
}

check_rust() {
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

check_cpp() {
    echo "[INFO] Checking CPP"
    files=$(find . -type f \( -name "*.cpp" -o -name "*.cc" -o -name "*.cxx" -o -name "*.hpp" -o -name "*.h" -o -name "*.cu" -o -name "*.cuh" \))
    if [ -z "$files" ]; then
        echo "No C++ files to check."
        exit 0
    fi
    echo "$files"

    set +e
    unformatted=$(clang-format --dry-run --Werror $files 2>&1)
    status=$?
    set -e

    echo "$unformatted"
    if [ $status -ne 0 ]; then
        echo "[ERROR] Formatting errors found in the following files:"
        echo "$unformatted"
        echo ""
        echo "Run the following to fix formatting:"
        echo "clang-format -i $files"
        exit 1
    fi
    echo "[OK] CPP check passed!"
}

check() {
    check_rust
    check_cpp
}

build() {
    echo "[INFO] Building..."
    cargo build --release
    echo "[OK] Build completed!"
}

test() {
    echo "[INFO] Running CPU tests..."
    cargo nextest run --no-default-features || return 1

    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "[INFO] CUDA GPU detected — running GPU tests..."
        cargo nextest run --features gpu || return 1
    else
        echo "[WARN] CUDA GPU not detected — skipping GPU tests."
        echo "[INFO] (CUDA toolkit may be installed, but no working GPU/driver found)"
    fi

    echo "[OK] Testing completed!"
}

help() {
    echo "Usage: $0 [setup|check|build|deploy|all|help]"
    echo
    echo "Commands:"
    echo "  setup   - Install Rust and cargo-nextest"
    echo "  check   - Run cargo check, fmt, and clippy"
    echo "  build   - Only build the workspace (runs check first)"
    echo "  test    - Only run tests"
    echo "  all     - Run check, build, and test"
    echo "  help    - Show this help message"
}

main() {
    cmd="$1"
    case "$cmd" in
        setup)
            setup
            ;;
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
            setup
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
