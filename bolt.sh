#!/usr/bin/env bash
set -eo pipefail

setup_rust(){
    echo "[INFO] Installing build and test dependencies..."
    sudo apt-get update -y
    sudo apt-get install -y \
        build-essential ca-certificates curl wget pkg-config libssl-dev \
        protobuf-compiler libprotobuf-dev clang-format

    # Make an existing rustup installation available in a fresh shell.
    local cargo_home="${CARGO_HOME:-$HOME/.cargo}"
    export PATH="$cargo_home/bin:$PATH"

    echo "[INFO] Checking Rust and Cargo installation..."
    if command -v rustc >/dev/null 2>&1 && command -v cargo >/dev/null 2>&1; then
        local CURRENT_VERSION
        CURRENT_VERSION=$(rustc --version | awk '{print $2}')
        echo "[INFO] Found Rust version $CURRENT_VERSION"
        if [ "$CURRENT_VERSION" != "1.92.0" ]; then
            echo "[INFO] Updating Rust to 1.92.0..."
            curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0
        else
            echo "[OK] Rust is already 1.92.0"
        fi
    else
        echo "[INFO] Rust or Cargo not found. Installing Rust 1.92.0 and Cargo..."
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain 1.92.0
    fi
    if [ -f "$cargo_home/env" ]; then
        source "$cargo_home/env"
    fi
    rustc --version
    cargo --version

    echo "[INFO] Installing cargo-nextest if missing..."
    if ! cargo nextest --version >/dev/null 2>&1; then
        cargo install cargo-nextest --locked
    fi
    cargo nextest --version
}

setup_cuda() {
    local keyring
    keyring=$(mktemp --suffix=.deb)
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -o "$keyring"
    sudo dpkg -i "$keyring"
    rm -f "$keyring"
    sudo apt-get update
    # NVRTC, runtime headers/libraries, and compiler headers used by JIT kernels.
    # Avoid the full toolkit's IDE, profiling tools, and unrelated math libraries.
    sudo apt-get install -y --no-install-recommends \
        cuda-nvcc-13-0 cuda-cudart-dev-13-0 cuda-nvrtc-dev-13-0
    sudo ldconfig
    echo "[INFO] GPU execution also requires a working host NVIDIA driver (nvidia-smi)."
}

setup() {
    case "${1:-}" in
        ""|--cuda) ;;
        *) echo "Usage: $0 setup [--cuda]" >&2; return 1 ;;
    esac
    setup_rust
    if [ "${1:-}" = --cuda ]; then
        setup_cuda
    fi
}

format() {
    cargo +nightly fmt;
    clang-format -i ./tachyon/gpu/src/ffi/kernels/*
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

check_miri() {
    echo "[INFO] Running cargo miri..."
    rustup +nightly component add miri
    export MIRIFLAGS="-Zmiri-disable-isolation"
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        echo "[INFO] CUDA GPU detected — but miri not supported for FFI( unsupported operation: can't call foreign function `cudaMalloc` on OS `linux`)..."
       // cargo +nightly miri nextest run --features gpu --no-fail-fast --test-threads=4
    else
        echo "[INFO] Running CPU tests with miri..."
        cargo +nightly miri nextest run --no-default-features --no-fail-fast --test-threads=4
        echo "[WARN] CUDA GPU not detected — skipping GPU tests."
        echo "[INFO] (CUDA toolkit may be installed, but no working GPU/driver found)"
    fi

    echo "[OK] Miri checks passed!"
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

has_nvidia_hardware() {
    local device vendor device_class
    for device in /sys/bus/pci/devices/*; do
        [ -r "$device/vendor" ] && [ -r "$device/class" ] || continue
        read -r vendor < "$device/vendor"
        read -r device_class < "$device/class"
        # NVIDIA display/3D controllers, rather than other NVIDIA PCI devices.
        if [[ "$vendor" = 0x10de && "$device_class" = 0x03* ]]; then
            return 0
        fi
    done
    [ -e /dev/nvidia0 ]
}

test() {
    local mode="${1:-auto}" gpu_ready=false gpu_status
    case "$mode" in
        auto|--cpu|--gpu) ;;
        *) echo "Usage: $0 test [--cpu|--gpu]" >&2; return 1 ;;
    esac

    if [ "$mode" != --cpu ]; then
        if command -v nvidia-smi >/dev/null 2>&1; then
            if gpu_status=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>&1) && [ -n "$gpu_status" ]; then
                gpu_ready=true
            fi
        else
            gpu_status="nvidia-smi is missing."
        fi
        if [ "$gpu_ready" = false ] && { [ "$mode" = --gpu ] || has_nvidia_hardware; }; then
            echo "[ERROR] NVIDIA GPU tests requested or GPU hardware detected, but the NVIDIA driver is unavailable." >&2
            echo "[ERROR] $gpu_status" >&2
            if [ -d /sys/module/nouveau ]; then
                echo "[ERROR] nouveau is loaded; CUDA requires the NVIDIA compute driver." >&2
            fi
            echo "[INFO] Install a compatible host NVIDIA driver, reboot if required, and verify nvidia-smi." >&2
            echo "[INFO] Use './bolt.sh test --cpu' to explicitly run CPU tests." >&2
            return 1
        fi
    fi

    if [ "$gpu_ready" = true ]; then
        echo "[INFO] CUDA GPU detected — running GPU tests..."
        cargo nextest run --features gpu --test-threads=4 || return 1
    else
        echo "[INFO] Running CPU tests..."
        cargo nextest run --no-default-features --test-threads=4 || return 1
        if [ "$mode" = --cpu ]; then
            echo "[INFO] GPU tests skipped because --cpu was selected."
        else
            echo "[INFO] No accessible NVIDIA GPU found — GPU tests skipped."
        fi
    fi

    echo "[OK] Testing completed!"
}

coverage(){
    cargo llvm-cov --workspace --summary-only
}

help() {
    echo "Usage: $0 [setup|format|check|build|test|coverage|all|help]"
    echo
    echo "Commands:"
    echo "  setup   - Install Rust, Cargo, cargo-nextest, and CPU build/test libraries"
    echo "            Add --cuda to install CUDA build libraries (host driver required)"
    echo "  format   - Format code"
    echo "  check   - Run cargo check, fmt, and clippy"
    echo "  build   - Only build the workspace (runs check first)"
    echo "  test    - Run tests; use --cpu or --gpu to select the backend explicitly"
    echo "  miri    - Run miri checks"
    echo "  coverage - Run coverage"
    echo "  all     - Run check, build, and test"
    echo "  help    - Show this help message"
}

main() {
    cmd="$1"
    case "$cmd" in
        setup)
            setup "${2:-}"
            ;;
        format)
            format
            ;;
        check)
            check
            ;;
        build)
            build
            ;;
        test)
            test "${2:-auto}"
            ;;
        miri)
            check_miri
            ;;
        coverage)
            coverage
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
