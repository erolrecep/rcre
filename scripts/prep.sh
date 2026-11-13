#!/usr/bin/env bash
# ==============================================================================
# RCRE (Recep's Code Run Everywhere) - Environment Preparation Script
# Automatically detects OS, architecture, bitness (32-bit vs 64-bit), and
# installs required build toolchains and dependencies.
# ==============================================================================

set -e

# ANSI Color codes
BOLD="\033[1m"
GREEN="\033[32m"
YELLOW="\033[33m"
CYAN="\033[36m"
RED="\033[31m"
RESET="\033[0m"

PROJECT_NAME="rcre"
echo -e "${BOLD}${CYAN}========================================================================${RESET}"
echo -e "${BOLD}${CYAN}  PREPARING ENVIRONMENT: ${PROJECT_NAME} (High-Performance CV & ML Engine)${RESET}"
echo -e "${BOLD}${CYAN}========================================================================${RESET}"

# 1. Detect Operating System
OS="$(uname -s)"
DISTRO="Unknown"
DISTRO_ID="unknown"
DISTRO_VER=""
PKG_MGR="unknown"

if [ "$OS" = "Darwin" ]; then
    DISTRO="macOS"
    DISTRO_ID="macos"
    DISTRO_VER="$(sw_vers -productVersion 2>/dev/null || echo '')"
    if command -v brew >/dev/null 2>&1; then
        PKG_MGR="brew"
    else
        PKG_MGR="xcode"
    fi
elif [ "$OS" = "Linux" ]; then
    if [ -f /etc/os-release ]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        DISTRO="${PRETTY_NAME:-$NAME}"
        DISTRO_ID="${ID:-linux}"
        DISTRO_VER="${VERSION_ID:-}"
    fi

    if command -v apt-get >/dev/null 2>&1; then
        PKG_MGR="apt"
    elif command -v pacman >/dev/null 2>&1; then
        PKG_MGR="pacman"
    elif command -v dnf >/dev/null 2>&1; then
        PKG_MGR="dnf"
    elif command -v yum >/dev/null 2>&1; then
        PKG_MGR="yum"
    elif command -v apk >/dev/null 2>&1; then
        PKG_MGR="apk"
    elif command -v zypper >/dev/null 2>&1; then
        PKG_MGR="zypper"
    fi
elif [ "$OS" = "FreeBSD" ]; then
    DISTRO="FreeBSD"
    DISTRO_ID="freebsd"
    PKG_MGR="pkg"
fi

# 2. Detect Hardware Architecture & Bitness
ARCH="$(uname -m)"
BITNESS="$(getconf LONG_BIT 2>/dev/null || echo 64)"

if [ "$BITNESS" -eq 32 ] || [ "$ARCH" = "armv7l" ] || [ "$ARCH" = "armv6l" ] || [ "$ARCH" = "i686" ] || [ "$ARCH" = "i386" ]; then
    IS_32BIT=1
    BIT_DESC="32-bit"
else
    IS_32BIT=0
    BIT_DESC="64-bit"
fi

# 3. Detect Hardware Model
HW_MODEL="Generic ($ARCH)"
if [ -f /proc/device-tree/model ]; then
    HW_MODEL="$(tr -d '\0' < /proc/device-tree/model 2>/dev/null || echo "$HW_MODEL")"
elif [ "$OS" = "Darwin" ]; then
    HW_MODEL="$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Apple Silicon / Mac ($ARCH)")"
elif command -v lscpu >/dev/null 2>&1; then
    CPU_NAME="$(lscpu 2>/dev/null | grep 'Model name' | head -1 | cut -d: -f2 | sed -e 's/^[ \t]*//')"
    if [ -n "$CPU_NAME" ]; then
        HW_MODEL="$CPU_NAME ($ARCH)"
    fi
fi

# 4. Detect SIMD Vector Engine
SIMD_FEATURES=""
if [ "$OS" = "Darwin" ] && [ "$ARCH" = "arm64" ]; then
    SIMD_FEATURES="NEON, AdvSIMD, FP16, Apple AMX"
elif [ -f /proc/cpuinfo ]; then
    if grep -q -E "(neon|asimd)" /proc/cpuinfo; then
        SIMD_FEATURES="NEON / ASIMD"
    fi
    if grep -q "avx512" /proc/cpuinfo; then
        SIMD_FEATURES="${SIMD_FEATURES:+$SIMD_FEATURES, }AVX-512"
    elif grep -q "avx2" /proc/cpuinfo; then
        SIMD_FEATURES="${SIMD_FEATURES:+$SIMD_FEATURES, }AVX2"
    elif grep -q "avx" /proc/cpuinfo; then
        SIMD_FEATURES="${SIMD_FEATURES:+$SIMD_FEATURES, }AVX"
    fi
    if grep -q "fma" /proc/cpuinfo; then
        SIMD_FEATURES="${SIMD_FEATURES:+$SIMD_FEATURES, }FMA"
    fi
fi
[ -z "$SIMD_FEATURES" ] && SIMD_FEATURES="Scalar"

# 5. Output Platform Discovery Summary
echo -e "${BOLD}Platform Discovery Summary:${RESET}"
echo -e "  • ${BOLD}Operating System:${RESET}  $DISTRO ($OS ${DISTRO_VER})"
echo -e "  • ${BOLD}Architecture:${RESET}      $ARCH ($BIT_DESC address space)"
echo -e "  • ${BOLD}Hardware Node:${RESET}     $HW_MODEL"
echo -e "  • ${BOLD}Vector SIMD:${RESET}       $SIMD_FEATURES"
echo -e "  • ${BOLD}Package Manager:${RESET}   $PKG_MGR"
echo ""

# 6. Privilege resolution for package installations
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    fi
fi

# 7. Install Required Packages based on Platform
echo -e "${BOLD}Checking & Installing Required Build Software...${RESET}"

case "$PKG_MGR" in
    apt)
        DEB_PKGS=("build-essential" "gcc" "clang" "make" "git" "pkg-config" "python3")
        # Video4Linux header for Linux webcam capture:
        DEB_PKGS+=("libv4l-dev")
        
        MISSING=()
        for pkg in "${DEB_PKGS[@]}"; do
            if ! dpkg -s "$pkg" >/dev/null 2>&1; then
                MISSING+=("$pkg")
            fi
        done

        if [ ${#MISSING[@]} -gt 0 ]; then
            echo -e "${YELLOW}[->] Installing missing packages:${RESET} ${MISSING[*]}"
            $SUDO apt-get update -y
            $SUDO apt-get install -y "${MISSING[@]}"
        else
            echo -e "${GREEN}[✓] All apt system packages already installed.${RESET}"
        fi
        ;;

    pacman)
        ARCH_PKGS=("base-devel" "clang" "gcc" "git" "pkgconf" "python" "v4l-utils")
        MISSING=()
        for pkg in "${ARCH_PKGS[@]}"; do
            if ! pacman -Qi "$pkg" >/dev/null 2>&1; then
                MISSING+=("$pkg")
            fi
        done

        if [ ${#MISSING[@]} -gt 0 ]; then
            echo -e "${YELLOW}[->] Installing missing packages via pacman:${RESET} ${MISSING[*]}"
            $SUDO pacman -Sy --noconfirm "${MISSING[@]}"
        else
            echo -e "${GREEN}[✓] All pacman system packages already installed.${RESET}"
        fi
        ;;

    dnf|yum)
        MGR="$PKG_MGR"
        FED_PKGS=("gcc" "gcc-c++" "clang" "make" "git" "pkgconf-pkg-config" "python3" "libv4l-devel")
        MISSING=()
        for pkg in "${FED_PKGS[@]}"; do
            if ! rpm -q "$pkg" >/dev/null 2>&1; then
                MISSING+=("$pkg")
            fi
        done

        if [ ${#MISSING[@]} -gt 0 ]; then
            echo -e "${YELLOW}[->] Installing missing packages via $MGR:${RESET} ${MISSING[*]}"
            $SUDO $MGR install -y "${MISSING[@]}"
        else
            echo -e "${GREEN}[✓] All $MGR system packages already installed.${RESET}"
        fi
        ;;

    apk)
        ALP_PKGS=("build-base" "clang" "gcc" "make" "git" "pkgconf" "python3" "v4l-utils-dev")
        MISSING=()
        for pkg in "${ALP_PKGS[@]}"; do
            if ! apk info -e "$pkg" >/dev/null 2>&1; then
                MISSING+=("$pkg")
            fi
        done

        if [ ${#MISSING[@]} -gt 0 ]; then
            echo -e "${YELLOW}[->] Installing missing packages via apk:${RESET} ${MISSING[*]}"
            $SUDO apk add "${MISSING[@]}"
        else
            echo -e "${GREEN}[✓] All apk system packages already installed.${RESET}"
        fi
        ;;

    brew)
        if ! command -v xcode-select >/dev/null 2>&1 || ! xcode-select -p >/dev/null 2>&1; then
            echo -e "${YELLOW}[->] Installing Xcode Command Line Tools...${RESET}"
            xcode-select --install || true
        else
            echo -e "${GREEN}[✓] Apple Xcode Command Line Tools present.${RESET}"
        fi

        if ! command -v python3 >/dev/null 2>&1; then
            echo -e "${YELLOW}[->] Installing python3 via Homebrew...${RESET}"
            brew install python3
        else
            echo -e "${GREEN}[✓] Python 3 present ($(python3 --version)).${RESET}"
        fi
        ;;

    xcode)
        if ! xcode-select -p >/dev/null 2>&1; then
            echo -e "${YELLOW}[->] Installing Apple Command Line Tools...${RESET}"
            xcode-select --install
        else
            echo -e "${GREEN}[✓] Apple Command Line Tools active.${RESET}"
        fi
        ;;

    *)
        echo -e "${YELLOW}[!] Unknown package manager. Please ensure gcc/clang, make, git, and python3 are installed.${RESET}"
        ;;
esac

# 8. Architecture-specific tuning guidance
echo ""
echo -e "${BOLD}Architecture & Bitness Configuration:${RESET}"
if [ "$IS_32BIT" -eq 1 ]; then
    echo -e "  ${YELLOW}[32-bit Mode Detected]${RESET}"
    echo -e "  • Target: ${BOLD}$ARCH${RESET} (32-bit addressing)"
    if [ "$ARCH" = "armv7l" ] || [ "$ARCH" = "armv6l" ]; then
        echo -e "  • Recommended Flags: ${CYAN}-march=armv7-a -mfpu=neon-vfpv4 -mfloat-abi=hard -O3${RESET}"
        echo -e "  • Memory Optimization: Use 64MB–128MB Arena scratchpads for 1GB RAM constraints."
    fi
else
    echo -e "  ${GREEN}[64-bit Mode Detected]${RESET}"
    echo -e "  • Target: ${BOLD}$ARCH${RESET} (64-bit addressing)"
    if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
        echo -e "  • Hardware Vectorization: ARMv8-A/ARMv9 AdvSIMD (NEON) natively active."
    elif [ "$ARCH" = "x86_64" ]; then
        echo -e "  • Hardware Vectorization: AVX2/FMA dispatch active."
    fi
fi

# 9. Verify C23 Compiler Toolchain
echo ""
echo -e "${BOLD}Testing C23 Compiler Toolchain Support...${RESET}"
C23_OK=0
DETECTED_CC=""
DETECTED_FLAG=""

TMP_SRC="$(mktemp /tmp/c23_test_XXXXXX.c 2>/dev/null || echo "/tmp/c23_test.c")"
cat << 'EOF' > "$TMP_SRC"
#include <stddef.h>
int main(void) {
    void *ptr = nullptr;
    return (ptr == NULL) ? 0 : 1;
}
EOF

for compiler in "${CC:-}" clang gcc cc; do
    [ -z "$compiler" ] && continue
    if command -v "$compiler" >/dev/null 2>&1; then
        for flag in -std=c2x -std=c23; do
            if "$compiler" "$flag" "$TMP_SRC" -o /tmp/c23_test_bin >/dev/null 2>&1; then
                DETECTED_CC="$compiler"
                DETECTED_FLAG="$flag"
                C23_OK=1
                break 2
            fi
        done
    fi
done
rm -f "$TMP_SRC" /tmp/c23_test_bin 2>/dev/null || true

if [ "$C23_OK" -eq 1 ]; then
    CC_VER="$("$DETECTED_CC" --version 2>&1 | head -n 1)"
    echo -e "${GREEN}[✓] C23 Standard Verified:${RESET} $DETECTED_CC ($DETECTED_FLAG)"
    echo -e "    Version: $CC_VER"
else
    echo -e "${YELLOW}[!] Warning: No C23 compiler detected supporting -std=c2x or -std=c23.${RESET}"
    echo -e "    Please install Clang 16+ or GCC 13+."
fi

# 10. Summary
echo ""
echo -e "${BOLD}${GREEN}========================================================================${RESET}"
echo -e "${BOLD}${GREEN}  ✅ ENVIRONMENT PREPARATION COMPLETE FOR: ${PROJECT_NAME}${RESET}"
echo -e "${BOLD}${GREEN}========================================================================${RESET}"
echo -e "You can now build and test the project:"
echo -e "  ${BOLD}make all${RESET}   - Build release libraries and examples"
echo -e "  ${BOLD}make test${RESET}  - Run the complete verification test battery"
echo ""
