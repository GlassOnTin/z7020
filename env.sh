#!/bin/bash
# Vivado 2025.2 environment for Zynq 7020 development
# Source this file: source ~/Code/z7020/env.sh

VIVADO_BASE="/opt/Xilinx/2025.2/Vivado"

if [ -f "$VIVADO_BASE/settings64.sh" ]; then
    source "$VIVADO_BASE/settings64.sh"
    echo "Vivado 2025.2 environment loaded"
else
    echo "ERROR: Vivado not found at $VIVADO_BASE"
    echo "Check installation path"
fi

# Verify
if command -v vivado &>/dev/null; then
    echo "vivado: $(which vivado)"
else
    echo "WARNING: vivado not found in PATH"
    echo "Check install location and update VIVADO_BASE in this script"
fi
