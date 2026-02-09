#!/bin/bash
# Manage usbip JTAG connection to Zynq 7020 board via Raspberry Pi 5
# Pi server: 192.168.0.234:3240, Device: 1-1 (Digilent FT2232H)

PI_HOST="192.168.0.234"
PI_PORT="3240"
DEVICE_BUSID="1-1"

case "${1:-status}" in
    attach)
        echo "Loading vhci-hcd module..."
        sudo -A modprobe vhci-hcd
        echo "Attaching FPGA JTAG from $PI_HOST..."
        sudo -A usbip attach -r "$PI_HOST" -b "$DEVICE_BUSID"
        sleep 1
        # Show result
        usbip port 2>/dev/null
        echo "---"
        lsusb | grep -i -E 'digilent|ftdi|0403'
        ;;
    detach)
        echo "Detaching FPGA JTAG..."
        # Find the port number
        PORT=$(usbip port 2>/dev/null | grep -oP 'Port \K\d+' | head -1)
        if [ -n "$PORT" ]; then
            sudo -A usbip detach -p "$PORT"
            echo "Detached port $PORT"
        else
            echo "No attached devices found"
        fi
        ;;
    list)
        echo "Available devices on $PI_HOST:"
        usbip list -r "$PI_HOST"
        ;;
    status|"")
        echo "=== usbip attached devices ==="
        usbip port 2>/dev/null || echo "(none)"
        echo ""
        echo "=== Local USB (Digilent/FTDI) ==="
        lsusb | grep -i -E 'digilent|ftdi|0403' || echo "(none found)"
        ;;
    *)
        echo "Usage: $0 {attach|detach|list|status}"
        exit 1
        ;;
esac
