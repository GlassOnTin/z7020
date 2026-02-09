# Vivado 2025.2 Installation Notes

## System
- **OS**: Ubuntu 25.10 (Questing Quokka), x86_64, kernel 6.17.0-8-generic
- **Date**: 2026-02-08

## Installer
- **File**: `FPGAs_AdaptiveSoCs_Unified_SDI_2025.2_1114_2157_Lin64.bin` (web installer, 347 MB)
- **MD5**: `abe838aa2e2d3d9b10fea94165e9a303` (verified)
- **Source**: https://www.xilinx.com/support/download.html (requires AMD account)

### Makeself extraction bug

The makeself 2.1.5 self-extractor bundled in the .bin has a race condition in its
piped `dd` commands (`MS_dd` function). Both the internal MD5 and CRC integrity
checks produce different wrong results each run. The outer file MD5 is correct.

**Workaround**: Extract payload manually at byte offset 9686:

```bash
mkdir -p /tmp/vivado_extract
tail -c +9687 FPGAs_AdaptiveSoCs_Unified_SDI_2025.2_1114_2157_Lin64.bin | gzip -cd | tar xf - -C /tmp/vivado_extract
```

Alternatively, split the header, patch checksums to skip values, and recombine:

```bash
dd if=installer.bin bs=9686 count=1 status=none > header.bin
dd if=installer.bin bs=9686 skip=1 status=none > payload.bin
sed -i 's/MD5="ff6fa3910cb98a34ddaeb50dac575efe"/MD5="00000000000000000000000000000000"/' header.bin
sed -i 's/CRCsum="1155118907"/CRCsum="0000000000"/' header.bin
cat header.bin payload.bin > patched_installer.bin
chmod +x patched_installer.bin
```

## Download phase (web installer)

The extracted `xsetup` was launched as a GUI (`DISPLAY=:0`) since batch mode
AuthTokenGen requires an interactive console for AMD account credentials.

The web installer downloaded an 84.18 GB offline image to `~/Downloads/2025.2/`.
Download stalled near completion (~360 MB remaining) at 11 kB/sec due to CDN
throttling. Toggling a VPN connection reset the TCP session and restored ~9 MB/sec.

## Installation

Batch install from the offline image:

```bash
# Generate config (interactive — select 2=Vivado, 1=Vivado ML Standard)
echo -e "2\n1\n" | ~/Downloads/2025.2/xsetup -b ConfigGen

# Edit ~/.Xilinx/install_config.txt:
#   Destination=/opt/Xilinx
#   Zynq-7000 All Programmable SoC:1  (all other devices :0)
#   Vitis Model Composer:0
#   DocNav:1

# Create install directory
sudo mkdir -p /opt/Xilinx && sudo chown ian:ian /opt/Xilinx

# Run batch install
~/Downloads/2025.2/xsetup --agree XilinxEULA,3rdPartyEULA --batch Install \
    --config ~/.Xilinx/install_config.txt
```

Install completed in ~4 minutes (extraction from local offline image).

### Output
```
Tool installation completed. To run the tool successfully, please run the script
"installLibs.sh" under /opt/Xilinx/2025.2/Vivado/scripts to install the necessary
OS packages, which requires the root privilege.
```

## Post-install

### OS library dependencies
```bash
sudo /opt/Xilinx/2025.2/Vivado/scripts/installLibs.sh
```
Installed `libgpg-error-dev`, `libgcrypt20-dev`, `libsecret-1-dev` and dependencies.

### Compatibility symlinks (Ubuntu 25.10 ships v6 only)
```bash
sudo ln -sf /usr/lib/x86_64-linux-gnu/libtinfo.so.6 /usr/lib/x86_64-linux-gnu/libtinfo.so.5
sudo ln -sf /usr/lib/x86_64-linux-gnu/libncurses.so.6 /usr/lib/x86_64-linux-gnu/libncurses.so.5
```

### Cable drivers
```bash
sudo /opt/Xilinx/2025.2/data/xicom/cable_drivers/lin64/install_script/install_drivers/install_drivers
```
Output:
```
Successfully installed Digilent Cable Drivers
INFO: Digilent Return code = 0
INFO: Xilinx Return code = 0
INFO: Xilinx FTDI Return code = 0
INFO: Driver installation successful.
```

Installed udev rules:
- `/etc/udev/rules.d/52-xilinx-digilent-usb.rules`
- `/etc/udev/rules.d/52-xilinx-ftdi-usb.rules`
- `/etc/udev/rules.d/52-xilinx-pcusb.rules`

## Verification

```
$ source /opt/Xilinx/2025.2/Vivado/settings64.sh
$ vivado -version
vivado v2025.2 (64-bit)
Tool Version Limit: 2025.11
SW Build 6299465 on Fri Nov 14 12:34:56 MST 2025
IP Build 6300035 on Fri Nov 14 10:48:45 MST 2025
SharedData Build 6298862 on Thu Nov 13 04:50:51 MST 2025
```

## Installed layout

| Path | Size | Contents |
|------|------|----------|
| `/opt/Xilinx/2025.2/Vivado/` | ~53 GB | Vivado ML Standard + Zynq-7000 device support |
| `/opt/Xilinx/2025.2/Vivado/bin/vivado` | | Main executable |
| `/opt/Xilinx/2025.2/Vivado/bin/hw_server` | | Hardware server for JTAG |
| `/opt/Xilinx/2025.2/Vivado/settings64.sh` | | Environment setup script |

## Usage

```bash
source ~/Code/z7020/env.sh        # Load Vivado into PATH
./fpga-jtag.sh attach             # Connect Zynq 7020 JTAG via usbip
vivado                            # Launch Vivado GUI
```

## Cleanup

The 86 GB offline download image can be removed after verifying the installation:
```bash
rm -rf ~/Downloads/2025.2/
```
