# Makefile for mlp_loader bare-metal application
#
# Prerequisites:
#   - arm-none-eabi-gcc
#   - BSP generated from XSA (by run_ps_build.tcl)
#
# Usage:
#   make BSP_DIR=../vivado/bad_apple_build/fsbl/zynq_fsbl_bsp

CC = arm-none-eabi-gcc

# BSP directory (contains ps7_cortexa9_0/include and ps7_cortexa9_0/lib)
BSP_DIR ?= ../vivado/bad_apple_build/fsbl/zynq_fsbl_bsp

CFLAGS = -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard \
         -O2 -Wall \
         -I$(BSP_DIR)/ps7_cortexa9_0/include

LDFLAGS = -mcpu=cortex-a9 -mfpu=vfpv3 -mfloat-abi=hard \
          -Wl,-build-id=none \
          -specs=Xilinx.spec \
          -Wl,-T -Wl,lscript.ld \
          -L$(BSP_DIR)/ps7_cortexa9_0/lib

LIBS = -Wl,--start-group -lxilffs -lxil -lgcc -lc -Wl,--end-group

TARGET = mlp_loader.elf

all: $(TARGET)

$(TARGET): mlp_loader.c lscript.ld
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ mlp_loader.c $(LIBS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
