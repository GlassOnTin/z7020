# Makefile for Mandelbrot RTL simulation with Icarus Verilog
#
# Targets:
#   make test_mul     - Run fixed-point multiplier tests
#   make test_neuron  - Run neuron core tests
#   make test_all     - Run all tests
#   make ref          - Run Python reference implementation
#   make clean        - Remove generated files

IVERILOG = iverilog
VVP      = vvp
PYTHON   = python3

RTL_DIR  = ../rtl
SIM_DIR  = .

# RTL sources
RTL_COMMON = $(RTL_DIR)/fixed_mul.v

.PHONY: all test_all test_mul test_neuron test_scheduler ref clean

all: test_all

# ============================================
# Fixed-point multiplier test
# ============================================
test_mul: tb_fixed_mul.vvp
	$(VVP) $<

tb_fixed_mul.vvp: tb_fixed_mul.v $(RTL_COMMON)
	$(IVERILOG) -o $@ -I$(RTL_DIR) $^

# ============================================
# Neuron core test
# ============================================
test_neuron: tb_neuron_core.vvp
	$(VVP) $<

tb_neuron_core.vvp: tb_neuron_core.v $(RTL_COMMON) $(RTL_DIR)/neuron_core.v
	$(IVERILOG) -o $@ -I$(RTL_DIR) $^

# ============================================
# Pixel scheduler test
# ============================================
test_scheduler: tb_pixel_scheduler.vvp
	$(VVP) $<

tb_pixel_scheduler.vvp: tb_pixel_scheduler.v $(RTL_DIR)/pixel_scheduler.v
	$(IVERILOG) -o $@ -I$(RTL_DIR) $^

# ============================================
# All tests
# ============================================
test_all: test_mul test_neuron test_scheduler

# ============================================
# Python reference
# ============================================
ref:
	$(PYTHON) mandelbrot_ref.py

# ============================================
# Clean
# ============================================
clean:
	rm -f *.vvp *.vcd
