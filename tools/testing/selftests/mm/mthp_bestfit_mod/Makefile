# SPDX-License-Identifier: GPL-2.0
#
# mthp_bestfit_mod — Makefile
#
# On-device build (RPi5, kernel headers installed):
#   make
#
# Cross-compile for RPi5 from x86 host:
#   make ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- KDIR=/path/to/rpi-linux
#
# In-tree build (inside kernel source):
#   obj-$(CONFIG_MTHP_BESTFIT_MOD) += mthp_bestfit_mod.o
#   (add to mm/Makefile and mm/Kconfig)

obj-m := mthp_bestfit_mod.o

KDIR ?= /lib/modules/$(shell uname -r)/build
PWD  := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install
	depmod -a

# RPi5 cross-compile convenience target
# Requires: sudo apt install gcc-aarch64-linux-gnu
# Set KDIR to your RPi Linux source tree with .config for bcm2712
rpi5:
	$(MAKE) ARCH=arm64 CROSS_COMPILE=aarch64-linux-gnu- \
		KDIR=$(KDIR) M=$(PWD) modules
