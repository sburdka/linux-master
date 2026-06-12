obj-m += mthp_bestfit.o

KDIR ?= /lib/modules/$(shell uname -r)/build
ARCH ?= arm64
LLVM ?= 1

default:
	$(MAKE) -C $(KDIR) M=$(PWD) ARCH=$(ARCH) LLVM=$(LLVM) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) ARCH=$(ARCH) LLVM=$(LLVM) clean
