# Build both modules against a configured GKI 6.12 tree.
# Note: mthp_bestfit resolves mthp_monitor_get_sysinfo at runtime via
# symbol_get() (string lookup), so no KBUILD_EXTRA_SYMBOLS is required
# and either module builds independently.

KDIR ?= /lib/modules/$(shell uname -r)/build
ARCH ?= arm64
LLVM ?= 1
SUBDIRS := mthp_monitor mthp_bestfit

all:
	@for d in $(SUBDIRS); do \
		$(MAKE) -C $(KDIR) M=$(CURDIR)/$$d ARCH=$(ARCH) LLVM=$(LLVM) modules || exit 1; \
	done

clean:
	@for d in $(SUBDIRS); do \
		$(MAKE) -C $(KDIR) M=$(CURDIR)/$$d ARCH=$(ARCH) LLVM=$(LLVM) clean; \
	done
