# otpu_pcie, openTPU's driver for its LitePCIe bitstreams (docs/litepcie.md). Out of tree:
#   make                  against the running kernel's headers (KDIR to name others)
# opentpu/host/setup_pcie.sh installs it through DKMS (dkms.conf). csr.h is
# tools/litepcie/gen_core.py's, from the core the bitstream carries.
ifneq ($(KERNELRELEASE),)
obj-m := otpu_pcie.o
else
KDIR ?= /lib/modules/$(shell uname -r)/build

all:
	$(MAKE) -C $(KDIR) M=$(CURDIR) modules

clean:
	$(MAKE) -C $(KDIR) M=$(CURDIR) clean
endif
