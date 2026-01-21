# Root Makefile to invoke the Makefile in the src directory

.PHONY: all clean

all:
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean