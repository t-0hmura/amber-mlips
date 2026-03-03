CC      ?= gcc
CFLAGS  ?= -O2 -Wall -Wextra
TARGET   = qchem_shim

.PHONY: all clean

all: $(TARGET)

$(TARGET): qchem_shim.c
	$(CC) $(CFLAGS) -o $@ $< -lm

clean:
	rm -f $(TARGET)
