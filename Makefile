CLCXX = clang
CLCFLAGS = -Wall -cl-std=CL2.0 -target spir64 -O0 -emit-llvm

all: format

kernel:
	$(CLCXX) $(CLCFLAGS) -c -o /dev/null kernel.cl

format:
	find . -name '*.c' -o -name '*.h' -o -name '*.cl' | xargs clang-format -i -style=Mozilla
