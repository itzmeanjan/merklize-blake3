# for host code written in c
CXX = clang
CXX_FLAGS = -std=c17 -Wall
INCLUDE_DIR = -I./include
LINK_FLAGS = -lOpenCL
PROG = run

# for kernel written in opencl c
CLCXX = clang
CLCXX_FLAGS = -Wall -cl-std=CL2.0 -target spir64 -O0 -emit-llvm

all: kernel $(PROG)

$(PROG): main.c include/*.h
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIR) $< $(LINK_FLAGS) -o $@

kernel:
	$(CLCXX) $(CLCXX_FLAGS) -c -o /dev/null kernel.cl

format:
	find . -name '*.c' -o -name '*.h' -o -name '*.cl' | xargs clang-format -i -style=Mozilla

clean:
	find . -name 'a.out' -o -name 'run' | xargs rm -f
