# for host code written in c
CXX = clang
CXX_FLAGS = -std=c2x -Wall
INCLUDE_DIR = -I./include
LINK_FLAGS = -lOpenCL -lm
USE_SPIRV_FLAG = -DPROGRAM_FROM_IL

# for kernel written in opencl c
# which are compiled to llvm IR, emitting llvm bitcode
CLCXX = clang
CLCXX_FLAGS = -c -Wall -cl-std=CL2.0 -target spir64 -O0 -emit-llvm -DTO_IL
LLVM_IR_0 = kernel_0.bc
LLVM_IR_1 = kernel_1.bc
LLVM_IR_2 = kernel_2.bc

# after we've llvm ir, it'll be translated to spirv IR
# using opensource llvm-spirv tool
LLVM_SPIRV = llvm-spirv
SPIRV_IR_0 = kernel_0.spv
SPIRV_IR_1 = kernel_1.spv
SPIRV_IR_2 = kernel_2.spv

all: run

run: main.c include/*.h kernel.cl
	$(CXX) $(CXX_FLAGS) $(INCLUDE_DIR) $< -o $@ $(LINK_FLAGS)

aot: main.c include/*.h $(SPIRV_IR_0) $(SPIRV_IR_1) $(SPIRV_IR_2)
	$(CXX) $(CXX_FLAGS) $(USE_SPIRV_FLAG) -DSPIRV_IR_0=$(SPIRV_IR_0) -DSPIRV_IR_1=$(SPIRV_IR_1) -DSPIRV_IR_2=$(SPIRV_IR_2) $(INCLUDE_DIR) $< -o run $(LINK_FLAGS)

$(SPIRV_IR_0): $(LLVM_IR_0)
	$(LLVM_SPIRV) $< -o $@

$(SPIRV_IR_1): $(LLVM_IR_1)
	$(LLVM_SPIRV) $< -o $@

$(SPIRV_IR_2): $(LLVM_IR_2)
	$(LLVM_SPIRV) $< -o $@

$(LLVM_IR_0): kernel.cl
	$(CLCXX) $(CLCXX_FLAGS) -DLE_BYTES_TO_WORDS -DWORDS_TO_LE_BYTES -DEXPOSE_BLAKE3_HASH $< -o $@

$(LLVM_IR_1): kernel.cl
	$(CLCXX) $(CLCXX_FLAGS) -DEXPOSE_BLAKE3_HASH $< -o $@

$(LLVM_IR_2): kernel.cl
	$(CLCXX) $(CLCXX_FLAGS) $< -o $@

format:
	find . -name '*.c' -o -name '*.h' -o -name '*.cl' | xargs clang-format -i -style=Mozilla

clean:
	find . -name 'a.out' -o -name  '*.o' -o -name 'run' -o -name 'kernel_*.bc' -o -name 'kernel_*.spv' | xargs rm -f
