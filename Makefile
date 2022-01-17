all: format

format:
	find . -name '*.c' -o -name '*.h' -o -name '*.cl' | xargs clang-format -i -style=Mozilla
