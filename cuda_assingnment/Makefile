NVCC = nvcc
# Updated flags: Use C++17 and suppress deprecated target warnings
NVCC_FLAGS = -O3 -std=c++17 -Wno-deprecated-gpu-targets

SRCDIR = src
OBJDIR = obj
BINDIR = bin

TARGET = $(BINDIR)/nbody_sim

all: directories $(TARGET)

directories:
	mkdir -p $(OBJDIR) $(BINDIR)

$(TARGET): $(SRCDIR)/main.cu
	$(NVCC) $(NVCC_FLAGS) $< -o $@

clean:
	rm -rf $(OBJDIR) $(BINDIR) *.csv *.png

run: all
	./$(TARGET) 4096
