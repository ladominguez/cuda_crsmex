CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
CC := gcc
LINK := nvcc -arch=sm_30
NVCC  := nvcc -arch=sm_30

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I/usr/local/sac/include
#INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include 

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)
#LIB_SAC := `sac-config --cflags --libs sac`
LIB_SAC  := -L/usr/local/sac/lib  -lsacio -lsac -lm 
OBJS = crsmex.cpp.o main.cu.o
TARGET = exec
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_SAC)

.SUFFIXES: .c .cpp .cu .o

%.cpp.o: %.cpp
	$(CXX) $(CFLAGS) -c $< -o $@

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

%.cu.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
