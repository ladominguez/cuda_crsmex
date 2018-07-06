CUDA_INSTALL_PATH := /usr/local/cuda

CXX := g++
CC := gcc
LINK := g++ -fPIC
NVCC  := nvcc

# Includes
INCLUDES = -I. -I$(CUDA_INSTALL_PATH)/include -I/usr/local/sac/include

# Common flags
COMMONFLAGS += $(INCLUDES)
NVCCFLAGS += $(COMMONFLAGS)
CXXFLAGS += $(COMMONFLAGS)
CFLAGS += $(COMMONFLAGS)

LIB_SAC  := -L/usr/local/sac/lib  -lsacio -lsac -lm 
OBJS = main.c.o
TARGET = exec
LINKLINE = $(LINK) -o $(TARGET) $(OBJS) $(LIB_SAC)

.SUFFIXES: .c .cpp .cu .o

%.c.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

$(TARGET): $(OBJS) Makefile
	$(LINKLINE)
