.SUFFIXES: .h .cpp
#-----------------------------------------------
# Makefile for the program 'ALM'.
# Please modify the variables properly.
# We recommend to use Intel c++ compiler.
#-----------------------------------------------

# CXX = icpc 
# CXXFLAGS = -O2 -xHOST -qopenmp -std=c++11
# #CXXFLAGS = -O2 -xHOST -qopenmp -std=c++11 -D_HDF5
# INCLUDE = -I../include -I$(HOME)/include -I$(SPGLIB_ROOT)/include
# #INCLUDE = -I../include -I$(HOME)/include -I$(SPGLIB_ROOT)/include -I$(HDF5_ROOT)/include

# CXXL = ${CXX}
# LDFLAGS = -mkl -L$(SPGLIB_ROOT)/lib -lsymspg
# #LDFLAGS = -mkl -L$(SPGLIB_ROOT)/lib -lsymspg -L$(HDF5_ROOT)/lib -lhdf5_cpp -lhdf5

CXX = mpiicpc
CXXFLAGS = -O2 -xHOST -qopenmp -std=c++11
#CXXFLAGS = -O2 -xHOST -qopenmp -std=c++11 -D_HDF5
# 2022/4/10 need eigen, fftw, boost
# 2022/4/10 do not use "/home/k0151/k015124/src/boost_1_78_0/boost" !! because codes uses #include "boost/~".
INCLUDE = -I../include -I$(HOME)/include -I$(SPGLIB_ROOT)/include -I/home/k0151/k015124/src/eigen-3.4.0 -I/home/k0151/k015124/src/eigen-3.4.0/Eigen -I/home/local/intel/mkl/include/fftw -I/home/k0151/k015124/src/boost_1_78_0/
#INCLUDE = -I../include -I$(HOME)/include -I$(SPGLIB_ROOT)/include -I$(HDF5_ROOT)/include

CXXL = ${CXX}
# 2022/4/10 change here from lib to lib64. symspg is included in lib64 directory.
LDFLAGS = -mkl -L$(SPGLIB_ROOT)/lib64 -lsymspg
# LDFLAGS = -mkl -L$(SPGLIB_ROOT)/lib -lsymspg original
#LDFLAGS = -mkl -L$(SPGLIB_ROOT)/lib -lsymspg -L$(HDF5_ROOT)/lib -lhdf5_cpp -lhdf5


LAPACK = 
LIBS = ${LAPACK}

#-----------------------------------------------
# General rules 
#-----------------------------------------------

.cpp.o:
	${CXX} ${CXXFLAGS} ${INCLUDE} -c $<

PROG = alm

CXXSRC= $(wildcard *.cpp)

OBJS= ${CXXSRC:.cpp=.o}

default: alm

all: ${PROG}

alm: ${OBJS}
	${CXXL} ${LDFLAGS} -o $@ ${OBJS} ${LIBS}

clean:
	rm -f ${OBJS}

.PHONY: clean

depend:
	gcc -MM *.cpp > .depend

-include .depend
