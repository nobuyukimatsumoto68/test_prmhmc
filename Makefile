CXX      = g++
CXXFLAGS = -O3 -std=c++17 -fopenmp -g -Wall
INCLUDES = -I/usr/local/include -I/opt/eigen/ -I/opt/eigen/Eigen/

PROG = tes.out
SRCS = test.cc

# OBJS = $(SRCS:%.cpp=%.o)
# DEPS = $(SRCS:%.cpp=%.d)

CXXFLAGS += $(INCLUDES)
###

$(PROG): $(SRCS) header.h
	$(CXX) $(CXXFLAGS) -o $@ $<

# -include $(DEPS)

# .cpp.o:
# 	$(CXX) $(CXXFLAGS) -c -MMD -MP $<

###

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS)
