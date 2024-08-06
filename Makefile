CXX      = g++-14
CXXFLAGS = -O3 -Wall -std=c++23 -fopenmp -g
INCLUDES = -I/usr/local/include

PROG = test.out
SRCS = test.cpp

OBJS = $(SRCS:%.cpp=%.o)
DEPS = $(SRCS:%.cpp=%.d)

CXXFLAGS += $(INCLUDES)
###

$(PROG): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

-include $(DEPS)

.cpp.o:
	$(CXX) $(CXXFLAGS) -c -MMD -MP $<

###

.PHONY: clean
clean:
	rm -f $(OBJS) $(DEPS)
