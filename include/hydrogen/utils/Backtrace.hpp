#ifndef HYDROGEN_UTILS_BACKTRACE_HPP_
#define HYDROGEN_UTILS_BACKTRACE_HPP_

#include <iostream>

namespace hydrogen
{

bool global_doing_gemm();
void global_start_gemm();
void global_stop_gemm();

class Backtrace
{
public:
    Backtrace();
    void Print(size_t max_frames, std::ostream& os) const;
    void PrintAll(std::ostream& os) const;
};// class Backtrace

}// namespace hydrogen
#endif // HYDROGEN_UTILS_BACKTRACE_HPP_
