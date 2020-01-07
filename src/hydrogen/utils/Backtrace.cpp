#include "hydrogen/utils/Backtrace.hpp"

#define UNW_LOCAL_ONLY
#include <cxxabi.h>
#include <libunwind.h>

#include <iomanip>
#include <iostream>
#include <limits>

// Be fairly liberal here because (1) I don't care about the 8k and (2) Elemental has some stupidly long symbols and I want to get them all.
constexpr size_t MAX_MANGLED_SIZE = 8192;

namespace hydrogen
{

Backtrace::Backtrace() {}

namespace
{

std::string AttemptDemangle(char const* mangled)
{
    std::string ret;

    int status;
    auto name =
        abi::__cxa_demangle(mangled, nullptr, nullptr, &status);
    if (status == 0)
    {
        ret = name;
        free(name);
    }
    else
    {
        ret = mangled;
    }
    return ret;
}

std::pair<std::string, unw_word_t>
GetSymbolName(unw_cursor_t& cursor)
{
    std::string ret;
    unw_word_t offset;

    char mangled[MAX_MANGLED_SIZE];
    if (unw_get_proc_name(&cursor, mangled, MAX_MANGLED_SIZE, &offset) != 0)
    {
        throw std::runtime_error("Couldn't even get mangled symbol.");
    }

    return std::make_pair(AttemptDemangle(mangled), offset);
}

void PrintFrame(unw_cursor_t& cursor, size_t frame_id, std::ostream& os)
{
    unw_word_t instr_ptr /* "ip" */, stack_ptr /* "sp" */;

    unw_get_reg(&cursor, UNW_REG_IP, &instr_ptr);
    unw_get_reg(&cursor, UNW_REG_SP, &stack_ptr);

    auto frame_name = GetSymbolName(cursor);
    os << std::dec << "  Frame " << frame_id << ": "
       << std::hex << std::setw(20) << std::right
       << std::showbase << instr_ptr << ": " << frame_name.first
       << " (+" << frame_name.second << ")"
       << std::dec << std::noshowbase << std::endl;
}

void PrintImpl(size_t max_frames, std::ostream& os)
{
    unw_cursor_t cursor;
    unw_context_t ctxt;

    unw_getcontext(&ctxt);
    unw_init_local(&cursor, &ctxt);

    // Increment the cursor and print things (This _should_ skip this
    // frame but catch everything else.
    for (size_t frame_id = 0UL;
         unw_step(&cursor) > 0 && frame_id < max_frames; ++frame_id)
    {
        PrintFrame(cursor, frame_id, os);
    }
}

bool global_doing_gemm_ = false;

}// namespace <anon>

bool global_doing_gemm() { return  false; } //global_doing_gemm_; }
void global_start_gemm() { global_doing_gemm_ = true; }
void global_stop_gemm() { global_doing_gemm_ = false; }

void Backtrace::Print(size_t max_frames, std::ostream& os) const
{
    PrintImpl(max_frames, os);
}

void Backtrace::PrintAll(std::ostream& os) const
{
    PrintImpl(std::numeric_limits<size_t>::max(), os);
}

}// namespace hydrogen
