#pragma once
#ifndef EL_CORE_EXCEPTION_HPP_
#define EL_CORE_EXCEPTION_HPP_

#include <iostream>
#include <sstream>
#include <stdexcept>

namespace El
{

// Helper function for debugging
void break_on_me() noexcept;

inline void BuildStream(std::ostream&) {}

template <typename T, typename... ArgPack>
void BuildStream(std::ostream& os, T const& item, ArgPack const&... args)
{
    os << item;
    BuildStream(os, args...);
}

template <typename... ArgPack>
std::string BuildString(ArgPack const&... args )
{
    std::ostringstream os;
    BuildStream(os, args...);
    return os.str();
}

/** \class ArgException
 *  \brief Indicates a problem with an argument.
 */
class ArgException : public std::runtime_error
{
public:
    ArgException(char const* msg="")
        : ArgException{std::string{msg}}
    { }
    ArgException(std::string msg)
        : std::runtime_error{std::move(msg)}
    { }
};

/** \class UnrecoverableException
 *  \brief An exception from which no recovery is possible
 */
class UnrecoverableException
    : public std::runtime_error
{
public:
    UnrecoverableException(char const* msg="Unrecoverable exception")
        : UnrecoverableException{std::string{msg}}
    { }
    UnrecoverableException(std::string msg)
        : std::runtime_error{std::move(msg)}
    { }
};

/** \class SingularMatrixException
 *  \brief Signifies that a matrix was unexpectedly singular.
 */
class SingularMatrixException
    : public std::runtime_error
{
public:
    SingularMatrixException(const char* msg="Matrix was singular")
        : SingularMatrixException{std::string{msg}}
    { }
    SingularMatrixException(std::string msg)
        : std::runtime_error{std::move(msg)}
    { }
};

template<typename... ArgPack>
void LogicError(ArgPack const& ... args)
{
    break_on_me();

    throw std::logic_error(BuildString(args...));
}

template<typename... ArgPack>
void RuntimeError(ArgPack const& ... args)
{
    break_on_me();

    throw std::runtime_error(BuildString(args...));
}


template <typename... ArgPack>
void UnrecoverableError(ArgPack const&... args)
{
    break_on_me();

    throw UnrecoverableException(BuildString(args...));
}

void ReportException(std::exception const& e, std::ostream& os=std::cout);

}
#endif /* EL_CORE_EXCEPTION_HPP_ */
