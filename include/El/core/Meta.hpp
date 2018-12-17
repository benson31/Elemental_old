#pragma once
#ifndef EL_CORE_META_HPP_
#define EL_CORE_META_HPP_

#include <type_traits>

namespace El
{

// Metafunction for "And"; not SFINAE-safe (i.e., it's an error if any
// of the predicates cannot be evaluated)
template <typename... Ts> struct And;
template <> struct And<> : std::true_type {};
template <typename T, typename... Ts>
struct And<T,Ts...>
{
    static constexpr bool value = T::value && And<Ts...>::value;
};

// Metafunction for "Or"; not SFINAE-safe
template <typename... Ts> struct Or;
template <> struct Or<> : std::false_type {};
template <typename T, typename... Ts>
struct Or<T,Ts...>
{
    static constexpr bool value = T::value || Or<Ts...>::value;
};

// Metafunction for "Not"; not SFINAE-safe
template <typename T> struct Not
{
    static constexpr bool value = !T::value;
};

// Basic typelist implementation. Hi Lisp.
template <typename... Ts> struct TypeList {};

// Get the "car" of the list.
template <typename T> struct HeadT;
template <typename T, typename... Ts>
struct HeadT<TypeList<T,Ts...>>
{
    using type = T;
};

// Get the "cdr" of the list.
template <typename T> struct TailT;
template <typename T, typename... Ts>
struct TailT<TypeList<T, Ts...>>
{
    using type = TypeList<Ts...>;
};

// Convenience Head/Tail functions.
template <typename T> using Head = typename HeadT<T>::type;
template <typename T> using Tail = typename TailT<T>::type;

// Wrapper around std::conditional
template <typename B, typename T, typename U>
using Select = typename std::conditional<B::value, T, U>::type;

/** \class SelectFirstMatch
 *  \brief Metafunction that returns the first match in the list.
 *
 *  When Pred<U,Head<List>> returns TrueType, this function returns
 *  Head<List>. It is an error if no type matches.

 *  \tparam List Expected to be a TypeList.
 *  \tparam U The test type.
 *  \tparam Pred A predicate class that takes Head<List> and U as arguments
 */
template <typename List, typename U, template <class,class> class Pred>
struct SelectFirstMatch
    : Select<Pred<U,Head<List>>, HeadT<List>,
             SelectFirstMatch<Tail<List>,U,Pred>>
{};

// Predicate that returns true if Pred<T, X> is true_type for any X in List.
template <typename List, typename T, template <class, class> class Pred>
struct IsTrueForAny;

template <typename T, template <class, class> class Pred>
struct IsTrueForAny<TypeList<>, T, Pred> : std::false_type {};

template <typename List, typename T, template <class, class> class Pred>
struct IsTrueForAny
    : Or<Pred<T,Head<List>>, IsTrueForAny<Tail<List>,T,Pred>>
{};

// Predicate that returns true if Pred<T, X> is true_type for all X in List.
template <typename List, typename T, template <class, class> class Pred>
struct IsTrueForAll;

template <typename T, template <class, class> class Pred>
struct IsTrueForAll<TypeList<>, T, Pred> : std::true_type {};

template <typename List, typename T, template <class, class> class Pred>
struct IsTrueForAll
    : And<Pred<T,Head<List>>, IsTrueForAll<Tail<List>,T,Pred>>
{};

// Metafunction for enum equality
template <typename EnumT, EnumT A, EnumT B>
struct EnumSame : std::false_type {};
template <typename EnumT, EnumT A>
struct EnumSame<EnumT,A,A> : std::true_type {};

// Rename the STL predicate for type equivalence
template <typename S,typename T>
using IsSame = std::is_same<S,T>;

// Rename the STL predicate for integral-ness.
template <typename T>
using IsIntegral = std::is_integral<T>;

// SFINAE wrapper for true conditions
template <typename Condition, class T=void>
using EnableIf = typename std::enable_if<Condition::value,T>::type;

// SFINAE wrapper for false conditions
template <typename Condition,class T=void>
using DisableIf = typename std::enable_if<!Condition::value,T>::type;

template<typename T> struct IsScalar : std::false_type {};
template<> struct IsScalar<unsigned> : std::true_type {};
template<> struct IsScalar<int> : std::true_type {};
template<> struct IsScalar<unsigned long> : std::true_type {};
template<> struct IsScalar<long int> : std::true_type {};
template<> struct IsScalar<unsigned long long> : std::true_type {};
template<> struct IsScalar<long long int> : std::true_type {};
template<> struct IsScalar<float> : std::true_type {};
template<> struct IsScalar<double> : std::true_type {};
template<> struct IsScalar<long double> : std::true_type {};

template<typename T> struct IsField : std::false_type {};
template<> struct IsField<float> : std::true_type {};
template<> struct IsField<double> : std::true_type {};
template<> struct IsField<long double> : std::true_type {};

template<typename T> struct IsStdScalar : std::false_type {};
template<> struct IsStdScalar<unsigned> : std::true_type {};
template<> struct IsStdScalar<int> : std::true_type {};
template<> struct IsStdScalar<unsigned long> : std::true_type {};
template<> struct IsStdScalar<long int> : std::true_type {};
template<> struct IsStdScalar<unsigned long long> : std::true_type {};
template<> struct IsStdScalar<long long int> : std::true_type {};
template<> struct IsStdScalar<float> : std::true_type {};
template<> struct IsStdScalar<double> : std::true_type {};
template<> struct IsStdScalar<long double> : std::true_type {};

template<typename T> struct IsStdField : std::false_type {};
template<> struct IsStdField<float> : std::true_type {};
template<> struct IsStdField<double> : std::true_type {};
template<> struct IsStdField<long double> : std::true_type {};

}// namespace El
#endif /* EL_CORE_META_HPP_ */
