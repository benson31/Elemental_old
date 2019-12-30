# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  ColumnMinAbs.cpp
  ColumnNorms.cpp
  Copy.cpp
  HilbertSchmidt.cpp
  Instantiate.cpp
  Max.cpp
  MaxAbs.cpp
  MaxAbsLoc.cpp
  MaxLoc.cpp
  Min.cpp
  MinAbsLoc.cpp
  MinLoc.cpp
  RowMinAbs.cpp
  RowNorms.cpp
  Swap.cpp
  Symmetric2x2Inv.cpp
  Transform2x2.cpp
  NormsFromScaledSquares.hpp
  )

# Setup the Copy ETI generator
include(HydrogenETIGeneration)

set(HYDROGEN_COPY_TYPES
  "float"
  "double"
  )
if (HYDROGEN_HAVE_HALF)
  list(APPEND HYDROGEN_COPY_TYPES "cpu_half_type")
endif ()
if (HYDROGEN_GPU_USE_FP16)
  list(APPEND HYDROGEN_COPY_TYPES "gpu_half_type")
endif ()

set(HYDROGEN_COPY_DEVICES "Device::CPU")
if (HYDROGEN_HAVE_GPU)
  list(APPEND HYDROGEN_COPY_DEVICES "Device::GPU")
endif ()

string(CONCAT _H_COPY_STRING
  "template void Copy("
  "Matrix<@_ARG0@, @_ARG1@> const&, Matrix<@_ARG2@, @_ARG3@>&);\n"
  "template void CopyAsync("
  "Matrix<@_ARG0@, @_ARG1@> const&, Matrix<@_ARG2@, @_ARG3@>&)")
h_generate_tensor_product_expression(
  _H_COPY_ETI_LIST
  "${_H_COPY_STRING}"
  HYDROGEN_COPY_TYPES
  HYDROGEN_COPY_DEVICES
  HYDROGEN_COPY_TYPES
  HYDROGEN_COPY_DEVICES
  )
h_func_list_to_string(H_COPY_ETI_BIG _H_COPY_ETI_LIST)

# Add in the few "sane" types.
set(HYDROGEN_COPY_TYPES "uint8_t" "int" "Complex<float>" "Complex<double>")
h_generate_tensor_product_expression(
  _H_COPY_ETI_LIST
  "EL_EXTERN template void Copy(Matrix<@_ARG0@, @_ARG1@> const&, Matrix<@_ARG0@, @_ARG1@>&)"
  HYDROGEN_COPY_TYPES
  HYDROGEN_COPY_DEVICES
  )
h_func_list_to_string(H_COPY_ETI_SANE _H_COPY_ETI_LIST)
string(JOIN "\n" H_COPY_ETI "${H_COPY_ETI_BIG}" "${H_COPY_ETI_SANE}")

configure_file("${CMAKE_CURRENT_SOURCE_DIR}/CopyETI.cpp.in"
  "${CMAKE_CURRENT_BINARY_DIR}/CopyETI.cpp")

list(APPEND THIS_DIR_SOURCES
  "${CMAKE_CURRENT_BINARY_DIR}/CopyETI.cpp"
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
