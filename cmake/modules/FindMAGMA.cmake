# Exports the following variables
#
#   MAGMA_FOUND
#   MAGMA_INCLUDE_PATH
#   MAGMA_LIBRARIES
#
# Also adds the following imported target:
#
#   cuda::magma
#

find_path(MAGMA_INCLUDE_PATH magma.h
  HINTS ${MAGMA_DIR} $ENV{MAGMA_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES include
  NO_DEFAULT_PATH
  DOC "The MAGMA header directory."
  )
find_path(MAGMA_INCLUDE_PATH magma.h)

find_library(MAGMA_LIBRARY magma
  HINTS ${MAGMA_DIR} $ENV{MAGMA_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib lib64/stubs lib/stubs
  NO_DEFAULT_PATH
  DOC "The MAGMA library.")
find_library(MAGMA_LIBRARY magma)

find_library(MAGMA_SPARSE_LIBRARY magma_sparse
  HINTS ${MAGMA_DIR} $ENV{MAGMA_DIR} ${CUDA_TOOLKIT_ROOT_DIR} ${CUDA_SDK_ROOT_DIR}
  PATH_SUFFIXES lib64 lib lib64/stubs lib/stubs
  NO_DEFAULT_PATH
  DOC "The MAGMA sparse library.")
find_library(MAGMA_SPARSE_LIBRARY magma_sparse)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MAGMA
  DEFAULT_MSG
  MAGMA_LIBRARY MAGMA_SPARSE_LIBRARY MAGMA_INCLUDE_PATH)

# Setup the imported target
if (NOT TARGET cuda::magma)
  add_library(cuda::magma INTERFACE IMPORTED)
endif (NOT TARGET cuda::magma)

# Set the include directories for the target
set_property(TARGET cuda::magma
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MAGMA_INCLUDE_PATH})

set_property(TARGET cuda::magma
  PROPERTY INTERFACE_LINK_LIBRARIES ${MAGMA_LIBRARY} ${MAGMA_SPARSE_LIBRARY})

#
# Cleanup
#

# Set the include directories
mark_as_advanced(FORCE MAGMA_INCLUDE_DIRS)

# Set the libraries
set(MAGMA_LIBRARIES cuda::magma)
