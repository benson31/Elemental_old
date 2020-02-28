add_subdirectory(blas)

set(SOURCES "${SOURCES}" PARENT_SCOPE)
if (HYDROGEN_HAVE_GPU)
  set(CUDA_SOURCES "${CUDA_SOURCES}" PARENT_SCOPE)
endif ()
