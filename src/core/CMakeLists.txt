# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  DistMap.cpp
  Element.cpp
  Environment.cpp
  Exception.cpp
  Grid.cpp
  Instantiate.cpp
  Profiling.cpp
  Serialize.cpp
  Timer.cpp
  callStack.cpp
  environment_impl.cpp
  indent.cpp
  logging.cpp
  mpi_register.cpp
  random.cpp
  types.cpp
  )

# Add the subdirectories
add_subdirectory(DistMatrix)
#add_subdirectory(FlamePart)
add_subdirectory(imports)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
