# FIXME:
# On the long run we do not wish to keep this distinction between MPI libraries.
# This should be re-worked and/or upstreamed into MPI libraries so that we don't need to perform such checks.

function(kokkoscomm_set_mpi_vendor_variables)
  # Initialize the variables to false
  set(KokkosComm_IMPL_MPI_IS_MPICH FALSE CACHE BOOL "MPI is MPICH")
  set(KokkosComm_IMPL_MPI_IS_OPENMPI FALSE CACHE BOOL "MPI is Open MPI")
  set(KokkosComm_IMPL_MPI_HAS_EXT_H FALSE CACHE BOOL "MPI has `mpi-ext.h` header")

  if(KokkosComm_ENABLE_MPI)
    if(NOT MPIEXEC_EXECUTABLE)
      message(WARNING "Unable to determine MPI vendor - `$MPIEXEC_EXECUTABLE` not set.")
      return()
    endif()

    # Get the directory of the MPI executable
    get_filename_component(MPIEXEC_DIR ${MPIEXEC_EXECUTABLE} DIRECTORY)

    # Check for mpichversion and ompi_info
    find_program(MPICHVERSION_EXECUTABLE mpichversion HINTS ${MPIEXEC_DIR} NO_DEFAULT_PATH)
    find_program(OMPI_INFO_EXECUTABLE ompi_info HINTS ${MPIEXEC_DIR} NO_DEFAULT_PATH)

    if(MPICHVERSION_EXECUTABLE AND OPENMPI_INFO_EXECUTABLE)
      message(
        WARNING
        "Unable to determine MPI vendor - both `$MPICHVERSION_EXECUTABLE` and `$OMPI_INFO_EXECUTABLE` are set."
      )
    elseif(MPICHVERSION_EXECUTABLE)
      message(STATUS "Detected MPI as MPICH.")
      set(KokkosComm_IMPL_MPI_IS_MPICH TRUE CACHE BOOL "MPI is MPICH" FORCE)
    elseif(OMPI_INFO_EXECUTABLE)
      message(STATUS "Detected MPI as Open MPI.")
      set(KokkosComm_IMPL_MPI_IS_OPENMPI TRUE CACHE BOOL "MPI is Open MPI" FORCE)
    endif()

    # Use CXX module because `LANGUAGE C` is not enabled by the KokkosComm project
    include(CheckIncludeFileCXX)
    check_include_file_cxx(mpi-ext.h HAVE_MPI_EXT_H)
    if(HAVE_MPI_EXT_H)
      message(STATUS "MPI has `mpi-ext.h` header")
      set(KokkosComm_IMPL_MPI_HAS_MPI_EXT_H TRUE CACHE BOOL "MPI has `mpi-ext.h` header" FORCE)
    endif()
  endif()
endfunction()
