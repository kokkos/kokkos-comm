# FIXME:
# On the long run we do not wish to keep this distinction between MPI libraries.
# This should be re-worked and/or upstreamed into MPI libraries so that we don't need to perform such checks.

function(kokkoscomm_set_mpi_vendor_variables)

  if(KOKKOSCOMM_IMPL_MPI_IS_MPICH)
    message(STATUS "Using defined MPI vendor: MPICH")
    return()
  elseif(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI)
    message(STATUS "Using defined MPI vendor: OPENMPI")
    return()
  elseif(KokkosComm_ENABLE_MPI)
    if(MPIEXEC_EXECUTABLE)
      # Prefer the launcher directory because it is tied to the MPI runtime.
      get_filename_component(MPI_BIN_DIR ${MPIEXEC_EXECUTABLE} DIRECTORY)
    elseif(MPI_CXX_COMPILER)
      # Some clusters expose an MPI compiler wrapper but intentionally disable
      # mpiexec on login nodes.
      get_filename_component(MPI_BIN_DIR ${MPI_CXX_COMPILER} DIRECTORY)
    else()
      message(WARNING "Unable to determine MPI vendor - neither `MPIEXEC_EXECUTABLE` nor `MPI_CXX_COMPILER` is set")
      return()
    endif()

    # Check for mpichversion and ompi_info
    find_program(MPICHVERSION_EXECUTABLE mpichversion HINTS ${MPI_BIN_DIR} NO_DEFAULT_PATH)
    find_program(OMPI_INFO_EXECUTABLE ompi_info HINTS ${MPI_BIN_DIR} NO_DEFAULT_PATH)

    if(MPICHVERSION_EXECUTABLE AND OMPI_INFO_EXECUTABLE)
      message(
        WARNING
        "Unable to determine MPI vendor - both `MPICHVERSION_EXECUTABLE` and `OMPI_INFO_EXECUTABLE` are set"
      )
    elseif(MPICHVERSION_EXECUTABLE)
      message(STATUS "Detected MPI as MPICH")
      set(KOKKOSCOMM_IMPL_MPI_IS_MPICH TRUE CACHE BOOL "MPI is MPICH" FORCE)
    elseif(OMPI_INFO_EXECUTABLE)
      message(STATUS "Detected MPI as Open MPI")
      set(KOKKOSCOMM_IMPL_MPI_IS_OPENMPI TRUE CACHE BOOL "MPI is Open MPI" FORCE)
    else()
      message(WARNING "Unable to determine MPI vendor - unknown MPI implementation")
    endif()
  endif()

  # Use CXX module because `LANGUAGE C` is not enabled by the KokkosComm project
  include(CheckIncludeFileCXX)
  check_include_file_cxx(mpi-ext.h MPI_HAS_MPIEXT_H)
  if(MPI_HAS_MPIEXT_H)
    message(STATUS "MPI vendor has `mpi-ext.h` header")
    set(KOKKOSCOMM_IMPL_MPI_HAS_MPIEXT_H TRUE CACHE BOOL "MPI vendor has `mpi-ext.h` header")
  endif()

endfunction()
