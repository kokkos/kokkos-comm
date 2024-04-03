add_library(KokkosComm INTERFACE)

target_include_directories(KokkosComm INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(KokkosComm INTERFACE MPI::MPI_CXX Kokkos::kokkos)
if(KOKKOSCOMM_USE_KOKKOS_MDSPAN)
  target_link_libraries(KokkosComm INTERFACE mdspan)
endif()

# FIXME Not sure it is really possible to set BUILD_INTERFACE for header only libraries
# We should ensure that we do not export convenience flags to users
kokkoscomm_add_cxx_flags(TARGET KokkosComm INTERFACE)

set (KokkosCommPublicHeaders
        KokkosComm.hpp
        KokkosComm_collective.hpp
        KokkosComm_pack_traits.hpp
        KokkosComm_request.hpp
        KokkosComm_traits.hpp
        impl/KokkosComm_concepts.hpp
        impl/KokkosComm_include_mpi.hpp
        impl/KokkosComm_isend.hpp
        impl/KokkosComm_mdspan.hpp
        impl/KokkosComm_packer.hpp
        impl/KokkosComm_recv.hpp
        impl/KokkosComm_reduce.hpp
        impl/KokkosComm_send.hpp
        impl/KokkosComm_types.hpp
)

set_target_properties(KokkosComm PROPERTIES PUBLIC_HEADER "${KokkosCommPublicHeaders}")
