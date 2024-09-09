Handle<> handle;
Kokkos::View<double*> recv_view("recv_view", 100);
auto req = recv(handle, recv_view, 1/*src*/);
KokkosComm::wait(req);