#include <KokkosComm/KokkosComm.hpp>
#include "test_utils.hpp"
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>
#include <stream-triggering.h>

template <class ExecSpace>
struct SpaceInstance {
  static ExecSpace create() { return ExecSpace(); }
  static void destroy(ExecSpace&) {}
  static bool overlap() { return false; }
};

#ifdef KOKKOS_ENABLE_CUDA
template <>
struct SpaceInstance<Kokkos::Cuda> {
  static Kokkos::Cuda create() {
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    return Kokkos::Cuda(stream);
  }
  static void destroy(Kokkos::Cuda& space) {
    cudaStream_t stream = space.cuda_stream();
    cudaStreamDestroy(stream);
  }
  static bool overlap() {
    bool value          = true;
    auto local_rank_str = std::getenv("CUDA_LAUNCH_BLOCKING");
    if (local_rank_str) {
      value = (std::stoi(local_rank_str) == 0);
    }
    return value;
  }
};
#endif

struct CommHelper {
  MPI_Comm comm;
  KokkosComm::Experimental::stream::StreamContext ctx;

  int nx, ny, nz;  // Num MPI ranks in each dimension
  int me;          // My rank
  int nranks;      // N ranks
  int x, y, z;     // My pos in proc grid

  // Neighbor Ranks
  int up, down, left, right, front, back;

  CommHelper(MPI_Comm comm_) {
    comm = comm_;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);

    nx = std::pow(1.0 * nranks, 1.0 / 3.0);
    while (nranks % nx != 0) nx++;
    ny = std::sqrt(1.0 * (nranks / nx));
    while ((nranks / nx) % ny != 0) ny++;

    nz    = nranks / nx / ny;
    x     = me % nx;
    y     = (me / nx) % ny;
    z     = (me / nx / ny);
    left  = x == 0 ? -1 : me - 1;
    right = x == nx - 1 ? -1 : me + 1;
    down  = y == 0 ? -1 : me - nx;
    up    = y == ny - 1 ? -1 : me + nx;
    front = z == 0 ? -1 : me - nx * ny;
    back  = z == nz - 1 ? -1 : me + nx * ny;
  }
};

using buffer_t = Kokkos::View<double**, Kokkos::LayoutLeft, Kokkos::HIP>;

struct InnerDT {
  Kokkos::View<double***, Kokkos::HIP> T, dT;
  double q;

  KOKKOS_FUNCTION
  void operator()(int x, int y, int z) const {
    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);
    dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    dT(x, y, z) = dT_xyz;
  }
};

enum Direction { left, right, down, up, front, back };

template <int Surface>
struct SurfaceDT {
  Kokkos::View<double***, Kokkos::HIP> T, dT;
  buffer_t T_left, T_right, T_up, T_down, T_front, T_back;
  int X_lo, Y_lo, Z_lo, X_hi, Y_hi, Z_hi, X, Y, Z;
  double q, sigma, P;

  KOKKOS_FUNCTION void operator()(int i, int j) const {
    int NX = T.extent(0);
    int NY = T.extent(1);
    int NZ = T.extent(2);
    int x, y, z;
    if (Surface == left) {
      x = 0;
      y = i;
      z = j;
    }
    if (Surface == right) {
      x = NX - 1;
      y = i;
      z = j;
    }
    if (Surface == down) {
      x = i;
      y = 0;
      z = j;
    }
    if (Surface == up) {
      x = i;
      y = NY - 1;
      z = j;
    }
    if (Surface == front) {
      x = i;
      y = j;
      z = 0;
    }
    if (Surface == back) {
      x = i;
      y = j;
      z = NZ - 1;
    }

    double dT_xyz = 0.0;
    double T_xyz  = T(x, y, z);

    // Heat conduction to inner body
    if (x > 0) dT_xyz += q * (T(x - 1, y, z) - T_xyz);
    if (x < NX - 1) dT_xyz += q * (T(x + 1, y, z) - T_xyz);
    if (y > 0) dT_xyz += q * (T(x, y - 1, z) - T_xyz);
    if (y < NY - 1) dT_xyz += q * (T(x, y + 1, z) - T_xyz);
    if (z > 0) dT_xyz += q * (T(x, y, z - 1) - T_xyz);
    if (z < NZ - 1) dT_xyz += q * (T(x, y, z + 1) - T_xyz);

    // Heat conduction with Halo
    if (x == 0 && X_lo != 0) dT_xyz += q * (T_left(y, z) - T_xyz);
    if (x == (NX - 1) && X_hi != X) dT_xyz += q * (T_right(y, z) - T_xyz);
    if (y == 0 && Y_lo != 0) dT_xyz += q * (T_down(x, z) - T_xyz);
    if (y == (NY - 1) && Y_hi != Y) dT_xyz += q * (T_up(x, z) - T_xyz);
    if (z == 0 && Z_lo != 0) dT_xyz += q * (T_front(x, y) - T_xyz);
    if (z == (NZ - 1) && Z_hi != Z) dT_xyz += q * (T_back(x, y) - T_xyz);

    // Incoming Power
    if (x == 0 && X_lo == 0) dT_xyz += P;

    // thermal radiation
    int num_surfaces = ((x == 0 && X_lo == 0) ? 1 : 0) + ((x == (NX - 1) && X_hi == X) ? 1 : 0) +
                       ((y == 0 && Y_lo == 0) ? 1 : 0) + ((y == (NY - 1) && Y_hi == Y) ? 1 : 0) +
                       ((z == 0 && Z_lo == 0) ? 1 : 0) + ((z == (NZ - 1) && Z_hi == Z) ? 1 : 0);
    dT_xyz -= sigma * T_xyz * T_xyz * T_xyz * T_xyz * num_surfaces;
    dT(x, y, z) = dT_xyz;
  }
};

// Some compilers have deduction issues if this were just a tagged operator, so a full Functor here instead
struct UpdateT {
  Kokkos::View<double***, Kokkos::HIP> T, dT;
  double dt;
  UpdateT(Kokkos::View<double***, Kokkos::HIP> T_, Kokkos::View<double***, Kokkos::HIP> dT_, double dt_) : T(T_), dT(dT_), dt(dt_) {}
  KOKKOS_FUNCTION
  void operator()(int x, int y, int z, double& sum_T) const {
    sum_T += T(x, y, z);
    T(x, y, z) += dt * dT(x, y, z);
  }
};

struct SystemKC {
  // Communicator
  CommHelper comm;
  MPIS_Request reqs_[12];
  int active_reqs = 0;

  // size of system
  int X, Y, Z;
  // Local box
  int X_lo, Y_lo, Z_lo;
  int X_hi, Y_hi, Z_hi;

  int N;  // number of timesteps
  int I;  // interval for print

  // Temperature and delta Temperature
  Kokkos::View<double***, Kokkos::HIP> T, dT;
  // Halo data
  buffer_t T_left, T_right, T_up, T_down, T_front, T_back;
  buffer_t T_left_out, T_right_out, T_up_out, T_down_out, T_front_out, T_back_out;

  Kokkos::HIP E_left, E_right, E_up, E_down, E_front, E_back, E_bulk;

  double T0;     // Initial temperature
  double dt;     // timestep width
  double q;      // thermal transfer coefficient
  double sigma;  // thermal radiation coefficient (assume Stefan Boltzmann law P = sigma*A*T^4
  double P;      // incoming power

  // init_system
  SystemKC(MPI_Comm comm_) : comm(comm_) {
    X = Y = Z = 200;
    X_lo = Y_lo = Z_lo = 0;
    X_hi = Y_hi = Z_hi = X;
    N                  = 10000;  // 10000 reduced for quick testing
    I                  = N - 1;
    T                  = Kokkos::View<double***, Kokkos::HIP>();
    dT                 = Kokkos::View<double***, Kokkos::HIP>();
    T0                 = 0.0;
    dt                 = 0.1;
    q                  = 1.0;
    sigma              = 1.0;
    P                  = 1.0;
    E_left             = SpaceInstance<Kokkos::HIP>::create();
    E_right            = SpaceInstance<Kokkos::HIP>::create();
    E_up               = SpaceInstance<Kokkos::HIP>::create();
    E_down             = SpaceInstance<Kokkos::HIP>::create();
    E_front            = SpaceInstance<Kokkos::HIP>::create();
    E_back             = SpaceInstance<Kokkos::HIP>::create();
    E_bulk             = SpaceInstance<Kokkos::HIP>::create();
  }

  void destroy_exec_spaces() {
    SpaceInstance<Kokkos::HIP>::destroy(E_left);
    SpaceInstance<Kokkos::HIP>::destroy(E_right);
    SpaceInstance<Kokkos::HIP>::destroy(E_front);
    SpaceInstance<Kokkos::HIP>::destroy(E_back);
    SpaceInstance<Kokkos::HIP>::destroy(E_up);
    SpaceInstance<Kokkos::HIP>::destroy(E_down);
    SpaceInstance<Kokkos::HIP>::destroy(E_bulk);
  }

  void setup_subdomain() {
    int dX = (X + comm.nx - 1) / comm.nx;
    X_lo   = dX * comm.x;
    X_hi   = X_lo + dX;
    if (X_hi > X) X_hi = X;
    int dY = (Y + comm.ny - 1) / comm.ny;
    Y_lo   = dY * comm.y;
    Y_hi   = Y_lo + dY;
    if (Y_hi > Y) Y_hi = Y;
    int dZ = (Z + comm.nz - 1) / comm.nz;
    Z_lo   = dZ * comm.z;
    Z_hi   = Z_lo + dZ;
    if (Z_hi > Z) Z_hi = Z;
    T  = Kokkos::View<double***, Kokkos::HIP>("System::T", X_hi - X_lo, Y_hi - Y_lo, Z_hi - Z_lo);
    dT = Kokkos::View<double***, Kokkos::HIP>("System::dT", T.extent(0), T.extent(1), T.extent(2));
    Kokkos::deep_copy(T, T0);

    // incoming halos
    if (X_lo != 0) T_left = buffer_t("System::T_left", Y_hi - Y_lo, Z_hi - Z_lo);
    if (X_hi != X) T_right = buffer_t("System::T_right", Y_hi - Y_lo, Z_hi - Z_lo);
    if (Y_lo != 0) T_down = buffer_t("System::T_down", X_hi - X_lo, Z_hi - Z_lo);
    if (Y_hi != Y) T_up = buffer_t("System::T_up", X_hi - X_lo, Z_hi - Z_lo);
    if (Z_lo != 0) T_front = buffer_t("System::T_front", X_hi - X_lo, Y_hi - Y_lo);
    if (Z_hi != Z) T_back = buffer_t("System::T_back", X_hi - X_lo, Y_hi - Y_lo);

    // outgoing halo
    if (X_lo != 0) T_left_out = buffer_t("System::T_left_out", Y_hi - Y_lo, Z_hi - Z_lo);
    if (X_hi != X) T_right_out = buffer_t("System::T_right_out", Y_hi - Y_lo, Z_hi - Z_lo);
    if (Y_lo != 0) T_down_out = buffer_t("System::T_down_out", X_hi - X_lo, Z_hi - Z_lo);
    if (Y_hi != Y) T_up_out = buffer_t("System::T_up_out", X_hi - X_lo, Z_hi - Z_lo);
    if (Z_lo != 0) T_front_out = buffer_t("System::T_front_out", X_hi - X_lo, Y_hi - Y_lo);
    if (Z_hi != Z) T_back_out = buffer_t("System::T_back_out", X_hi - X_lo, Y_hi - Y_lo);
  }

  // run_time_loops
  void timestep() {
    Kokkos::Timer timer;
    double old_time = 0.0;
    double time_all = 0.0;
    double GUPs     = 0.0;
    double time_a, time_b, time_c, time_d;
    double time_inner, time_surface, time_update;
    time_inner = time_surface = time_update = 0.0;
    for (int t = 0; t <= N; t++) {
      if (t > N / 2) P = 0.0;
      time_a = timer.seconds();
      pack_T_halo();       // Overlap O1
      compute_inner_dT();  // Overlap O1
      Kokkos::fence();
      time_b = timer.seconds();
      exchange_T_halo();
      compute_surface_dT();
      Kokkos::fence();
      time_c       = timer.seconds();
      double T_ave = update_T();
      time_d       = timer.seconds();
      time_inner += time_b - time_a;
      time_surface += time_c - time_b;
      time_update += time_d - time_c;
      T_ave /= 1e-9 * (X * Y * Z);
      if ((t % I == 0 || t == N) && (comm.me == 0)) {
        double time = timer.seconds();
        time_all += time - old_time;
        GUPs += 1e-9 * (dT.size() / time_inner);
        if ((t == N) && (comm.me == 0)) {
          printf(
              "heat3D,Kokkos+KC_MPI,%i,%i,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%i,%f\n", comm.nranks, t, T_ave, time_inner,
              time_surface, time_update, time - old_time, /* time last iter */
              time_all,                                   /* current runtime  */
              GUPs / t, X, 1e-6 * (X * sizeof(double))
          );
          old_time = time;
        }
      }
    }
  }

  void compute_inner_dT() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, int>;
    int myX        = T.extent(0);
    int myY        = T.extent(1);
    int myZ        = T.extent(2);
    Kokkos::parallel_for(
        "InnerDT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {1, 1, 1}, {myX - 1, myY - 1, myZ - 1}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight
        ),
        InnerDT{T, dT, q}
    );
  };

  void pack_T_halo() {
    if (X_lo != 0) {
      Kokkos::deep_copy(E_left, T_left_out, Kokkos::subview(T, 0, Kokkos::ALL, Kokkos::ALL));
    }
    if (Y_lo != 0) {
      Kokkos::deep_copy(E_down, T_down_out, Kokkos::subview(T, Kokkos::ALL, 0, Kokkos::ALL));
    }
    if (Z_lo != 0) {
      Kokkos::deep_copy(E_front, T_front_out, Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, 0));
    }
    if (X_hi != X) {
      Kokkos::deep_copy(E_right, T_right_out, Kokkos::subview(T, X_hi - X_lo - 1, Kokkos::ALL, Kokkos::ALL));
    }
    if (Y_hi != Y) {
      Kokkos::deep_copy(E_up, T_up_out, Kokkos::subview(T, Kokkos::ALL, Y_hi - Y_lo - 1, Kokkos::ALL));
    }
    if (Z_hi != Z) {
      Kokkos::deep_copy(E_back, T_back_out, Kokkos::subview(T, Kokkos::ALL, Kokkos::ALL, Z_hi - Z_lo - 1));
    }
  }

  void setup_halo_exchange(){
    active_reqs = 0;
    if (X_lo != 0) {
      KokkosComm::Experimental::stream::recv(T_left, comm.left, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
      KokkosComm::Experimental::stream::send(T_left_out, comm.left, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
    }
    if (Y_lo != 0) {
      KokkosComm::Experimental::stream::recv(T_down, comm.down, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
      KokkosComm::Experimental::stream::send(T_down_out, comm.down, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
    }
    if (Z_lo != 0) {
      KokkosComm::Experimental::stream::recv(T_front, comm.front, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
      KokkosComm::Experimental::stream::send(T_front_out, comm.front, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
    }
    if (X_hi != X) {
      KokkosComm::Experimental::stream::recv(T_right, comm.right, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
      KokkosComm::Experimental::stream::send(T_right_out, comm.right, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
    }
    if (Y_hi != Y) {
      KokkosComm::Experimental::stream::recv(T_up, comm.up, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
      KokkosComm::Experimental::stream::send(T_up_out, comm.up, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
    }
    if (Z_hi != Z) {
      KokkosComm::Experimental::stream::recv(T_back, comm.back, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
      KokkosComm::Experimental::stream::send(T_back_out, comm.back, 0, comm.comm, comm.ctx.get_mem_info(), &reqs_[active_reqs++]);
    }
    MPIS_Matchall(active_reqs, reqs_, MPI_STATUS_IGNORE);
  }

  void exchange_T_halo() {
    MPIS_Enqueue_startall(comm.ctx.get_queue(), active_reqs, reqs_);
  }

  void teardown_halo_exchange() {
    for (int i = 0; i < active_reqs; ++i) MPIS_Request_free(&reqs_[i]);
  }

  void compute_surface_dT() {
    MPIS_Enqueue_waitall(comm.ctx.get_queue());
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<2>, int>;
    int x          = T.extent(0);
    int y          = T.extent(1);
    int z          = T.extent(2);

    SurfaceDT<left> f_left{T,    dT,   T_left, T_right, T_up, T_down, T_front, T_back, X_lo,  Y_lo,
                           Z_lo, X_hi, Y_hi,   Z_hi,    X,    Y,      Z,       q,      sigma, P};
    SurfaceDT<right> f_right{T,    dT,   T_left, T_right, T_up, T_down, T_front, T_back, X_lo,  Y_lo,
                             Z_lo, X_hi, Y_hi,   Z_hi,    X,    Y,      Z,       q,      sigma, P};
    SurfaceDT<down> f_down{T,    dT,   T_left, T_right, T_up, T_down, T_front, T_back, X_lo,  Y_lo,
                           Z_lo, X_hi, Y_hi,   Z_hi,    X,    Y,      Z,       q,      sigma, P};
    SurfaceDT<up> f_up{T,    dT,   T_left, T_right, T_up, T_down, T_front, T_back, X_lo,  Y_lo,
                       Z_lo, X_hi, Y_hi,   Z_hi,    X,    Y,      Z,       q,      sigma, P};
    SurfaceDT<front> f_front{T,    dT,   T_left, T_right, T_up, T_down, T_front, T_back, X_lo,  Y_lo,
                             Z_lo, X_hi, Y_hi,   Z_hi,    X,    Y,      Z,       q,      sigma, P};
    SurfaceDT<back> f_back{T,    dT,   T_left, T_right, T_up, T_down, T_front, T_back, X_lo,  Y_lo,
                           Z_lo, X_hi, Y_hi,   Z_hi,    X,    Y,      Z,       q,      sigma, P};

    Kokkos::parallel_for(
        "ComputeSurfaceDT_Left",
        Kokkos::Experimental::require(
            policy_t(E_left, {0, 0}, {y, z}), Kokkos::Experimental::WorkItemProperty::HintLightWeight
        ),
        f_left
    );
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Right",
        Kokkos::Experimental::require(
            policy_t(E_right, {0, 0}, {y, z}), Kokkos::Experimental::WorkItemProperty::HintLightWeight
        ),
        f_right
    );
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Down",
        Kokkos::Experimental::require(
            policy_t(E_down, {1, 0}, {x - 1, z}), Kokkos::Experimental::WorkItemProperty::HintLightWeight
        ),
        f_down
    );
    Kokkos::parallel_for(
        "ComputeSurfaceDT_Up",
        Kokkos::Experimental::require(
            policy_t(E_up, {1, 0}, {x - 1, z}), Kokkos::Experimental::WorkItemProperty::HintLightWeight
        ),
        f_up
    );
    Kokkos::parallel_for(
        "ComputeSurfaceDT_front",
        Kokkos::Experimental::require(
            policy_t(E_front, {1, 1}, {x - 1, y - 1}), Kokkos::Experimental::WorkItemProperty::HintLightWeight
        ),
        f_front
    );
    Kokkos::parallel_for(
        "ComputeSurfaceDT_back",
        Kokkos::Experimental::require(
            policy_t(E_back, {1, 1}, {x - 1, y - 1}), Kokkos::Experimental::WorkItemProperty::HintLightWeight
        ),
        f_back
    );
  }

  double update_T() {
    using policy_t = Kokkos::MDRangePolicy<Kokkos::Rank<3>, Kokkos::IndexType<int>>;
    int x          = T.extent(0);
    int y          = T.extent(1);
    int z          = T.extent(2);
    double my_T    = 0.0;
    Kokkos::parallel_reduce(
        "UpdateT",
        Kokkos::Experimental::require(
            policy_t(E_bulk, {0, 0, 0}, {x, y, z}, {10, 10, 10}),
            Kokkos::Experimental::WorkItemProperty::HintLightWeight
        ),
        UpdateT(T, dT, dt), my_T
    );
    double sum_T;
    MPI_Allreduce(&my_T, &sum_T, 1, MPI_DOUBLE, MPI_SUM, comm.comm);
    return sum_T;
  }
};

void benchmark_heat3d_kc_st(benchmark::State& state) {
  while (state.KeepRunning()) {
    auto start = std::chrono::high_resolution_clock::now();
    SystemKC sys(MPI_COMM_WORLD);
    sys.setup_subdomain();
    sys.setup_halo_exchange();
    sys.timestep();
    sys.teardown_halo_exchange();
    sys.destroy_exec_spaces();
    auto end             = std::chrono::high_resolution_clock::now();
    auto elapsed_seconds = std::chrono::duration_cast<std::chrono::duration<double>>(end - start);
    state.SetIterationTime(elapsed_seconds.count());
  }
}

BENCHMARK(benchmark_heat3d_kc_st)
    ->Iterations(1)
    ->Repetitions(10)
    ->ReportAggregatesOnly(false)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond);