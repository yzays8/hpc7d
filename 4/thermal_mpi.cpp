#include <cstdio>
#include <fstream>
#include <iostream>
#include <vector>
#include <mpi.h>

constexpr int L = 128;
constexpr int STEP = 100000;
constexpr int DUMP = 1000;

void onestep(std::vector<double>& lattice, const double h, int rank, int procs) {
  const int size = lattice.size();
  static std::vector<double> orig(size);
  std::copy(lattice.begin(), lattice.end(), orig.begin());
  const int left = (rank - 1 + procs) % procs;
  const int right = (rank + 1) % procs;
  MPI_Status st;
  MPI_Sendrecv(&(lattice[size - 2]), 1, MPI_DOUBLE, right, 0, &(orig[0]), 1, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &st);
  MPI_Sendrecv(&(lattice[1]), 1, MPI_DOUBLE, left, 0, &(orig[size - 1]), 1, MPI_DOUBLE, right, 0, MPI_COMM_WORLD, &st);

  for (int i = 1; i < size - 1; ++i) {
    lattice[i] += h * (orig[i - 1] - 2.0 * orig[i] + orig[i + 1]);
  }
}

void dump(std::vector<double>& data) {
  static int index = 0;
  char filename[256];
  sprintf(filename, "data%03d.dat", index);
  std::cout << filename << std::endl;
  std::ofstream ofs{filename};
  for (unsigned int i = 0; i < data.size(); ++i) {
    ofs << i << " " << data[i] << std::endl;
  }
  ++index;
}

void dump_mpi(std::vector<double>& local, int rank, int procs) {
  static std::vector<double> global(L);
  MPI_Gather(&(local[1]), L / procs, MPI_DOUBLE, global.data(), L / procs, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  if (rank == 0) {
    dump(global);
  }
}

void fixed_temperature(std::vector<double>& lattice, int rank, int procs) {
  const double h = 0.01;
  const double Q = 1.0;
  const int s = L / procs;
  for (int i = 0; i < STEP; ++i) {
    onestep(lattice, h, rank, procs);
    if (rank == (L / 4 / s)) {
      lattice[L / 4 - rank * s + 1] = Q;
    }
    if (rank == (3 * L / 4 / s)) {
      lattice[3 * L / 4 - rank * s + 1] = -Q;
    }
    if ((i % DUMP) == 0) dump_mpi(lattice, rank, procs);
  }
}

void uniform_heating(std::vector<double>& lattice, int rank, int procs) {
  const double h = 0.2;
  const double Q = 1.0;
  for (int i = 0; i < STEP; ++i) {
    onestep(lattice, h, rank, procs);
    for (auto& s : lattice) {
      s += Q * h;
    }
    if (rank == 0) {
      lattice[1] = 0.0;
    }
    if (rank == procs - 1) {
      lattice[lattice.size() - 2] = 0.0;
    }
    if ((i % DUMP) == 0) dump_mpi(lattice, rank, procs);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  const int mysize = L / procs + 2;
  std::vector<double> local(mysize);
  uniform_heating(local, rank, procs);
  // fixed_temperature(local, rank, procs);
  MPI_Finalize();
}