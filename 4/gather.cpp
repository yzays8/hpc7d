#include <cstdio>
#include <vector>
#include <mpi.h>

constexpr int L = 8;

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, procs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  const int mysize = L / procs;

  // local data
  std::vector<int> local(mysize, rank);
  // global data to receive
  std::vector<int> global(L);
  // connection (collect at rank 0)
  MPI_Gather(local.data(), mysize, MPI_INT, global.data() , mysize, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < L; ++i) {
      printf("%d", global[i]);
    }
    printf("\n");
  }
  MPI_Finalize();
}