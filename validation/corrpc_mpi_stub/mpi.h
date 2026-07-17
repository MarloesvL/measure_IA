// Single-process stub for the deprecated MPI C++ bindings used by corr_pc
// (MPI::Init, MPI::Finalize, MPI::COMM_WORLD.Get_size/Get_rank only).
#ifndef MPI_STUB_H
#define MPI_STUB_H

namespace MPI {

inline void Init(int &, char **&) {}
inline void Finalize() {}

struct Comm {
	int Get_size() const { return 1; }
	int Get_rank() const { return 0; }
};

inline Comm COMM_WORLD;

}  // namespace MPI

#endif
