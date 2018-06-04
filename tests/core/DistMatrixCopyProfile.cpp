/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
using namespace El;

template <typename T, Device D>
void DoCopy(Int m, Int n, Grid const& grid)
{
    DistMatrix<T,STAR,VC,ELEMENT,D> A(grid);
    DistMatrix<T,MC,MR,ELEMENT,D> B(grid);

    const T center = 0;
    const Base<T> radius = 5;
    Uniform(A, m, n, center, radius);

    B = A;
}

int
main(int argc, char* argv[])
{
    Environment env(argc, argv);
    mpi::Comm comm = mpi::COMM_WORLD;

    try
    {
        int gridHeight = Input("--gridHeight","height of process grid",0);
        const Int m = Input("--m","height of matrix",50);
        const Int n = Input("--n","width of matrix",50);
        const Int count = Input("--count","number of times to loop", 10);
        ProcessInput();
        PrintInputReport();

        if (gridHeight == 0)
            gridHeight = Grid::DefaultHeight(mpi::Size(comm));
        const Grid grid(comm, gridHeight, COLUMN_MAJOR);

        OutputFromRoot(comm,
                       "Grid = ", grid.Height(), "x", grid.Width(), "\n");

        for (Int ii = 0; ii < count; ++ii)
        {
            DoCopy<float,Device::GPU>(m, n, grid);
            DoCopy<double,Device::GPU>(m, n, grid);
        }
    }
    catch(std::exception& e) { ReportException(e); }

    return 0;
}
