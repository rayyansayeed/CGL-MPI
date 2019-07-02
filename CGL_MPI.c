#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <complex.h>
#include <fftw3-mpi.h>
#include <mpi.h>

double M_PI = 3.141592;

/* % A = cgl(N,omega,tol)
% Pseudo-spectral solution of complex Ginsburg-Landau equation
% N = number of grid points in both dimensions
% c1,c3 = equation coefficients
% M = number of time steps */

void fftshift(ptrdiff_t, double complex *);

// For seeding

extern void srand48();
extern double drand48();

int main(int argc, char *argv[]) {

    // MPI initialization

	MPI_Init(&argc, &argv);
	int rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int size;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	fftw_mpi_init();

	// Start runtime clock

	clock_t start_time = clock();

	// Set up data to be taken as inputs

	// N = dimension of square grid

	ptrdiff_t N = atoi(argv[1]);

	// c1, c3 are equation coefficients, set up to be taken as input

	double c_1 = atof(argv[2]);
	double c_3 = atof(argv[3]);

    // M = # timesteps

	int M = atoi(argv[4]);

	// Seeding

	long int seed = (long int)time(NULL);
	if (argc >= 6) {
		seed = atol(argv[5]);
	}
	srand48(seed);

	// Set parameter values

	double L = 128 * M_PI;
	int T = 10000;
	double dt =(double) T / M;

	// Allocate memory

	ptrdiff_t localM, local0;
	ptrdiff_t alloc_local = fftw_mpi_local_size_2d(N, N, MPI_COMM_WORLD, &localM, &local0);

	fftw_complex* A = fftw_alloc_complex(alloc_local);
	fftw_complex* A1 = fftw_alloc_complex(alloc_local);
	fftw_complex* d2A = fftw_alloc_complex(alloc_local);
	fftw_complex* d2A_local = fftw_alloc_complex(alloc_local);
	fftw_complex* d2A_temp = fftw_alloc_complex(alloc_local);

	double complex * d2A_global = malloc(N*N * sizeof(*d2A_global));
	double complex *A_soln = malloc(N*N * sizeof(*A_soln));

	// FFT shift function

	void fftshift(ptrdiff_t N, double complex *A) {

	    // 1st half

	for (int j = 0; j < N / 2; ++j){
		for (int i = 0; i < N / 2; ++i)
		{
			double complex val = A[j*N + i];
			A[j*N + i] = A[(j + N / 2)*N + i + N / 2];
			A[(j + N / 2)*N + i + N / 2] = val;
		}
	}

	for (int j = N / 2; j < N; ++j){
        // 2nd half
		for (int i = 0; i < N / 2; ++i)
		{
			double complex val = A[j*N + i];
			A[j*N + i] = A[(j - N / 2)*N + i + N / 2];
			A[(j - N / 2)*N + i + N / 2] = val;
		}
	}
}

	//  IC step

	for (int i = 0; i < N; ++i){
		for (int j = 0; j < N; ++j)
		{
		    // Initialize A with random data
			A_soln[i*N + j] = 3 * drand48() - 1.5 + I * (3 * drand48() - 1.5);
		}
	}

	// Write result of each step to A_soln

	for (int i = 0; i < localM; ++i){
		for (int j = 0; j < N; ++j)
		{
			A[i*N + j] = A_soln[(i + local0)*N + j];
		}
	}

	//  FFTW

	fftw_plan plan_for = fftw_mpi_plan_dft_2d(N, N, d2A_local, d2A_temp, MPI_COMM_WORLD, FFTW_FORWARD, FFTW_ESTIMATE);
	fftw_plan plan_back = fftw_mpi_plan_dft_2d(N, N, d2A_local, d2A_temp, MPI_COMM_WORLD, FFTW_BACKWARD, FFTW_ESTIMATE);

	// Create and open target file

	FILE *fileid = fopen("CGL_MPI.out", "w");

	/////////////////////////////////////

	// BEGIN MAIN LOOP

	for (int n = 1; n < M; ++n) {

		////////////// Begin RK4, 1st step

		// Write data to d2A_local
		for (int i = 0; i < localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[i*N + j] = A[i*N + j];
			}
		}

        //////////// 2nd deriv

		// Fourier transform

		fftw_execute(plan_for);

		//  Gather

		MPI_Allgather(d2A_temp, 2 * localM*N, MPI_DOUBLE, d2A_global, 2 * localM*N, MPI_DOUBLE, MPI_COMM_WORLD);

		//  FFT shift

		fftshift(N, d2A_global);

		//  2nd deriv in freq

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[(i - local0)*N + j] = -pow((-N / 2 + j), 2)*d2A_global[i*N + j]
					- pow((-N / 2 + i), 2)*d2A_global[i*N + j];
			}
		}

		//  Local to global gather

		MPI_Allgather(d2A_local, 2 * localM*N, MPI_DOUBLE, d2A_global, 2 * localM*N, MPI_DOUBLE, MPI_COMM_WORLD);

		//  FFT shift again

		fftshift(N, d2A_global);

		//  Write back from global to local, /N^2

		for (int i = local0; i < local0 + localM; ++i)
        {
			for (int j = 0; j < N; ++j)
			{
				d2A_local[(i - local0)*N + j] = d2A_global[i*N + j] / (N*N);
			}
		}

		// Backward FFTW

		fftw_execute(plan_back);

		// Local to global gather

		MPI_Allgather(d2A_temp, 2 * localM*N, MPI_COMPLEX, d2A_global, 2 * localM*N, MPI_COMPLEX, MPI_COMM_WORLD);

        ///////////// End 2nd deriv

        ///////////// End RK4, 1st step


		///////////// Begin update to RK4, 1st step

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				A1[(i - local0)*N + j] = A[(i - local0)*N + j] + dt / 4 * (A[(i - local0)*N + j]
					+ (1 + I*c_1)*(2*M_PI/L)*(2*M_PI/L)*d2A_global[(i - local0)*N + j]
					- (1 - I*c_3)*cabs(A[(i - local0)*N + j])*cabs(A[(i - local0)*N + j])*A[(i - local0)*N + j]);
			}
		}

		///////////// End update to RK4, 1st step


		///////////// Begin RK4, 2nd step

		// Write data to d2A_local

		for (int i = 0; i < localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[i*N + j] = A1[i*N + j];
			}
		}

        /////////// 2nd deriv

		//  Fourier transform

		fftw_execute(plan_for);

		//  Gather data

		MPI_Allgather(d2A_temp, 2 * localM*N, MPI_DOUBLE, d2A_global, 2 * localM*N, MPI_DOUBLE, MPI_COMM_WORLD);

		//  FFT shift

		fftshift(N, d2A_global);

		//  2nd deriv in freq

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[(i - local0)*N + j] = -pow((-N / 2 + j), 2)*d2A_global[i*N + j]
					- pow((-N / 2 + i), 2)*d2A_global[i*N + j];
			}
		}

		//  Local to global gather

		MPI_Allgather(d2A_local, 2 * localM*N, MPI_DOUBLE, d2A_global, 2 * localM*N, MPI_DOUBLE, MPI_COMM_WORLD);

		//  FFT shift

		fftshift(N, d2A_global);

		//  Write back from global to local, /N^2

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[(i - local0)*N + j] = d2A_global[i*N + j] / (N*N);
			}
		}

		//  Backward FFTW

		fftw_execute(plan_back);

		// Local to global gather

		MPI_Allgather(d2A_temp, 2 * localM*N, MPI_COMPLEX, d2A_global, 2 * localM*N, MPI_COMPLEX, MPI_COMM_WORLD);

        ///////// End 2nd deriv

        ///////////// End RK4, 2nd step


		///////////// Begin update to RK4, 2nd step

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				A1[(i - local0)*N + j] = A[(i - local0)*N + j] + dt / 3 * (A1[(i - local0)*N + j]
					+ (1 + I * c_1)*(2 * M_PI / L)*(2 * M_PI / L)*d2A_global[(i - local0)*N + j]
					- (1 - I * c_3)*cabs(A1[(i - local0)*N + j])*cabs(A1[(i - local0)*N + j])*A1[(i - local0)*N + j]);
			}
		}

		///////////// End update to RK4, 2nd step



		///////////// Begin RK4, 3rd step

		// Write data to d2A_local

		for (int i = 0; i < localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[i*N + j] = A1[i*N + j];
			}
		}

        /////////// 2nd deriv

		//  Forward FFTW

		fftw_execute(plan_for);

		//  Gather data

		MPI_Allgather(d2A_temp, 2 * localM*N, MPI_DOUBLE, d2A_global, 2 * localM*N, MPI_DOUBLE, MPI_COMM_WORLD);

		//  FFT shift

		fftshift(N, d2A_global);

		//  2nd deriv in freq

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[(i - local0)*N + j] = -pow((-N / 2 + j), 2)*d2A_global[i*N + j]
					- pow((-N / 2 + i), 2)*d2A_global[i*N + j];
			}
		}

		//  Local to global gather

		MPI_Allgather(d2A_local, 2 * localM*N, MPI_DOUBLE, d2A_global, 2 * localM*N, MPI_DOUBLE, MPI_COMM_WORLD);

		//  FFT shift

		fftshift(N, d2A_global);

		//  Write back from global to local, /N^2

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[(i - local0)*N + j] = d2A_global[i*N + j] / (N*N);
			}
		}

		// Backward FFTW

		fftw_execute(plan_back);

		// Local to global gather

		MPI_Allgather(d2A_temp, 2 * localM*N, MPI_COMPLEX, d2A_global, 2 * localM*N, MPI_COMPLEX, MPI_COMM_WORLD);

		///////////// End 2nd deriv

        ///////////// End RK4, 3rd step


		///////////// Begin update to RK4, 3rd step

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				A1[(i - local0)*N + j] = A[(i - local0)*N + j] + dt / 2 * (A1[(i - local0)*N + j]
					+ (1 + I * c_1)*(2 * M_PI / L)*(2 * M_PI / L)*d2A_global[(i - local0)*N + j]
					- (1 - I * c_3)*cabs(A1[(i - local0)*N + j])*cabs(A1[(i - local0)*N + j])*A1[(i - local0)*N + j]);
			}
		}
        ///////////// End update to RK4, 3rd step


        ///////////// Begin RK4, 4th step

		// Write data to d2A_local

		for (int i = 0; i < localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[i*N + j] = A1[i*N + j];
			}
		}

		///////// 2nd deriv

		//  Forward FFTW

		fftw_execute(plan_for);

		//  Gather data

		MPI_Allgather(d2A_temp, 2 * localM*N, MPI_DOUBLE, d2A_global, 2 * localM*N, MPI_DOUBLE, MPI_COMM_WORLD);

		//  FFT shift

		fftshift(N, d2A_global);

		//  2nd deriv in freq

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[(i - local0)*N + j] = -pow((-N / 2 + j), 2)*d2A_global[i*N + j]
					- pow((-N / 2 + i), 2)*d2A_global[i*N + j];
			}
		}

		//  Local to global gather

		MPI_Allgather(d2A_local, 2 * localM*N, MPI_DOUBLE, d2A_global, 2 * localM*N, MPI_DOUBLE, MPI_COMM_WORLD);

		//  FFT shift

		fftshift(N, d2A_global);

		//  Write back from global to local, /N^2

		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				d2A_local[(i - local0)*N + j] = d2A_global[i*N + j] / (N*N);
			}
		}

		//  Backward FFTW

		fftw_execute(plan_back);

		// Local to global gather

		MPI_Allgather(d2A_temp, 2 * localM*N, MPI_COMPLEX, d2A_global, 2 * localM*N, MPI_COMPLEX, MPI_COMM_WORLD);

		///////////// End 2nd deriv

        ///////////// End RK4, 4th step


		///////////// Begin update to RK4, 4th step (final step)
		for (int i = local0; i < local0 + localM; ++i)
		{
			for (int j = 0; j < N; ++j)
			{
				A[(i - local0)*N + j] = A[(i - local0)*N + j] + dt * (A1[(i - local0)*N + j]
					+ (1 + I * c_1)*(2 * M_PI / L)*(2 * M_PI / L)*d2A_global[(i - local0)*N + j]
					- (1 - I * c_3)*cabs(A1[(i - local0)*N + j])*cabs(A1[(i - local0)*N + j])*A1[(i - local0)*N + j]);
			}
		}
		///////////// End update to RK4, 4th step

		// Output solution, 10 sets

		if ((n+1) % ((N+1)/10) == 0)
        {
			MPI_Allgather(A, 2 * localM*N, MPI_COMPLEX, A_soln, 2 * localM*N, MPI_COMPLEX, MPI_COMM_WORLD);
			if (rank == 0)
			{
				fwrite(A_soln, sizeof(double complex), N*N, fileid);
			}
		}
	}

	// END MAIN LOOP

	// Gather data

	MPI_Allgather(A, 2 * localM*N, MPI_COMPLEX, A_soln, 2 * localM*N, MPI_COMPLEX, MPI_COMM_WORLD);
	if (rank == 0)
	{
		fwrite(A_soln, sizeof(double complex), N*N, fileid);
		fclose(fileid);
	}

	// Free all memory used

	free(A);
	free(A1);
	free(d2A);
	free(d2A_local);
	free(d2A_global);
	free(d2A_temp);
	free(A_soln);
	fftw_destroy_plan(plan_for);
	fftw_destroy_plan(plan_back);

	// End clock

	clock_t end_time = clock();
    MPI_Finalize();

	// Output runtime

	if (rank == 0){
        printf("Runtime:%g s.\n", (float)(end_time - start_time)/CLOCKS_PER_SEC);
	return 0;
	}
}
