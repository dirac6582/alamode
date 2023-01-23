/*
 dielec.cpp

 Copyright (c) 2019 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/

#include "mpi_common.h"
#include "dielec.h"
#include "constants.h"
#include "dynamical.h"
#include "error.h"
#include "mathfunctions.h"
#include "memory.h"
#include "system.h"
#include "write_phonons.h"
#include "parsephon.h"
#include "phonon_dos.h"
#include "fcs_phonon.h"
#include "ewald.h" // add by amano for nonanalytic=3 calculation in mode_analysis.cpp
#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <complex>

using namespace PHON_NS;

Dielec::Dielec(PHON *phon) : Pointers(phon)
{
    set_default_variables();
}

Dielec::~Dielec()
{
    deallocate_variables();
}

void Dielec::set_default_variables()
{
    calc_dielectric_constant = 0;
    dielec = nullptr;
    omega_grid = nullptr;
    emin = 0.0;
    emax = 1.0;
    delta_e = 1.0;
    nomega = 1;
}

void Dielec::deallocate_variables()
{
    if (dielec) {
        deallocate(dielec);
    }
    if (omega_grid) {
        deallocate(omega_grid);
    }
}

void Dielec::init()
{
    // This should be called after Dos::setup()

    if (mympi->my_rank == 0) {
        emax = dos->emax;
        emin = dos->emin;
        delta_e = dos->delta_e;
        nomega = static_cast<int>((emax - emin) / delta_e);
    }

    MPI_Bcast(&calc_dielectric_constant, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nomega, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);
    MPI_Bcast(&emin, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&emax, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&delta_e, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (calc_dielectric_constant) {

        if (mympi->my_rank == 0) {
            if (dynamical->file_born == "") {
                exitall("Dielec::init()", "BORNINFO must be set when DIELEC = 1.");
            }
        }

        allocate(omega_grid, nomega);

        for (auto i = 0; i < nomega; ++i) {
            omega_grid[i] = emin + delta_e * static_cast<double>(i);
        }

        // If borncharge in dynamical class is not initialized, do it here.
        if (!dynamical->borncharge) {
            const auto verbosity_level = 1;
            dynamical->setup_dielectric(verbosity_level);
        }
    }
}

double *Dielec::get_omega_grid(unsigned int &nomega_in) const
{
    nomega_in = nomega;
    return omega_grid;
}

double ***Dielec::get_dielectric_func() const
{
    return dielec;
}

void Dielec::run_dielec_calculation()
{
    double *xk;
    double *eval;
    std::complex<double> **evec;
    const auto ns = dynamical->neval;

    allocate(xk, 3);
    allocate(eval, ns);
    allocate(evec, ns, ns);
    allocate(dielec, nomega, 3, 3);

    for (auto i = 0; i < 3; ++i) xk[i] = 0.0;

    dynamical->eval_k(xk, xk, fcs_phonon->fc2_ext, eval, evec, true);

    compute_dielectric_function(nomega, omega_grid,
                                eval, evec, dielec);

    deallocate(xk);
    deallocate(eval);
    deallocate(evec);
}

void Dielec::compute_dielectric_function(const unsigned int nomega_in,
                                         double *omega_grid_in,
                                         double *eval_in,
                                         std::complex<double> **evec_in,
                                         double ***dielec_out)
{
    const auto ns = dynamical->neval;
    const auto zstar = dynamical->borncharge;

#ifdef _DEBUG
    for (auto i = 0; i < ns; ++i) {
        std::cout << "eval = " << eval_in[i] << std::endl;
        for (auto j = 0; j < ns; ++j) {
            std::cout << std::setw(15) << evec_in[i][j].real();
            std::cout << std::setw(15) << evec_in[i][j].imag();
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif

    for (auto i = 0; i < ns; ++i) {
        for (auto j = 0; j < ns; ++j) {
            evec_in[i][j] /= std::sqrt(system->mass[system->map_p2s[j / 3][0]]);
        }
    }

#ifdef _DEBUG
    for (auto i = 0; i < ns; ++i) {
        std::cout << "U = " << eval_in[i] << std::endl;
        for (auto j = 0; j < ns; ++j) {
            std::cout << std::setw(15) << real(evec_in[i][j]);
            std::cout << std::setw(15) << evec_in[i][j].imag();
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
#endif

    double ***s_born;
    double **zstar_u;

    allocate(zstar_u, 3, ns);
    allocate(s_born, 3, 3, ns);

    for (auto i = 0; i < 3; ++i) {
        for (auto is = 0; is < ns; ++is) {
            zstar_u[i][is] = 0.0;

            for (auto j = 0; j < ns; ++j) {
                zstar_u[i][is] += zstar[j / 3][i][j % 3] * evec_in[is][j].real();
            }
        }
    }

#ifdef _DEBUG
    std::cout << "Zstar_u:\n";
    for (auto is = 0; is < ns; ++is) {
        for (auto i = 0; i < 3; ++i) {
            std::cout << std::setw(15) << zstar_u[i][is];
        }
        std::cout << '\n';
    }
    std::cout << std::endl;

    std::cout << "S_born:\n";
    for (auto is = 0; is < ns; ++is) {
        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                std::cout << std::setw(15) << s_born[i][j][is];
            }
            std::cout << '\n';
        }
        std::cout << '\n';
    }

#endif

    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto is = 0; is < ns; ++is) {
                s_born[i][j][is] = zstar_u[i][is] * zstar_u[j][is];
            }
        }
    }

    auto freq_conv_factor = time_ry * time_ry / (Hz_to_kayser * Hz_to_kayser);
    auto factor = 8.0 * pi / system->volume_p;
    double w2_tmp;
    for (auto iomega = 0; iomega < nomega_in; ++iomega) {
        w2_tmp = omega_grid_in[iomega] * omega_grid_in[iomega] * freq_conv_factor;

        for (auto i = 0; i < 3; ++i) {
            for (auto j = 0; j < 3; ++j) {
                dielec_out[iomega][i][j] = 0.0;

                for (auto is = 3; is < ns; ++is) {
                    dielec_out[iomega][i][j] += s_born[i][j][is] / (eval_in[is] - w2_tmp);
                }
                dielec_out[iomega][i][j] *= factor;
            }
        }
    }

    deallocate(zstar_u);
    deallocate(s_born);
}

std::vector<std::vector<double>> Dielec::get_zstar_mode() const
{
    const auto ns = dynamical->neval;
    std::vector<std::vector<double>> zstar_mode(ns, std::vector<double>(3));
    compute_mode_effective_charge(zstar_mode, false);
    return zstar_mode;
}


void Dielec::compute_mode_effective_charge(std::vector<std::vector<double>> &zstar_mode,
                                           const bool do_normalize) const
{
    // Compute the effective charges of normal coordinate at q = 0.

    if (mympi->my_rank == 0) {
        if (dynamical->file_born.empty()) {
            exitall("Dielec::compute_mode_effective_charge()",
                    "BORNINFO must be set when DIELEC = 1.");
        }
    }
    
    // If borncharge in dynamical class is not initialized, do it here.
    if (!dynamical->borncharge) {
        const auto verbosity_level = 0;
        dynamical->setup_dielectric(verbosity_level);
    }

    std::vector<double> xk(3);
    double *eval;
    std::complex<double> **evec;
    const auto ns = dynamical->neval;
    const auto zstar_atom = dynamical->borncharge;

    allocate(eval, ns);
    allocate(evec, ns, ns);

    for (auto i = 0; i < 3; ++i) xk[i] = 0.0;

    // Probably, I need to symmetrize the eigenvector here.
    std::vector<std::vector<double>> projectors;
    std::vector<double> vecs(3);

    if (!dynamical->get_projection_directions().empty()) {
        if (mympi->my_rank == 0) { 
            std::cout << " DEBUG :: degenerate eigenvectors !! " << "\n";
        }
        dynamical->project_degenerate_eigenvectors(system->lavec_p,
                                                   fcs_phonon->fc2_ext,
                                                   &xk[0],
                                                   dynamical->get_projection_directions(),
                                                   evec);
    } else {
        // eval_kというのはnonanalytic=0,1,2の時に有効．
        // nonanalytic=3の時にはeval_k_ewald関数を利用する必要がある．
        // 詳細はdynamical.cppのDynamical::get_eigenvalues_dymat関数を参照すること．
        if (dynamical->nonanalytic == 3) {
            if (mympi->my_rank == 0) { 
                std::cout << "CAUTION !! \n" ;
                std::cout << " we are using nonanalytic=3 calculation for mode effective charge. \n";
                std::cout << " This is basically meaningless, but sometimes eigenvectors from nonanalytic=3 and others are different.\n";
                std::cout << " So this setting is useful when you compare results with mode=phonon's. \n";
                std::cout << "\n";
            }
            dynamical->eval_k_ewald(&xk[0], &xk[0], ewald->fc2_without_dipole, eval, evec, true);
        } else {
            dynamical->eval_k(&xk[0], &xk[0], fcs_phonon->fc2_ext, eval, evec, true);
        }
    }

    // Divide by sqrt of atomic mass to get normal coordinate
    for (auto i = 0; i < ns; ++i) {
        for (auto j = 0; j < ns; ++j) { // amano :: caution !! evec[i][j] represents atom=i/3,cartesian=i%3, mode=j ?
            evec[i][j] /= std::sqrt(system->mass[system->map_p2s[j / 3][0]] / amu_ry);
//            evec[i][j] /= std::sqrt(system->mass[system->map_p2s[j / 3][0]]);
        }
    }

    // Compute the mode effective charges defined by Eq. (53) or its numerator of
    // Gonze & Lee, PRB 55, 10355 (1997).
    for (auto i = 0; i < 3; ++i) {
        for (auto is = 0; is < ns; ++is) {
            zstar_mode[is][i] = 0.0;
            auto normalization_factor = 0.0;

            for (auto j = 0; j < ns; ++j) {
                zstar_mode[is][i] += zstar_atom[j / 3][i][j % 3] * evec[is][j].real();
                normalization_factor += std::norm(evec[is][j]); // calculate norm of eigenvector(evec)
            }
            if (do_normalize) zstar_mode[is][i] /= std::sqrt(normalization_factor);
        }
    }

    // (for DEBUG) print eigen vectors
    // causion !! evec[is][j] represents mode=is, atom_id=j/3, cartesian_id=j%3
    if (mympi->my_rank == 0) { // output .zmode and .strength
        std::cout << " ns is ... " << ns << "\n";
        for (auto is = 0; is < ns; ++is) {
            std::cout << "mode = " << is << "\n";
            for (auto j = 0; j < ns; ++j) {
                if (j%3 == 0) std::cout << '\n';
                // std::cout << std::setw(5) << evec[is][j].real() << std::setw(5) << evec[is][j].real() << std::setw(5) << evec[is][j].real() << '\n';
                std::cout << std::setw(15) << evec[is][j].real();
            }
        }
    }
    deallocate(eval);
    deallocate(evec);
}

// add by me
void Dielec::compute_mode_oscillator_strength(double ***s_born) const
// calculate mode oscillator strength
{
    const auto ns = dynamical->neval;

    // calculate mode effective charge 
    std::vector<std::vector<double>> zstar_born(ns, std::vector<double>(3));
    compute_mode_effective_charge(zstar_born, false);

    // calculate mode oscillator strength into s_born
    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 3; ++j) {
            for (auto is = 0; is < ns; ++is) {
                s_born[i][j][is] = zstar_born[is][i] * zstar_born[is][j]; // not [i][is] unlike compute_dielectric_function
            }
        }
    }
}