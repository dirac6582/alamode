/*
 selfenergy.cpp

 Copyright (c) 2014 Terumasa Tadano

 This file is distributed under the terms of the MIT license.
 Please see the file 'LICENCE.txt' in the root directory 
 or http://opensource.org/licenses/mit-license.php for information.
*/
#include <iomanip>
#include "mpi_common.h"
#include "selfenergy.h"
#include "constants.h"
#include "anharmonic_core.h"
#include "dynamical.h"
#include "kpoint.h"
#include "memory.h"
#include "thermodynamics.h"
#include "mathfunctions.h"
#include "integration.h"
#include "phonon_dos.h"

using namespace PHON_NS;

Selfenergy::Selfenergy(PHON *phon) : Pointers(phon)
{
}

Selfenergy::~Selfenergy()
{
}

void Selfenergy::setup_selfenergy()
{
    ns = dynamical->neval;
    epsilon = integration->epsilon; // 積分時のsmearing幅
}


void Selfenergy::mpi_reduce_complex(unsigned int N, 
                                    std::complex<double> *in_mpi,
                                    std::complex<double> *out) const
{
  /*
    mpi_reduce_complex          : 
    mpi_reduce関数は元々のくみこみ関数:
    comm内の全プロセスのsendbufに、opで指定した演算を施して、rootプロセスのrecvbufへ送る。右の図の例では、4つのプロセスがそれぞれ1、2、3、4という値を持っていて、これに「加算」という演算が施され（1＋2＋3＋4＝10）、その結果がプロセス0へ送られている。送受信に参加する全てのプロセスがMPI_Reduceをコールする必要があり、root、comm、opなどはその全てのプロセスが同じ値を指定しなければならない。
    input ::
    ----------
      N :: 配列の要素数
      in_mpi :: input配列（要素数N）
      out :: output配列
  */
    
#ifdef MPI_COMPLEX16
    MPI_Reduce(&in_mpi[0], &out[0], N, MPI_COMPLEX16, MPI_SUM, 0, MPI_COMM_WORLD);
#else
    unsigned int i;
    double *ret_mpi_re, *ret_mpi_im;
    double *ret_re, *ret_im;

    allocate(ret_mpi_re, N);
    allocate(ret_mpi_im, N);
    allocate(ret_im, N);
    allocate(ret_re, N);

    for (i = 0; i < N; ++i) {
        ret_mpi_re[i] = in_mpi[i].real();
        ret_mpi_im[i] = in_mpi[i].imag();
    }
    MPI_Reduce(&ret_mpi_re[0], &ret_re[0], N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&ret_mpi_im[0], &ret_im[0], N, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    for (i = 0; i < N; ++i) {
        out[i] = ret_re[i] + im * ret_im[i];
    }
    deallocate(ret_mpi_re);
    deallocate(ret_mpi_im);
    deallocate(ret_re);
    deallocate(ret_im);
#endif
}



void Selfenergy::selfenergy_tadpole(const unsigned int N,
                                    const double *T,
                                    const double omega,
                                    const unsigned int knum,
                                    const unsigned int snum,
                                    const KpointMeshUniform *kmesh_in,
                                    const double *const *eval_in,
                                    const std::complex<double> *const *const *evec_in,
                                    std::complex<double> *ret) const
{
    unsigned int i;
    unsigned int arr_cubic1[3], arr_cubic2[3];
    std::complex<double> *ret_mpi, *ret_tmp;
    double n2;
    const auto nk = kmesh_in->nk;

    arr_cubic1[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_cubic1[1] = ns * knum + snum;

    allocate(ret_mpi, N);
    allocate(ret_tmp, N);

    for (i = 0; i < N; ++i) ret[i] = std::complex<double>(0.0, 0.0);

    for (unsigned int is1 = 0; is1 < ns; ++is1) {
        arr_cubic1[2] = is1;
        arr_cubic2[0] = is1;
        auto omega1 = eval_in[0][is1];

        if (omega1 < eps8) continue;

        auto v3_tmp1 = anharmonic_core->V3(arr_cubic1);

        for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0);

        for (unsigned int ik2 = mympi->my_rank; ik2 < nk; ik2 += mympi->nprocs) {
            for (unsigned int is2 = 0; is2 < ns; ++is2) {
                arr_cubic2[1] = ns * ik2 + is2;
                arr_cubic2[2] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;

                auto v3_tmp2 = anharmonic_core->V3(arr_cubic2);
                const auto omega2 = eval_in[ik2][is2];

                if (omega2 < eps8) continue;

                for (i = 0; i < N; ++i) {
                    const auto T_tmp = T[i];
                    if (thermodynamics->classical) {
                        n2 = thermodynamics->fC(omega2, T_tmp);
                        ret_mpi[i] += v3_tmp2 * 2.0 * n2;
                    } else {
                        n2 = thermodynamics->fB(omega2, T_tmp);
                        ret_mpi[i] += v3_tmp2 * (2.0 * n2 + 1.0);
                    }
                }
            }
        }
        mpi_reduce_complex(N, ret_mpi, ret_tmp);

        for (i = 0; i < N; ++i) {
            ret[i] += ret_tmp[i] * v3_tmp1 / omega1;
        }
    }

    const auto factor = -1.0 / (static_cast<double>(nk) * std::pow(2.0, 3));
    for (i = 0; i < N; ++i) ret[i] *= factor;

    deallocate(ret_tmp);
    deallocate(ret_mpi);
}

void Selfenergy::selfenergy_a(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*
    Diagram (a):Bubble
    Matrix elements that appear : V3^2
    Computational cost          : O(N_k * N^2)
    --------------------
    input

     const unsigned int N       : 温度の数(len(T))
     const double *T,           : 温度の配列
     const double omega,        : 振動数
     const unsigned int knum,   : 
     const unsigned int snum,
     const KpointMeshUniform *kmesh_in, : inputから計算されるkmesh
     const double *const *eval_in,      : harmonicの振動数リスト
     const std::complex<double> *const *const *evec_in, : harmonicの固有ベクトル，使ってない
     std::complex<double> *ret) const   : これを返す
     
    */
    unsigned int i;
    unsigned int arr_cubic[3];
    double xk_tmp[3];
    std::complex<double> omega_sum[2];
    std::complex<double> *ret_mpi;

    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;
    double n1, n2;
    double f1, f2;

    arr_cubic[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;

    std::complex<double> omega_shift = omega + im * epsilon; //虚数を加える.
        if (mympi->my_rank == 0) {
        std::cout << " (bubble) alamodeの振動数は "  << std::setprecision(16) << omega << std::endl;
        }
    // 温度Nに関するMPI(openMPではない！)
    allocate(ret_mpi, N);
    
    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0); //返す自己エネルギーを0で初期化

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) { //

        xk_tmp[0] = xk[knum][0] - xk[ik1][0];
        xk_tmp[1] = xk[knum][1] - xk[ik1][1];
        xk_tmp[2] = xk[knum][2] - xk[ik1][2];

        const auto ik2 = kmesh_in->get_knum(xk_tmp);

        for (unsigned int is1 = 0; is1 < ns; ++is1) {

            arr_cubic[1] = ns * ik1 + is1;
            double omega1 = eval_in[ik1][is1];

            for (unsigned int is2 = 0; is2 < ns; ++is2) {

                arr_cubic[2] = ns * ik2 + is2;
                double omega2 = eval_in[ik2][is2];

                double v3_tmp = std::norm(anharmonic_core->V3(arr_cubic));

                omega_sum[0] = 1.0 / (omega_shift + omega1 + omega2) - 1.0 / (omega_shift - omega1 - omega2);
                omega_sum[1] = 1.0 / (omega_shift + omega1 - omega2) - 1.0 / (omega_shift - omega1 + omega2);

                for (i = 0; i < N; ++i) {
                    double T_tmp = T[i];
                    if (thermodynamics->classical) {
                        n1 = thermodynamics->fC(omega1, T_tmp);
                        n2 = thermodynamics->fC(omega2, T_tmp);
                        f1 = n1 + n2;
                        f2 = n2 - n1;
                    } else {
                        n1 = thermodynamics->fB(omega1, T_tmp);
                        n2 = thermodynamics->fB(omega2, T_tmp);
                        f1 = n1 + n2 + 1.0;
                        f2 = n2 - n1;
                    }
                    ret_mpi[i] += v3_tmp * (f1 * omega_sum[0] + f2 * omega_sum[1]);
                }
            }
        }
    }

    double factor = 1.0 / (static_cast<double>(nk) * std::pow(2.0, 4));
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;
    
    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}

void Selfenergy::selfenergy_a_amano(const unsigned int N, //温度Tの数．まずこれが1じゃなかったらerrorを返すようにする．
                              const double *T, //温度Tの配列
                              const unsigned int knum,
                              const unsigned int snum, //モード番号（0始まり）
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in, //固有値
                              const std::complex<double> *const *const *evec_in, //フォノン固有ベクトル
                              const unsigned int nomega, // num of omega 
                              const double *omega,       // omega list
                              std::complex<double> *ret) const
{
    /*
    Diagram (a):Bubble
    Matrix elements that appear : V3^2
    Computational cost          : O(N_k * N^2)
    --------------------
    input

     const unsigned int N       : 温度の数(len(T))
     const double *T,           : 温度の配列
     const double omega,        : 振動数
     const unsigned int knum,   : 
     const unsigned int snum,
     const KpointMeshUniform *kmesh_in, : inputから計算されるkmesh
     const double *const *eval_in,      : harmonicの振動数リスト
     const std::complex<double> *const *const *evec_in, : harmonicの固有ベクトル，使ってない
     std::complex<double> *ret) const   : これを返す
     
    */
    unsigned int i; // for iteration
    unsigned int arr_cubic[3];
    double xk_tmp[3];
    std::complex<double> omega_sum[2];
    // ---------------    
    std::complex<double> *ret_mpi; // answer self-energy (original)
    // 温度Nに関するMPI(openMPではない！)
    allocate(ret_mpi, N);
    
    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0); //initialize self-energy to 0
    // ---------------    

    // ここから自分のコード //
    std::complex<double> *myomega;   //オメガ配列用 (add im*epsilon)
    std::complex<double> *ret_mpi3; //結果を格納する用2 (alamodeの振動数用)
    allocate(myomega,  nomega);
    allocate(ret_mpi3, nomega);

    for (i = 0; i < nomega; ++i) ret_mpi3[i] = std::complex<double>(0.0, 0.0); //self energy initialize
    for (i = 0; i < nomega; ++i) myomega[i] = omega[i]+ im * epsilon; // add infinitesimal(epsilon) imaginary part

#ifdef _DEBUG //デバック用に計算するomegaメッシュの振動数を出力する．
    if (mympi->my_rank == 0) {
            for (i = 0; i < nomega; ++i) {
                std::cout << "alamodeで定義された振動数は" << std::setprecision(16) << omega[i] << std::endl;
            }
        std::cout << "計算する温度(K) は " << std::setprecision(16) << T[0] << std::endl;
    }
#endif

    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;
    double n1, n2; // for BE function 
    double f1, f2; // for sum/difference of n

    arr_cubic[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum; //phononのq点(nsは多分mpiに関連している．)

    // std::complex<double> omega_shift = omega + im * epsilon; //虚数を加える.
    //    if (mympi->my_rank == 0) {
    //    std::cout << " (bubble) alamodeの振動数は "  << std::setprecision(16) << omega << std::endl;
    //    }

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) { //
        xk_tmp[0] = xk[knum][0] - xk[ik1][0];
        xk_tmp[1] = xk[knum][1] - xk[ik1][1];
        xk_tmp[2] = xk[knum][2] - xk[ik1][2];

        const auto ik2 = kmesh_in->get_knum(xk_tmp);

        for (unsigned int is1 = 0; is1 < ns; ++is1) {
            // k点1つめ
            arr_cubic[1] = ns * ik1 + is1;
            double omega1 = eval_in[ik1][is1];

            for (unsigned int is2 = 0; is2 < ns; ++is2) {
                // k点2つめ
                arr_cubic[2] = ns * ik2 + is2;
                double omega2 = eval_in[ik2][is2];
                // V(q,q1,q2)を取得
                double v3_tmp = std::norm(anharmonic_core->V3(arr_cubic));

                for (unsigned int iomega = 0; iomega < nomega; ++iomega){
                    //振動数オメガに関する繰り返し
                    //分母にくる振動数の差を計算
                    omega_sum[0] = 1.0 / (myomega[iomega] + omega1 + omega2) - 1.0 / (myomega[iomega] - omega1 - omega2);
                    omega_sum[1] = 1.0 / (myomega[iomega] + omega1 - omega2) - 1.0 / (myomega[iomega] - omega1 + omega2);

                    double T_tmp = T[0]; //とりあえず温度は1つだけ
                    if (thermodynamics->classical) {
                        n1 = thermodynamics->fC(omega1, T_tmp);
                        n2 = thermodynamics->fC(omega2, T_tmp);
                        f1 = n1 + n2;
                        f2 = n2 - n1;
                    } else {
                        n1 = thermodynamics->fB(omega1, T_tmp);
                        n2 = thermodynamics->fB(omega2, T_tmp);
                        f1 = n1 + n2 + 1.0;
                        f2 = n2 - n1;
                    }
                    ret_mpi3[iomega] += v3_tmp * (f1 * omega_sum[0] + f2 * omega_sum[1]);
                }
            }
        }
    }
    //最後に係数をかける
    double factor = 1.0 / (static_cast<double>(nk) * std::pow(2.0, 4));
    for (unsigned int iomega = 0; iomega < nomega; ++iomega ) ret_mpi3[iomega] *= factor;

    mpi_reduce_complex(nomega, ret_mpi3, ret); // ret_mpi3 to ret(answer)

    deallocate(ret_mpi);
    deallocate(ret_mpi3);
}

// void Selfenergy::selfenergy_a_amano(const unsigned int ntemp,
//                                            const double *temp_in,
//                                            const double omega_in,
//                                            const unsigned int ik_in,
//                                            const unsigned int is_in,
//                                            const KpointMeshUniform *kmesh_in,
//                                            const double *const *eval_in,
//                                            const std::complex<double> *const *const *evec_in,
//                                            double *ret)
// {
//   // bubble自己エネルギーの計算（Lorentian smearing）
//     // This function returns the imaginary part of phonon self-energy 
//     // for the given frequency omega_in.
//     // Lorentzian or Gaussian smearing will be used.
//     // This version employs the crystal symmetry to reduce the computational cost

//     const auto nk = kmesh_in->nk;
//     const auto ns = dynamical->neval;
//     const auto ns2 = ns * ns;
//     unsigned int i;
//     int ik;
//     unsigned int is, js;
//     unsigned int arr[3];

//     int k1, k2;

//     double T_tmp;
//     double n1, n2;
//     double omega_inner[2];

//     double multi;
//     // 出力するself_energy
//     for (i = 0; i < ntemp; ++i) ret[i] = 0.0;

//     double **v3_arr;
//     double ***delta_arr;
//     double ret_tmp;

//     double f1, f2;

//     const auto epsilon = integration->epsilon;

//     std::vector<KsListGroup> triplet;

//     //3つの波数kが-k+k2+k3=Gを満たすペアの生成．（デフォルトはsign=-1でこっちの符号）
//     kmesh_in->get_unique_triplet_k(ik_in,
//                                    symmetry->SymmList,
//                                    false,
//                                    false,
//                                    triplet);

//     const auto npair_uniq = triplet.size();

//     allocate(v3_arr, npair_uniq, ns * ns);
//     allocate(delta_arr, npair_uniq, ns * ns, 2);

//     const auto knum = kmesh_in->kpoint_irred_all[ik_in][0].knum;
//     const auto knum_minus = kmesh_in->kindex_minus_xk[knum];
// #ifdef _OPENMP
// #pragma omp parallel for private(multi, arr, k1, k2, is, js, omega_inner)
// #endif
//     for (ik = 0; ik < npair_uniq; ++ik) {
//         multi = static_cast<double>(triplet[ik].group.size());

//         arr[0] = ns * knum_minus + is_in;

//         k1 = triplet[ik].group[0].ks[0];
//         k2 = triplet[ik].group[0].ks[1];

//         for (is = 0; is < ns; ++is) {
//             arr[1] = ns * k1 + is;
//             omega_inner[0] = eval_in[k1][is];

//             for (js = 0; js < ns; ++js) {
//                 arr[2] = ns * k2 + js;
//                 omega_inner[1] = eval_in[k2][js];

//                 if (integration->ismear == 0) {
//                     delta_arr[ik][ns * is + js][0]
//                             = delta_lorentz(omega_in - omega_inner[0] - omega_inner[1], epsilon)
//                               - delta_lorentz(omega_in + omega_inner[0] + omega_inner[1], epsilon);
//                     delta_arr[ik][ns * is + js][1]
//                             = delta_lorentz(omega_in - omega_inner[0] + omega_inner[1], epsilon)
//                               - delta_lorentz(omega_in + omega_inner[0] - omega_inner[1], epsilon);
//                 } else if (integration->ismear == 1) {
//                     delta_arr[ik][ns * is + js][0]
//                             = delta_gauss(omega_in - omega_inner[0] - omega_inner[1], epsilon)
//                               - delta_gauss(omega_in + omega_inner[0] + omega_inner[1], epsilon);
//                     delta_arr[ik][ns * is + js][1]
//                             = delta_gauss(omega_in - omega_inner[0] + omega_inner[1], epsilon)
//                               - delta_gauss(omega_in + omega_inner[0] - omega_inner[1], epsilon);
//                 }
//             }
//         }
//     }

//     for (ik = 0; ik < npair_uniq; ++ik) {

//         k1 = triplet[ik].group[0].ks[0];
//         k2 = triplet[ik].group[0].ks[1];

//         multi = static_cast<double>(triplet[ik].group.size());

//         for (int ib = 0; ib < ns2; ++ib) {
//             is = ib / ns;
//             js = ib % ns;

//             arr[0] = ns * knum_minus + is_in;
//             arr[1] = ns * k1 + is;
//             arr[2] = ns * k2 + js;

//             v3_arr[ik][ib] = std::norm(anharmonic_core->V3(arr,
//                                           kmesh_in->xk,
//                                           eval_in,
//                                           evec_in,
//                                           anharmonic_core->phase_storage_dos)) * multi;
//         }
//     }
//     // 温度に関するループ
//     for (i = 0; i < ntemp; ++i) {
//         T_tmp = temp_in[i];
//         ret_tmp = 0.0;
// #ifdef _OPENMP
// #pragma omp parallel for private(k1, k2, is, js, omega_inner, n1, n2, f1, f2), reduction(+:ret_tmp)
// #endif
//         for (ik = 0; ik < npair_uniq; ++ik) {

//             k1 = triplet[ik].group[0].ks[0];
//             k2 = triplet[ik].group[0].ks[1];

//             for (is = 0; is < ns; ++is) {

//                 omega_inner[0] = eval_in[k1][is];

//                 for (js = 0; js < ns; ++js) {

//                     omega_inner[1] = eval_in[k2][js];

//                     if (thermodynamics->classical) {
//                         f1 = thermodynamics->fC(omega_inner[0], T_tmp);
//                         f2 = thermodynamics->fC(omega_inner[1], T_tmp);

//                         n1 = f1 + f2;
//                         n2 = f1 - f2;
//                     } else {
//                         f1 = thermodynamics->fB(omega_inner[0], T_tmp);
//                         f2 = thermodynamics->fB(omega_inner[1], T_tmp);

//                         n1 = f1 + f2 + 1.0;
//                         n2 = f1 - f2;
//                     }

//                     ret_tmp += v3_arr[ik][ns * is + js]
//                                * (n1 * delta_arr[ik][ns * is + js][0]
//                                   - n2 * delta_arr[ik][ns * is + js][1]);
//                 }
//             }
//         }
//         ret[i] = ret_tmp;
//     }

//     deallocate(v3_arr);
//     deallocate(delta_arr);
//     triplet.clear();

//     for (i = 0; i < ntemp; ++i) ret[i] *= pi * std::pow(0.5, 4) / static_cast<double>(nk);
// }





//
// Loop
//
void Selfenergy::selfenergy_b(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*
    Diagram (b) : Loop
    Matrix elements that appear : V4
    Computational cost          : O(N_k * N)
    Note                        : This give rise to the phonon frequency-shift only.(Loop)
    const unsigned int N       : 温度の数(len(T))
    const double *T,           : 温度の配列
    const double omega,        : 振動数
    const unsigned int knum,   : k点の指定のため？
    const unsigned int snum,   : k点の指定のため？
    const KpointMeshUniform *kmesh_in, : inputから計算されるkmesh
    const double *const *eval_in,      : harmonicの振動数リスト
    const std::complex<double> *const *const *evec_in, : harmonicの固有ベクトル，使ってない
    std::complex<double> *ret) const   : これを返す
     
    */

    unsigned int i;
    unsigned int arr_quartic[4];

    double n1;
    const auto nk = kmesh_in->nk;

    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);

    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0);

    arr_quartic[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_quartic[3] = ns * knum + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {
        for (unsigned int is1 = 0; is1 < ns; ++is1) {

            arr_quartic[1] = ns * ik1 + is1;
            arr_quartic[2] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;

            double omega1 = eval_in[ik1][is1];
            if (omega1 < eps8) continue;

            std::complex<double> v4_tmp = anharmonic_core->V4(arr_quartic);

            if (thermodynamics->classical) {
                for (i = 0; i < N; ++i) {
                    n1 = thermodynamics->fC(omega1, T[i]);
                    ret_mpi[i] += v4_tmp * 2.0 * n1;
                }
            } else {
                for (i = 0; i < N; ++i) {
                    n1 = thermodynamics->fB(omega1, T[i]);
                    ret_mpi[i] += v4_tmp * (2.0 * n1 + 1.0);
                }
            }

        }
    }

    double factor = -1.0 / (static_cast<double>(nk) * std::pow(2.0, 3));
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}


void Selfenergy::selfenergy_c(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*

    Diagram (c) : 4ph
    Matrix elements that appear : V4^2
    Computational cost          : O(N_k^2 * N^3) <-- about N_k * N times that of Diagram (a)
    Note  : 

    */

    unsigned int i;
    unsigned int arr_quartic[4];

    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;
    double xk_tmp[3];

    std::complex<double> omega_sum[4];
    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);
    if (mympi->my_rank == 0) {
        std::cout << "(4ph) Frequency " << std::setprecision(16) << omega << std::endl;
    }
    // 微小な複素数を加えている
    std::complex<double> omega_shift = omega + im * epsilon;

    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0); // MPI用? どうも温度についてのMPI化をやっている？
    
    // snumはモード数．
    arr_quartic[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {
        for (unsigned int ik2 = 0; ik2 < nk; ++ik2) {

            xk_tmp[0] = xk[knum][0] - xk[ik1][0] - xk[ik2][0];
            xk_tmp[1] = xk[knum][1] - xk[ik1][1] - xk[ik2][1];
            xk_tmp[2] = xk[knum][2] - xk[ik1][2] - xk[ik2][2];

            const auto ik3 = kmesh_in->get_knum(xk_tmp);

            for (unsigned int is1 = 0; is1 < ns; ++is1) {
	        // k点一つ目 
                arr_quartic[1] = ns * ik1 + is1;
                double omega1 = eval_in[ik1][is1];

                for (unsigned int is2 = 0; is2 < ns; ++is2) {
	        // k点2つ目 
                    arr_quartic[2] = ns * ik2 + is2;
                    double omega2 = eval_in[ik2][is2];

                    for (unsigned int is3 = 0; is3 < ns; ++is3) {
		        // k点3つ目 		      
                        arr_quartic[3] = ns * ik3 + is3;
                        double omega3 = eval_in[ik3][is3];
			//Vを計算
                        double v4_tmp = std::norm(anharmonic_core->V4(arr_quartic));
			//分母にくる振動数の差を計算
                        omega_sum[0]
                                = 1.0 / (omega_shift - omega1 - omega2 - omega3)
                                  - 1.0 / (omega_shift + omega1 + omega2 + omega3);
                        omega_sum[1]
                                = 1.0 / (omega_shift - omega1 - omega2 + omega3)
                                  - 1.0 / (omega_shift + omega1 + omega2 - omega3);
                        omega_sum[2]
                                = 1.0 / (omega_shift + omega1 - omega2 - omega3)
                                  - 1.0 / (omega_shift - omega1 + omega2 + omega3);
                        omega_sum[3]
                                = 1.0 / (omega_shift - omega1 + omega2 - omega3)
                                  - 1.0 / (omega_shift + omega1 - omega2 + omega3);

                        for (i = 0; i < N; ++i) {
                            double T_tmp = T[i];
			    // BE分布関数
                            double n1 = thermodynamics->fB(omega1, T_tmp);
                            double n2 = thermodynamics->fB(omega2, T_tmp);
                            double n3 = thermodynamics->fB(omega3, T_tmp);

                            double n12 = n1 * n2;
                            double n23 = n2 * n3;
                            double n31 = n3 * n1;
			    
                            ret_mpi[i] += v4_tmp
                                          * ((n12 + n23 + n31 + n1 + n2 + n3 + 1.0) * omega_sum[0]
                                             + (n31 + n23 + n3 - n12) * omega_sum[1]
                                             + (n12 + n31 + n1 - n23) * omega_sum[2]
                                             + (n23 + n12 + n2 - n31) * omega_sum[3]);
                        }
                    }
                }
            }
        }
    }
    //ここのファクターは要注意
    double factor = -1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 5) * 3.0);
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}

//
// 温度依存じゃなくて振動数依存にしたい．
//
void Selfenergy::selfenergy_c_amano(const unsigned int N, //温度Tの数．まずこれが1じゃなかったらerrorを返すようにする．
                              const double *T, //温度Tの配列
                              const unsigned int knum,
                              const unsigned int snum, //モード番号（0始まり）
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in, //固有値
                              const std::complex<double> *const *const *evec_in, //フォノン固有ベクトル
                              const unsigned int nomega,
                              const double *omega,
                              std::complex<double> *ret) const
{
    /*

    Diagram (c) : 4ph
    Matrix elements that appear : V4^2
    Computational cost          : O(N_k^2 * N^3) <-- about N_k * N times that of Diagram (a)
    Note  : こちらでは温度ではないMPI並列を目指す．

    */
    unsigned int i;
    unsigned int arr_quartic[4];

    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;
    double xk_tmp[3];
    std::complex<double> omega_sum[4];
    std::complex<double> *ret_mpi; // answer self-energy 
    allocate(ret_mpi, N);
    // フォノン振動数omegaに微小な複素数を加えている
    // std::complex<double> omega_shift = omega + im * epsilon;

#ifdef _DEBUG //デバック用に計算するomegaメッシュの振動数を出力する．
    if (mympi->my_rank == 0) {
            for (i = 0; i < nomega; ++i) {
                std::cout << "alamodeで定義された振動数は" << std::setprecision(16) << omega[i] << std::endl;
            }
        std::cout << "計算する温度(K) は " << std::setprecision(16) << T[0] << std::endl;
    }
#endif

    if (N != 1 ){
        if (mympi->my_rank == 0) {
        std::cout << "ERROR:: Nが1ではありません" << std::endl;
        }
        exit(1);
    }

    // ここから自分のコード //
    std::complex<double> *myomega;   //オメガ配列用 (add im*epsilon)
    std::complex<double> *ret_mpi3; //結果を格納する用2 (alamodeの振動数用)
    allocate(myomega,  nomega);
    allocate(ret_mpi3, nomega);

    for (i = 0; i < nomega; ++i) ret_mpi3[i] = std::complex<double>(0.0, 0.0); //初期化
    for (i = 0; i < nomega; ++i) myomega[i] = omega[i]+ im * epsilon; // 微小なepsilonをたしたもの．


    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0); // MPI用? どうも温度についてのMPI化をやっている？
    
    // arr_quarticの1つ目ということで，V(q1,q2,q3,q4)のうちの始めのωに対応するq点を出している？
    // 振動数依存になってもここは変更しなくて良いかも？（まずは変更せずに試す）
    arr_quartic[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {
        for (unsigned int ik2 = 0; ik2 < nk; ++ik2) {

            xk_tmp[0] = xk[knum][0] - xk[ik1][0] - xk[ik2][0];
            xk_tmp[1] = xk[knum][1] - xk[ik1][1] - xk[ik2][1];
            xk_tmp[2] = xk[knum][2] - xk[ik1][2] - xk[ik2][2];

            const auto ik3 = kmesh_in->get_knum(xk_tmp);

            for (unsigned int is1 = 0; is1 < ns; ++is1) {
    	        // k点一つ目 
                arr_quartic[1] = ns * ik1 + is1;
                double omega1 = eval_in[ik1][is1];

                for (unsigned int is2 = 0; is2 < ns; ++is2) {
        	        // k点2つ目 
                    arr_quartic[2] = ns * ik2 + is2;
                    double omega2 = eval_in[ik2][is2];

                    for (unsigned int is3 = 0; is3 < ns; ++is3) {
                        // k点3つ目 		      
                        arr_quartic[3] = ns * ik3 + is3;
                        double omega3 = eval_in[ik3][is3];
                        //V(q1,q2,q3,q4)を取得
                        double v4_tmp = std::norm(anharmonic_core->V4(arr_quartic));
                        for (unsigned int iomega = 0; iomega < nomega; ++iomega){
                            //分母にくる振動数の差を計算
                            omega_sum[0]
                                    = 1.0 / (myomega[iomega]  - omega1 - omega2 - omega3)
                                      - 1.0 / (myomega[iomega] + omega1 + omega2 + omega3);
                            omega_sum[1]
                                    = 1.0 / (myomega[iomega] - omega1 - omega2 + omega3)
                                      - 1.0 / (myomega[iomega] + omega1 + omega2 - omega3);
                            omega_sum[2]
                                    = 1.0 / (myomega[iomega] + omega1 - omega2 - omega3)
                                      - 1.0 / (myomega[iomega] - omega1 + omega2 + omega3);
                            omega_sum[3]
                                    = 1.0 / (myomega[iomega] - omega1 + omega2 - omega3)
                                      - 1.0 / (myomega[iomega] + omega1 - omega2 + omega3);
                            double T_tmp = T[0]; // 温度をT[0]のみに固定
                		    // BE分布関数
                            double n1 = thermodynamics->fB(omega1, T_tmp);
                            double n2 = thermodynamics->fB(omega2, T_tmp);
                            double n3 = thermodynamics->fB(omega3, T_tmp);

                            double n12 = n1 * n2;
                            double n23 = n2 * n3;
                            double n31 = n3 * n1;
    
                            ret_mpi3[iomega] += v4_tmp
                                        * ((n12 + n23 + n31 + n1 + n2 + n3 + 1.0) * omega_sum[0]
                                        + (n31 + n23 + n3 - n12) * omega_sum[1]
                                        + (n12 + n31 + n1 - n23) * omega_sum[2]
                                        + (n23 + n12 + n2 - n31) * omega_sum[3]);
                        }
                    }          
                }
            }
        }
    }
    //ここのファクターは要注意
    double factor = -1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 5) * 3.0);
    for (unsigned int iomega = 0; iomega < nomega; ++iomega ) ret_mpi3[iomega] *= factor;

    mpi_reduce_complex(nomega, ret_mpi3, ret);


    deallocate(ret_mpi);
    deallocate(ret_mpi3);


}



void Selfenergy::selfenergy_d(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*

    Diagram (d)
    Matrix elements that appear : V3^2 V4
    Computational cost          : O(N_k^2 * N^4)
    Note                        : 2 3-point vertexes and 1 4-point vertex.

    */

    unsigned int i;
    unsigned int arr_cubic1[3], arr_cubic2[3];
    unsigned int arr_quartic[4];
    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;

    double xk_tmp[3];

    std::complex<double> omega_sum[4];
    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);

    std::complex<double> omega_shift = omega + im * epsilon;

    for (i = 0; i < N; ++i) ret[i] = std::complex<double>(0.0, 0.0);

    arr_cubic1[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_cubic2[2] = ns * knum + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {

        xk_tmp[0] = xk[knum][0] - xk[ik1][0];
        xk_tmp[1] = xk[knum][1] - xk[ik1][1];
        xk_tmp[2] = xk[knum][2] - xk[ik1][2];

        const auto ik2 = kmesh_in->get_knum(xk_tmp);

        for (unsigned int ik3 = 0; ik3 < nk; ++ik3) {

            xk_tmp[0] = xk[knum][0] - xk[ik3][0];
            xk_tmp[1] = xk[knum][1] - xk[ik3][1];
            xk_tmp[2] = xk[knum][2] - xk[ik3][2];

            const auto ik4 = kmesh_in->get_knum(xk_tmp);

            for (unsigned int is1 = 0; is1 < ns; ++is1) {

                double omega1 = eval_in[ik1][is1];

                arr_cubic2[0] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;
                arr_quartic[0] = ns * ik1 + is1;

                for (unsigned int is2 = 0; is2 < ns; ++is2) {

                    double omega2 = eval_in[ik2][is2];

                    arr_cubic2[1] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;
                    arr_quartic[1] = ns * ik2 + is2;

                    std::complex<double> v3_tmp2 = anharmonic_core->V3(arr_cubic2);

                    for (unsigned int is3 = 0; is3 < ns; ++is3) {

                        double omega3 = eval_in[ik3][is3];

                        arr_cubic1[1] = ns * ik3 + is3;
                        arr_quartic[2] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;

                        for (unsigned int is4 = 0; is4 < ns; ++is4) {

                            double omega4 = eval_in[ik4][is4];

                            arr_cubic1[2] = ns * ik4 + is4;
                            arr_quartic[3] = ns * kmesh_in->kindex_minus_xk[ik4] + is4;

                            std::complex<double> v3_tmp1 = anharmonic_core->V3(arr_cubic1);
                            std::complex<double> v4_tmp = anharmonic_core->V4(arr_quartic);

                            std::complex<double> v_prod = v3_tmp1 * v3_tmp2 * v4_tmp;

                            omega_sum[0]
                                    = 1.0 / (omega_shift + omega1 + omega2)
                                      - 1.0 / (omega_shift - omega1 - omega2);
                            omega_sum[1]
                                    = 1.0 / (omega_shift + omega1 - omega2)
                                      - 1.0 / (omega_shift - omega1 + omega2);
                            omega_sum[2]
                                    = 1.0 / (omega_shift + omega3 + omega4)
                                      - 1.0 / (omega_shift - omega3 - omega4);
                            omega_sum[3]
                                    = 1.0 / (omega_shift + omega3 - omega4)
                                      - 1.0 / (omega_shift - omega3 + omega4);

                            for (i = 0; i < N; ++i) {
                                double T_tmp = T[i];

                                double n1 = thermodynamics->fB(omega1, T_tmp);
                                double n2 = thermodynamics->fB(omega2, T_tmp);
                                double n3 = thermodynamics->fB(omega3, T_tmp);
                                double n4 = thermodynamics->fB(omega4, T_tmp);

                                ret_mpi[i] += v_prod
                                              * ((1.0 + n1 + n2) * omega_sum[0] + (n2 - n1) * omega_sum[1])
                                              * ((1.0 + n3 + n4) * omega_sum[2] + (n4 - n3) * omega_sum[3]);
                            }
                        }
                    }
                }
            }
        }
    }

    double factor = -1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 7));
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}

void Selfenergy::selfenergy_e(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*

    Diagram (e)
    Matrix elements that appear : V3^2 V4
    Computational cost          : O(N_k^2 * N^4)
    Note                        : Double pole appears when omega1 = omega2.

    */

    unsigned int i;
    unsigned int is3, is4;
    unsigned int arr_cubic1[3], arr_cubic2[3];
    unsigned int arr_quartic[4];
    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;

    double T_tmp;
    double omega3, omega4;
    double n1, n3, n4;
    double xk_tmp[3];
    double D12[2];
    double T_inv;

    std::complex<double> v3_tmp1, v3_tmp2, v4_tmp;
    std::complex<double> v_prod;
    std::complex<double> omega_sum14[4], omega_sum24[4];
    std::complex<double> omega_prod[6];
    std::complex<double> *prod_tmp;
    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);
    allocate(prod_tmp, N);

    std::complex<double> omega_shift = omega + im * epsilon;

    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0);

    arr_cubic1[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_cubic2[2] = ns * knum + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {

        const auto ik2 = ik1;

        xk_tmp[0] = xk[knum][0] - xk[ik1][0];
        xk_tmp[1] = xk[knum][1] - xk[ik1][1];
        xk_tmp[2] = xk[knum][2] - xk[ik1][2];

        const auto ik4 = kmesh_in->get_knum(xk_tmp);

        for (unsigned int ik3 = 0; ik3 < nk; ++ik3) {

            for (unsigned int is1 = 0; is1 < ns; ++is1) {

                double omega1 = eval_in[ik1][is1];

                arr_cubic1[1] = ns * ik1 + is1;
                arr_quartic[0] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;

                for (unsigned int is2 = 0; is2 < ns; ++is2) {

                    double omega2 = eval_in[ik2][is2];

                    arr_cubic2[0] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;
                    arr_quartic[3] = ns * ik2 + is2;

                    if (std::abs(omega1 - omega2) < eps) {

                        for (is3 = 0; is3 < ns; ++is3) {

                            omega3 = eval_in[ik3][is3];

                            arr_quartic[1] = ns * ik3 + is3;
                            arr_quartic[2] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;

                            v4_tmp = anharmonic_core->V4(arr_quartic);

                            for (is4 = 0; is4 < ns; ++is4) {

                                omega4 = eval_in[ik4][is4];

                                arr_cubic1[2] = ns * ik4 + is4;
                                arr_cubic2[1] = ns * kmesh_in->kindex_minus_xk[ik4] + is4;

                                v3_tmp1 = anharmonic_core->V3(arr_cubic1);
                                v3_tmp2 = anharmonic_core->V3(arr_cubic2);

                                v_prod = v3_tmp1 * v3_tmp2 * v4_tmp;

                                for (i = 0; i < N; ++i) prod_tmp[i] = std::complex<double>(0.0, 0.0);

                                for (int ip1 = 1; ip1 >= -1; ip1 -= 2) {
                                    double dp1 = static_cast<double>(ip1) * omega1;
                                    double dp1_inv = 1.0 / dp1;

                                    for (int ip4 = 1; ip4 >= -1; ip4 -= 2) {
                                        double dp4 = static_cast<double>(ip4) * omega4;

                                        std::complex<double> omega_sum = 1.0 / (omega_shift + dp1 + dp4);

                                        for (i = 0; i < N; ++i) {
                                            T_tmp = T[i];

                                            n1 = thermodynamics->fB(dp1, T_tmp);
                                            n4 = thermodynamics->fB(dp4, T_tmp);

                                            if (std::abs(T_tmp) < eps) {
                                                //special treatment for T = 0
                                                // This is valid since beta always appears as a product beta*n
                                                // which is zero when T = 0.
                                                T_inv = 0.0;
                                            } else {
                                                T_inv = 1.0 / (thermodynamics->T_to_Ryd * T_tmp);
                                            }

                                            prod_tmp[i] += static_cast<double>(ip4) * omega_sum
                                                           * ((1.0 + n1 + n4) * omega_sum
                                                              + (1.0 + n1 + n4) * dp1_inv + n1 * (1.0 + n1) * T_inv);
                                        }
                                    }
                                }

                                for (i = 0; i < N; ++i) {
                                    T_tmp = T[i];

                                    n3 = thermodynamics->fB(omega3, T_tmp);
                                    ret_mpi[i] += v_prod * (2.0 * n3 + 1.0) * prod_tmp[i];
                                }
                            }
                        }

                    } else {

                        D12[0] = 1.0 / (omega1 + omega2) - 1.0 / (omega1 - omega2);
                        D12[1] = 1.0 / (omega1 + omega2) + 1.0 / (omega1 - omega2);

                        for (is3 = 0; is3 < ns; ++is3) {

                            omega3 = eval_in[ik3][is3];

                            arr_quartic[1] = ns * ik3 + is3;
                            arr_quartic[2] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;

                            v4_tmp = anharmonic_core->V4(arr_quartic);

                            for (is4 = 0; is4 < ns; ++is4) {

                                omega4 = eval_in[ik4][is4];

                                arr_cubic1[2] = ns * ik4 + is4;
                                arr_cubic2[1] = ns * kmesh_in->kindex_minus_xk[ik4] + is4;

                                v3_tmp1 = anharmonic_core->V3(arr_cubic1);
                                v3_tmp2 = anharmonic_core->V3(arr_cubic2);

                                v_prod = v3_tmp1 * v3_tmp2 * v4_tmp;

                                omega_sum14[0] = 1.0 / (omega_shift + omega1 + omega4);
                                omega_sum14[1] = 1.0 / (omega_shift + omega1 - omega4);
                                omega_sum14[2] = 1.0 / (omega_shift - omega1 + omega4);
                                omega_sum14[3] = 1.0 / (omega_shift - omega1 - omega4);

                                omega_sum24[0] = 1.0 / (omega_shift + omega2 + omega4);
                                omega_sum24[1] = 1.0 / (omega_shift + omega2 - omega4);
                                omega_sum24[2] = 1.0 / (omega_shift - omega2 + omega4);
                                omega_sum24[3] = 1.0 / (omega_shift - omega2 - omega4);

                                omega_prod[0] = D12[0] * (omega_sum14[0] - omega_sum14[1]);
                                omega_prod[1] = D12[0] * (omega_sum14[2] - omega_sum14[3]);
                                omega_prod[2] = D12[1] * (omega_sum24[0] - omega_sum24[1]);
                                omega_prod[3] = D12[1] * (omega_sum24[2] - omega_sum24[3]);
                                omega_prod[4] = (omega_sum14[1] - omega_sum14[3])
                                                * (omega_sum24[1] - omega_sum24[3]);
                                omega_prod[5] = (omega_sum14[0] - omega_sum14[2])
                                                * (omega_sum24[0] - omega_sum24[2]);

                                for (i = 0; i < N; ++i) {
                                    T_tmp = T[i];

                                    n1 = thermodynamics->fB(omega1, T_tmp);
                                    double n2 = thermodynamics->fB(omega2, T_tmp);
                                    n3 = thermodynamics->fB(omega3, T_tmp);
                                    n4 = thermodynamics->fB(omega4, T_tmp);

                                    ret_mpi[i] += v_prod * (2.0 * n3 + 1.0)
                                                  * ((1.0 + n1) * omega_prod[0] + n1 * omega_prod[1]
                                                     + (1.0 + n2) * omega_prod[2] + n2 * omega_prod[3]
                                                     + (1.0 + n4) * omega_prod[4] + n4 * omega_prod[5]);

                                    /*
                                    ret[i] *= v3_tmp1 * v3_tmp2 * v4_tmp * (2.0 * n3 + 1.0) * (2.0 * omega2) / (omega1 * omega1 - omega2 * omega2)
                                    * ((1.0 + n1 + n4) * (1.0 / (omega - omega1 - omega4 + im * epsilon) - 1.0 / (omega + omega1 + omega4 + im * epsilon))
                                    + (n4 - n1) * (1.0 / (omega - omega1 + omega4 + im * epsilon) - 1.0 / (omega + omega1 - omega4 + im * epsilon)));
                                    */
                                }
                            }
                        }

                    }
                }
            }
        }
    }

    double factor = -1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 6));
    //	factor = -1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 7));
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(prod_tmp);
    deallocate(ret_mpi);
}

void Selfenergy::selfenergy_f(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*
    Diagram (f)
    Matrix elements that appear : V3^4
    Computational cost          : O(N_k^2 * N^5)
    Note                        : Computationally expensive & double pole when omega1 = omega5.
    */

    unsigned int i;
    unsigned int arr_cubic1[3], arr_cubic2[3], arr_cubic3[3], arr_cubic4[3];
    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;

    int ip1, ip2, ip3, ip4;

    double n1, n2, n3, n4;
    double xk_tmp[3];
    double dp1, dp2, dp3, dp4;
    double T_tmp;
    double D134;
    double T_inv;

    std::complex<double> omega_sum[3];
    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);

    std::complex<double> omega_shift = omega + im * epsilon;

    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0);

    arr_cubic1[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_cubic4[2] = ns * knum + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {

        unsigned int ik5 = ik1;

        xk_tmp[0] = xk[knum][0] - xk[ik1][0];
        xk_tmp[1] = xk[knum][1] - xk[ik1][1];
        xk_tmp[2] = xk[knum][2] - xk[ik1][2];

        const auto ik2 = kmesh_in->get_knum(xk_tmp);

        for (unsigned int ik3 = 0; ik3 < nk; ++ik3) {

            xk_tmp[0] = xk[ik1][0] - xk[ik3][0];
            xk_tmp[1] = xk[ik1][1] - xk[ik3][1];
            xk_tmp[2] = xk[ik1][2] - xk[ik3][2];

            const auto ik4 = kmesh_in->get_knum(xk_tmp);

            for (unsigned int is1 = 0; is1 < ns; ++is1) {

                double omega1 = eval_in[ik1][is1];

                arr_cubic1[1] = ns * ik1 + is1;
                arr_cubic2[0] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;

                for (unsigned int is2 = 0; is2 < ns; ++is2) {

                    double omega2 = eval_in[ik2][is2];

                    arr_cubic1[2] = ns * ik2 + is2;
                    arr_cubic4[1] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;

                    std::complex<double> v3_tmp1 = anharmonic_core->V3(arr_cubic1);

                    for (unsigned int is5 = 0; is5 < ns; ++is5) {

                        double omega5 = eval_in[ik5][is5];

                        arr_cubic3[2] = ns * ik5 + is5;
                        arr_cubic4[0] = ns * kmesh_in->kindex_minus_xk[ik5] + is5;

                        std::complex<double> v3_tmp4 = anharmonic_core->V3(arr_cubic4);

                        for (unsigned int is3 = 0; is3 < ns; ++is3) {

                            double omega3 = eval_in[ik3][is3];

                            arr_cubic2[1] = ns * ik3 + is3;
                            arr_cubic3[0] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;

                            for (unsigned int is4 = 0; is4 < ns; ++is4) {

                                double omega4 = eval_in[ik4][is4];

                                arr_cubic2[2] = ns * ik4 + is4;
                                arr_cubic3[1] = ns * kmesh_in->kindex_minus_xk[ik4] + is4;

                                std::complex<double> v3_tmp2 = anharmonic_core->V3(arr_cubic2);
                                std::complex<double> v3_tmp3 = anharmonic_core->V3(arr_cubic3);

                                std::complex<double> v3_prod = v3_tmp1 * v3_tmp2 * v3_tmp3 * v3_tmp4;

                                if (std::abs(omega1 - omega5) < eps) {

                                    for (ip1 = 1; ip1 >= -1; ip1 -= 2) {
                                        dp1 = static_cast<double>(ip1) * omega1;
                                        double dp1_inv = 1.0 / dp1;

                                        for (ip2 = 1; ip2 >= -1; ip2 -= 2) {
                                            dp2 = static_cast<double>(ip2) * omega2;
                                            omega_sum[0] = 1.0 / (omega_shift + dp1 + dp2);

                                            for (ip3 = 1; ip3 >= -1; ip3 -= 2) {
                                                dp3 = static_cast<double>(ip3) * omega3;

                                                for (ip4 = 1; ip4 >= -1; ip4 -= 2) {
                                                    dp4 = static_cast<double>(ip4) * omega4;

                                                    D134 = 1.0 / (dp1 + dp3 + dp4);
                                                    omega_sum[1] = 1.0 / (omega_shift + dp2 + dp3 + dp4);

                                                    for (i = 0; i < N; ++i) {
                                                        T_tmp = T[i];

                                                        n1 = thermodynamics->fB(dp1, T_tmp);
                                                        n2 = thermodynamics->fB(dp2, T_tmp);
                                                        n3 = thermodynamics->fB(dp3, T_tmp);
                                                        n4 = thermodynamics->fB(dp4, T_tmp);

                                                        if (std::abs(T_tmp) < eps) {
                                                            T_inv = 0.0;
                                                        } else {
                                                            T_inv = 1.0 / (thermodynamics->T_to_Ryd * T_tmp);
                                                        }

                                                        ret_mpi[i]
                                                                += v3_prod * static_cast<double>(ip2 * ip3 * ip4)
                                                                   * (omega_sum[1]
                                                                      * (n2 * omega_sum[0]
                                                                         * ((1.0 + n3 + n4) * omega_sum[0] +
                                                                            (1.0 + n2 + n4)
                                                                            * dp1_inv)
                                                                         + (1.0 + n3) * (1.0 + n4) * D134 *
                                                                           (D134 + dp1_inv))
                                                                      + (1.0 + n1) * (1.0 + n3 + n4) * D134
                                                                        * omega_sum[0] *
                                                                        (omega_sum[0] + D134 + dp1_inv + n1 *
                                                                                                         T_inv));
                                                    }
                                                }
                                            }
                                        }
                                    }

                                } else {

                                    for (ip1 = 1; ip1 >= -1; ip1 -= 2) {
                                        dp1 = static_cast<double>(ip1) * omega1;

                                        for (int ip5 = 1; ip5 >= -1; ip5 -= 2) {
                                            double dp5 = static_cast<double>(ip5) * omega5;

                                            double D15 = 1.0 / (dp1 - dp5);

                                            for (ip2 = 1; ip2 >= -1; ip2 -= 2) {
                                                dp2 = static_cast<double>(ip2) * omega2;

                                                omega_sum[0] = 1.0 / (omega_shift + dp1 + dp2);
                                                omega_sum[1] = 1.0 / (omega_shift + dp5 + dp2);

                                                for (ip3 = 1; ip3 >= -1; ip3 -= 2) {
                                                    dp3 = static_cast<double>(ip3) * omega3;

                                                    for (ip4 = 1; ip4 >= -1; ip4 -= 2) {
                                                        dp4 = static_cast<double>(ip4) * omega4;

                                                        D134 = 1.0 / (dp1 + dp3 + dp4);
                                                        double D345 = 1.0 / (dp5 + dp3 + dp4);
                                                        omega_sum[2] = 1.0 / (omega_shift + dp2 + dp3 + dp4);

                                                        for (i = 0; i < N; ++i) {
                                                            T_tmp = T[i];

                                                            n1 = thermodynamics->fB(dp1, T_tmp);
                                                            n2 = thermodynamics->fB(dp2, T_tmp);
                                                            n3 = thermodynamics->fB(dp3, T_tmp);
                                                            n4 = thermodynamics->fB(dp4, T_tmp);
                                                            double n5 = thermodynamics->fB(dp5, T_tmp);

                                                            ret_mpi[i]
                                                                    += v3_prod *
                                                                       static_cast<double>(ip1 * ip2 * ip3 * ip4 *
                                                                                           ip5)
                                                                       * ((1.0 + n3 + n4)
                                                                          *
                                                                          (-(1.0 + n1 + n2) * D15 * D134
                                                                           * omega_sum[0]
                                                                           +
                                                                           (1.0 + n5 + n2)
                                                                           * D15 * D345
                                                                           * omega_sum[1])
                                                                          + (1.0 + n2 + n3 + n4 + n2 * n3 + n3 * n4 +
                                                                             n4 * n2)
                                                                            * D15 * (D345 - D134)
                                                                            * omega_sum[2]);
                                                        }
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double factor = 1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 7));
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}

void Selfenergy::selfenergy_g(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*
    Diagram (g)
    Matrix elements that appear : V3^2 V4
    Computational cost          : O(N_k^2 * N^4)
    */

    unsigned int i;

    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;

    unsigned int arr_quartic[4], arr_cubic1[3], arr_cubic2[3];

    double xk_tmp[3];

    std::complex<double> omega_sum[2];

    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);

    std::complex<double> omega_shift = omega + im * epsilon;

    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0);

    arr_quartic[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_cubic2[2] = ns * knum + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {

        for (unsigned int ik2 = 0; ik2 < nk; ++ik2) {

            xk_tmp[0] = xk[knum][0] - xk[ik1][0] - xk[ik2][0];
            xk_tmp[1] = xk[knum][1] - xk[ik1][1] - xk[ik2][1];
            xk_tmp[2] = xk[knum][2] - xk[ik1][2] - xk[ik2][2];

            const auto ik3 = kmesh_in->get_knum(xk_tmp);

            xk_tmp[0] = xk[knum][0] - xk[ik3][0];
            xk_tmp[1] = xk[knum][1] - xk[ik3][1];
            xk_tmp[2] = xk[knum][2] - xk[ik3][2];

            const auto ik4 = kmesh_in->get_knum(xk_tmp);

            for (unsigned int is1 = 0; is1 < ns; ++is1) {
                double omega1 = eval_in[ik1][is1];

                arr_quartic[1] = ns * ik1 + is1;
                arr_cubic1[0] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;

                for (unsigned int is2 = 0; is2 < ns; ++is2) {
                    double omega2 = eval_in[ik2][is2];

                    arr_quartic[2] = ns * ik2 + is2;
                    arr_cubic1[1] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;

                    for (unsigned int is3 = 0; is3 < ns; ++is3) {
                        double omega3 = eval_in[ik3][is3];

                        arr_quartic[3] = ns * ik3 + is3;
                        arr_cubic2[0] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;

                        std::complex<double> v4_tmp = anharmonic_core->V4(arr_quartic);

                        for (unsigned int is4 = 0; is4 < ns; ++is4) {
                            double omega4 = eval_in[ik4][is4];

                            arr_cubic1[2] = ns * ik4 + is4;
                            arr_cubic2[1] = ns * kmesh_in->kindex_minus_xk[ik4] + is4;

                            std::complex<double> v3_tmp1 = anharmonic_core->V3(arr_cubic1);
                            std::complex<double> v3_tmp2 = anharmonic_core->V3(arr_cubic2);

                            std::complex<double> v_prod = v4_tmp * v3_tmp1 * v3_tmp2;

                            for (int ip1 = 1; ip1 >= -1; ip1 -= 2) {
                                double dp1 = static_cast<double>(ip1) * omega1;
                                for (int ip2 = 1; ip2 >= -1; ip2 -= 2) {
                                    double dp2 = static_cast<double>(ip2) * omega2;
                                    for (int ip3 = 1; ip3 >= -1; ip3 -= 2) {
                                        double dp3 = static_cast<double>(ip3) * omega3;

                                        omega_sum[1] = 1.0 / (omega_shift + dp1 + dp2 + dp3);

                                        for (int ip4 = 1; ip4 >= -1; ip4 -= 2) {
                                            double dp4 = static_cast<double>(ip4) * omega4;

                                            omega_sum[0] = 1.0 / (omega_shift + dp3 + dp4);
                                            double D124 = 1.0 / (dp1 + dp2 - dp4);

                                            for (i = 0; i < N; ++i) {
                                                double T_tmp = T[i];

                                                double n1 = thermodynamics->fB(dp1, T_tmp);
                                                double n2 = thermodynamics->fB(dp2, T_tmp);
                                                double n3 = thermodynamics->fB(dp3, T_tmp);
                                                double n4 = thermodynamics->fB(dp4, T_tmp);

                                                ret_mpi[i]
                                                        += v_prod * static_cast<double>(ip1 * ip2 * ip3 * ip4) * D124
                                                           * ((1.0 + n1 + n2 + n3 + n4 + n1 * n3 + n1 * n4 + n2 * n3 +
                                                               n2 * n4)
                                                              * omega_sum[0]
                                                              - (1.0 + n1 + n2 + n3 + n1 * n2 + n2 * n3 + n1 * n3) *
                                                                omega_sum
                                                                [1]);

                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double factor = -1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 6));
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}

void Selfenergy::selfenergy_h(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*
    Diagram (h)
    Matrix elements that appear : V3^4
    Computational cost          : O(N_k^2 * N^5)
    Note                        : The most complicated diagram.
    */

    unsigned int i;
    unsigned int arr_cubic1[3], arr_cubic2[3], arr_cubic3[3], arr_cubic4[3];
    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;

    double xk_tmp[3];
    double N_prod[4];

    std::complex<double> omega_sum[4];
    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);

    std::complex<double> omega_shift = omega + im * epsilon;

    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0);

    arr_cubic1[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_cubic4[2] = ns * knum + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {

        xk_tmp[0] = xk[knum][0] - xk[ik1][0];
        xk_tmp[1] = xk[knum][1] - xk[ik1][1];
        xk_tmp[2] = xk[knum][2] - xk[ik1][2];

        const auto ik2 = kmesh_in->get_knum(xk_tmp);

        for (unsigned int ik3 = 0; ik3 < nk; ++ik3) {

            xk_tmp[0] = xk[ik1][0] - xk[ik3][0];
            xk_tmp[1] = xk[ik1][1] - xk[ik3][1];
            xk_tmp[2] = xk[ik1][2] - xk[ik3][2];

            const auto ik5 = kmesh_in->get_knum(xk_tmp);

            xk_tmp[0] = xk[knum][0] - xk[ik5][0];
            xk_tmp[1] = xk[knum][1] - xk[ik5][1];
            xk_tmp[2] = xk[knum][2] - xk[ik5][2];

            const auto ik4 = kmesh_in->get_knum(xk_tmp);

            for (unsigned int is1 = 0; is1 < ns; ++is1) {
                double omega1 = eval_in[ik1][is1];

                arr_cubic1[1] = ns * ik1 + is1;
                arr_cubic2[0] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;

                for (unsigned int is2 = 0; is2 < ns; ++is2) {
                    double omega2 = eval_in[ik2][is2];

                    arr_cubic1[2] = ns * ik2 + is2;
                    arr_cubic3[0] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;

                    std::complex<double> v3_tmp1 = anharmonic_core->V3(arr_cubic1);

                    for (unsigned int is3 = 0; is3 < ns; ++is3) {
                        double omega3 = eval_in[ik3][is3];

                        arr_cubic2[1] = ns * ik3 + is3;
                        arr_cubic3[1] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;

                        for (unsigned int is4 = 0; is4 < ns; ++is4) {
                            double omega4 = eval_in[ik4][is4];

                            arr_cubic3[2] = ns * ik4 + is4;
                            arr_cubic4[0] = ns * kmesh_in->kindex_minus_xk[ik4] + is4;

                            std::complex<double> v3_tmp3 = anharmonic_core->V3(arr_cubic3);

                            for (unsigned int is5 = 0; is5 < ns; ++is5) {
                                double omega5 = eval_in[ik5][is5];

                                arr_cubic2[2] = ns * ik5 + is5;
                                arr_cubic4[1] = ns * kmesh_in->kindex_minus_xk[ik5] + is5;

                                std::complex<double> v3_tmp2 = anharmonic_core->V3(arr_cubic2);
                                std::complex<double> v3_tmp4 = anharmonic_core->V3(arr_cubic4);

                                std::complex<double> v_prod = v3_tmp1 * v3_tmp2 * v3_tmp3 * v3_tmp4;

                                for (int ip1 = 1; ip1 >= -1; ip1 -= 2) {
                                    double dp1 = static_cast<double>(ip1) * omega1;

                                    for (int ip2 = 1; ip2 >= -1; ip2 -= 2) {
                                        double dp2 = static_cast<double>(ip2) * omega2;
                                        omega_sum[0] = 1.0 / (omega_shift + dp1 - dp2);

                                        for (int ip3 = 1; ip3 >= -1; ip3 -= 2) {
                                            double dp3 = static_cast<double>(ip3) * omega3;

                                            for (int ip4 = 1; ip4 >= -1; ip4 -= 2) {
                                                double dp4 = static_cast<double>(ip4) * omega4;

                                                double D2 = dp4 - dp3 - dp2;
                                                double D2_inv = 1.0 / D2;
                                                omega_sum[3] = 1.0 / (omega_shift + dp1 + dp3 - dp4);

                                                for (int ip5 = 1; ip5 >= -1; ip5 -= 2) {
                                                    double dp5 = static_cast<double>(ip5) * omega5;

                                                    double D1 = dp5 - dp3 - dp1;
                                                    double D1_inv = 1.0 / D1;
                                                    double D12_inv = D1_inv * D2_inv;

                                                    omega_sum[1] = 1.0 / (omega_shift - dp4 + dp5);
                                                    omega_sum[2] = 1.0 / (omega_shift - dp2 - dp3 + dp5);

                                                    for (i = 0; i < N; ++i) {
                                                        double T_tmp = T[i];

                                                        double n1 = thermodynamics->fB(dp1, T_tmp);
                                                        double n2 = thermodynamics->fB(dp2, T_tmp);
                                                        double n3 = thermodynamics->fB(dp3, T_tmp);
                                                        double n4 = thermodynamics->fB(dp4, T_tmp);
                                                        double n5 = thermodynamics->fB(dp5, T_tmp);

                                                        double N12 = n1 - n2;
                                                        double N34 = n3 - n4;
                                                        double N35 = n3 - n5;

                                                        N_prod[0] = N12 * (1.0 + n3);
                                                        N_prod[1] = (1.0 + n2 + n3) * (1.0 + n5) - (1.0 + n1 + n3) * (
                                                                1.0 + n4);
                                                        N_prod[2] = (1.0 + n2) * N35 - n3 * (1.0 + n5);
                                                        N_prod[3] = -((1.0 + n1) * N34 - n3 * (1.0 + n4));

                                                        ret_mpi[i]
                                                                += v_prod *
                                                                   static_cast<double>(ip1 * ip2 * ip3 * ip4 * ip5)
                                                                   * (D12_inv
                                                                      * (N_prod[0] * omega_sum[0]
                                                                         + N_prod[1] * omega_sum[1]
                                                                         + N_prod[2] * omega_sum[2]
                                                                         + N_prod[3] * omega_sum[3])
                                                                      +
                                                                      N12 * ((1.0 + n5) * D1_inv
                                                                             - (1.0 + n4) * D2_inv)
                                                                      * omega_sum[0] * omega_sum[1]);
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double factor = 1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 7));
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}

void Selfenergy::selfenergy_i(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*

    Diagram (i)
    Matrix elements that appear : V3^2 V4
    Computational cost          : O(N_k^2 * N^4)
    Note                        : Double pole when omega2 = omega4.
    : No frequency dependence.

    */

    unsigned int i;
    unsigned int is1, is3;
    unsigned int arr_quartic[4];
    unsigned int arr_cubic1[3], arr_cubic2[3];
    const auto nk = kmesh_in->nk;
    const auto xk = kmesh_in->xk;

    int ip1, ip2, ip3;

    double omega1, omega3;
    double n1, n2, n3;
    double dp1, dp2, dp3;
    double D123;
    double T_tmp;
    double xk_tmp[3];
    double N_prod[2];
    double T_inv;

    std::complex<double> v3_tmp1, v3_tmp2;
    std::complex<double> v_prod;
    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);

    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0);

    arr_quartic[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_quartic[3] = ns * knum + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {
        for (unsigned int ik2 = 0; ik2 < nk; ++ik2) {

            unsigned int ik4 = ik2;
            xk_tmp[0] = xk[ik2][0] - xk[ik1][0];
            xk_tmp[1] = xk[ik2][1] - xk[ik1][1];
            xk_tmp[2] = xk[ik2][2] - xk[ik1][2];

            const auto ik3 = kmesh_in->get_knum(xk_tmp);

            for (unsigned int is2 = 0; is2 < ns; ++is2) {
                double omega2 = eval_in[ik2][is2];

                arr_quartic[1] = ns * ik2 + is2;
                arr_cubic2[0] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;

                for (unsigned int is4 = 0; is4 < ns; ++is4) {
                    double omega4 = eval_in[ik4][is4];

                    arr_quartic[2] = ns * kmesh_in->kindex_minus_xk[ik4] + is4;
                    arr_cubic1[2] = ns * ik4 + is4;

                    std::complex<double> v4_tmp = anharmonic_core->V4(arr_quartic);

                    if (std::abs(omega2 - omega4) < eps) {

                        for (is3 = 0; is3 < ns; ++is3) {
                            omega3 = eval_in[ik3][is3];

                            arr_cubic1[1] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;
                            arr_cubic2[2] = ns * ik3 + is3;

                            for (is1 = 0; is1 < ns; ++is1) {
                                omega1 = eval_in[ik1][is1];

                                arr_cubic1[0] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;
                                arr_cubic2[1] = ns * ik1 + is1;

                                v3_tmp1 = anharmonic_core->V3(arr_cubic1);
                                v3_tmp2 = anharmonic_core->V3(arr_cubic2);

                                v_prod = v4_tmp * v3_tmp1 * v3_tmp2;

                                for (ip1 = 1; ip1 >= -1; ip1 -= 2) {
                                    dp1 = static_cast<double>(ip1) * omega1;

                                    for (ip2 = 1; ip2 >= -1; ip2 -= 2) {
                                        dp2 = static_cast<double>(ip2) * omega2;

                                        double dp2_inv = 1.0 / dp2;

                                        for (ip3 = 1; ip3 >= -1; ip3 -= 2) {
                                            dp3 = static_cast<double>(ip3) * omega3;

                                            D123 = 1.0 / (dp1 + dp2 + dp3);

                                            for (i = 0; i < N; ++i) {
                                                T_tmp = T[i];

                                                n1 = thermodynamics->fB(dp1, T_tmp);
                                                n2 = thermodynamics->fB(dp2, T_tmp);
                                                n3 = thermodynamics->fB(dp3, T_tmp);

                                                N_prod[0] = (1.0 + n1) * (1.0 + n3) + n2 * (1.0 + n2 + n3);
                                                N_prod[1] = n2 * (1.0 + n2) * (1.0 + n2 + n3);

                                                if (std::abs(T_tmp) < eps) {
                                                    T_inv = 0.0;
                                                } else {
                                                    T_inv = 1.0 / (thermodynamics->T_to_Ryd * T_tmp);
                                                }

                                                ret_mpi[i]
                                                        += v_prod * static_cast<double>(ip1 * ip3)
                                                           * (D123 * (N_prod[0] * D123 + N_prod[1] * T_inv + N_prod[0] *
                                                                                                             dp2_inv));
                                            }
                                        }
                                    }
                                }
                            }
                        }

                    } else {
                        for (is3 = 0; is3 < ns; ++is3) {
                            omega3 = eval_in[ik3][is3];

                            arr_cubic1[1] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;
                            arr_cubic2[2] = ns * ik3 + is3;

                            for (is1 = 0; is1 < ns; ++is1) {
                                omega1 = eval_in[ik1][is1];

                                arr_cubic1[0] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;
                                arr_cubic2[1] = ns * ik1 + is1;

                                v3_tmp1 = anharmonic_core->V3(arr_cubic1);
                                v3_tmp2 = anharmonic_core->V3(arr_cubic2);

                                v_prod = v4_tmp * v3_tmp1 * v3_tmp2;

                                for (ip1 = 1; ip1 >= -1; ip1 -= 2) {
                                    dp1 = static_cast<double>(ip1) * omega1;

                                    for (ip2 = 1; ip2 >= -1; ip2 -= 2) {
                                        dp2 = static_cast<double>(ip2) * omega2;

                                        for (ip3 = 1; ip3 >= -1; ip3 -= 2) {

                                            dp3 = static_cast<double>(ip3) * omega3;
                                            D123 = 1.0 / (dp1 - dp2 + dp3);

                                            for (int ip4 = 1; ip4 >= -1; ip4 -= 2) {
                                                double dp4 = static_cast<double>(ip4) * omega4;

                                                double D24 = 1.0 / (dp2 - dp4);
                                                double D134 = 1.0 / (dp1 + dp3 - dp4);

                                                for (i = 0; i < N; ++i) {
                                                    T_tmp = T[i];

                                                    n1 = thermodynamics->fB(dp1, T_tmp);
                                                    n2 = thermodynamics->fB(dp2, T_tmp);
                                                    n3 = thermodynamics->fB(dp3, T_tmp);
                                                    double n4 = thermodynamics->fB(dp4, T_tmp);

                                                    ret_mpi[i]
                                                            += v_prod * static_cast<double>(ip1 * ip2 * ip3 * ip4)
                                                               * ((1.0 + n1 + n3) * D24 * (n4 * D134 - n2 * D123)
                                                                  + D123 * D134 * n1 * n3);
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    double factor = -1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 7));
    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}

void Selfenergy::selfenergy_j(const unsigned int N,
                              const double *T,
                              const double omega,
                              const unsigned int knum,
                              const unsigned int snum,
                              const KpointMeshUniform *kmesh_in,
                              const double *const *eval_in,
                              const std::complex<double> *const *const *evec_in,
                              std::complex<double> *ret) const
{
    /*

    Diagram (j)
    Matrix elements that appear : V4^2
    Computational cost          : O(N_k^2 * N^3)
    Note                        : Double pole when omega1 = omega3

    */

    unsigned int i;
    unsigned int is2;
    unsigned int arr_quartic1[4], arr_quartic2[4];
    const auto nk = kmesh_in->nk;

    double T_tmp;
    double n1, n2;
    double omega2;
    double D13[2];
    double T_inv;

    std::complex<double> v4_tmp2;
    std::complex<double> v_prod;
    std::complex<double> *ret_mpi;

    allocate(ret_mpi, N);

    for (i = 0; i < N; ++i) ret_mpi[i] = std::complex<double>(0.0, 0.0);

    arr_quartic1[0] = ns * kmesh_in->kindex_minus_xk[knum] + snum;
    arr_quartic1[3] = ns * knum + snum;

    for (unsigned int ik1 = mympi->my_rank; ik1 < nk; ik1 += mympi->nprocs) {

        unsigned int ik3 = ik1;

        for (unsigned int ik2 = 0; ik2 < nk; ++ik2) {

            for (unsigned int is1 = 0; is1 < ns; ++is1) {
                double omega1 = eval_in[ik1][is1];

                arr_quartic1[1] = ns * ik1 + is1;
                arr_quartic2[0] = ns * kmesh_in->kindex_minus_xk[ik1] + is1;

                for (unsigned int is3 = 0; is3 < ns; ++is3) {
                    double omega3 = eval_in[ik1][is3];

                    arr_quartic1[2] = ns * kmesh_in->kindex_minus_xk[ik3] + is3;
                    arr_quartic2[3] = ns * ik3 + is3;

                    std::complex<double> v4_tmp1 = anharmonic_core->V4(arr_quartic1);

                    if (std::abs(omega1 - omega3) < eps) {
                        double omega1_inv = 1.0 / omega1;

                        for (is2 = 0; is2 < ns; ++is2) {
                            omega2 = eval_in[ik2][is2];

                            arr_quartic2[1] = ns * ik2 + is2;
                            arr_quartic2[2] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;

                            v4_tmp2 = anharmonic_core->V4(arr_quartic2);

                            v_prod = v4_tmp1 * v4_tmp2;

                            for (i = 0; i < N; ++i) {
                                T_tmp = T[i];

                                n1 = thermodynamics->fB(omega1, T_tmp);
                                n2 = thermodynamics->fB(omega2, T_tmp);

                                if (std::abs(T_tmp) < eps) {
                                    T_inv = 0.0;
                                } else {
                                    T_inv = 1.0 / (thermodynamics->T_to_Ryd * T_tmp);
                                }

                                ret_mpi[i]
                                        += v_prod * (2.0 * n2 + 1.0)
                                           * (-2.0 * (1.0 + n1) * n1 * T_inv
                                              - (2.0 * n1 + 1.0) * omega1_inv);
                            }
                        }
                    } else {

                        D13[0] = 1.0 / (omega1 - omega3);
                        D13[1] = 1.0 / (omega1 + omega3);

                        for (is2 = 0; is2 < ns; ++is2) {
                            omega2 = eval_in[ik2][is2];

                            arr_quartic2[1] = ns * ik2 + is2;
                            arr_quartic2[2] = ns * kmesh_in->kindex_minus_xk[ik2] + is2;

                            v4_tmp2 = anharmonic_core->V4(arr_quartic2);

                            v_prod = v4_tmp1 * v4_tmp2;

                            for (i = 0; i < N; ++i) {
                                T_tmp = T[i];

                                n1 = thermodynamics->fB(omega1, T_tmp);
                                n2 = thermodynamics->fB(omega2, T_tmp);
                                double n3 = thermodynamics->fB(omega3, T_tmp);

                                ret_mpi[i]
                                        += v_prod * 2.0
                                           * ((n1 - n3) * D13[0] - (1.0 + n1 + n3) * D13[1]);
                            }
                        }
                    }
                }
            }
        }
    }

    double factor = -1.0 / (std::pow(static_cast<double>(nk), 2) * std::pow(2.0, 6));

    for (i = 0; i < N; ++i) ret_mpi[i] *= factor;

    mpi_reduce_complex(N, ret_mpi, ret);

    deallocate(ret_mpi);
}
