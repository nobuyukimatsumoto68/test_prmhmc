#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <random>
#include <limits>

#include <Eigen/Core>
// #include <eigen-3.4.0/unsupported/Eigen/MatrixFunctions>
#include <unsupported/Eigen/MatrixFunctions>

int SWITCH = 3;
// 0: HMC
// 1: PRMHMC, differentiable rotation
// 2: PRMHMC, non-differentiable rotation
// 3: non-exact RMHMC, constant M
// 4: non-exact RMHMC, U-dependent M
// 10: PRMHMC, dS

const int Nc=2;
const int NA=Nc*Nc-1;
double beta=1.0;

const int seed = 45;
const double stot = 1.0;
const int nstep = 4;

const int ntraj = 1e7; // 1e6
const int nthermalize = 1e3;
const int interval = 10;


using Double = double;
using Complex=std::complex<Double>;
using MR = Eigen::Matrix< Double, Eigen::Dynamic, Eigen::Dynamic >;
using MC = Eigen::Matrix< Complex, Eigen::Dynamic, Eigen::Dynamic >;
using VR = Eigen::Matrix< Double, Eigen::Dynamic, 1 >;
using AR = Eigen::Array< Double, Eigen::Dynamic, 1 >;
constexpr Complex I = Complex(0.0, 1.0);

constexpr int Logprec = 15;
constexpr Double TOL = std::pow(10,-Logprec);

std::vector<MC> _t;

#include "header.h"

int main(int argc, char *argv[]){
  if(argc==3) {
    SWITCH = std::atoi(argv[1]);
    beta = std::atof(argv[2]);
  }

  _t = get_ta();
  gen.seed(seed);

  MC U = MC::Identity(Nc,Nc);

  AR MKinv_c = AR::Zero(NA);
  MKinv_c(0) = 1.0;
  MKinv_c(1) = 0.0;
  MKinv_c(2) = 0.0;

  AR base(NA);
  base << 1.0, 1.2, 0.8;
  std::function<AR(const MC&)> MKinv = [base](const MC& U) {
    AR res = base;

    if(SWITCH==4){
      res *= U.trace().real()/Nc;
      res = 2.0 + res;
    }
    return res;
  };

  std::function<std::vector<AR>(const MC&)> dMKinv = [base](const MC& U) {
    std::vector<AR> res(NA, AR::Zero(NA));

    if(SWITCH==4){
      AR tmp = base;
      for(int a=0; a<NA; a++) res[a] = (_t[a]*U).trace().real()/Nc * tmp;
    }
    return res;
  };

  std::function<MR(const MC&)> OKX = [](const MC& U) {

    // -- rotation --
    Double alpha = 0.0;
    if(SWITCH==1 || SWITCH==3 || SWITCH==4) alpha = U.trace().real()/Nc;
    if(SWITCH==2) alpha = std::acos( U.trace().real()/Nc );

    MR res(NA, NA);
    assert(Nc==2);
    if(SWITCH==1 || SWITCH==2 || SWITCH==3 || SWITCH==4){
      res <<
        std::cos(alpha), -std::sin(alpha), 0.0,
        std::sin(alpha), std::cos(alpha), 0.0,
        0.0, 0.0, 1.0;
    }

    if(SWITCH==10){
      VR ds = dS(U);
      ds /= ds.norm();
      res <<
        ds(0), ds(1), ds(2),
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0;
    }

    return res;
  };


  std::function<std::vector<MR>(const MC&)> dOKX = [](const MC& U) {
    std::vector<MR> res(NA, MR::Zero(NA,NA));
    assert(Nc==2);

    MR tmp(NA, NA);

    // -- rotation --
    const Double x = U.trace().real()/Nc;
    Double alpha = 0.0;
    if(SWITCH==1 || SWITCH==3 || SWITCH==4) alpha = x;
    if(SWITCH==2) alpha = std::acos( x );

    tmp <<
      -std::sin(alpha), -std::cos(alpha), 0.0,
      std::cos(alpha), -std::sin(alpha), 0.0,
      0.0, 0.0, 0.0;

    const Double dacos = (std::abs(x-1.0)<1.0e-15) ? 0.0 : -1.0/std::sqrt( 1.0-x*x );
    for(int a=0; a<NA; a++) {
      const Double dx = (_t[a]*U).trace().real()/Nc;
      if(SWITCH==1 || SWITCH==3 || SWITCH==4) res[a] = dx * tmp;
      if(SWITCH==2) res[a] = dx * dacos * tmp;
    }

    if(SWITCH==10){
      std::vector<VR> dds = d_dS(U);
      const VR ds = dS(U);
      Double norm_ds = ds.norm();
      for(int a=0; a<NA; a++){
        res[a] <<
          dds[a](0)/norm_ds - dds[a].dot(ds) * ds(0) / std::pow(norm_ds,3),
          dds[a](1)/norm_ds - dds[a].dot(ds) * ds(1) / std::pow(norm_ds,3),
          dds[a](2)/norm_ds - dds[a].dot(ds) * ds(2) / std::pow(norm_ds,3),
          0.0, 0.0, 0.0,
          0.0, 0.0, 0.0;
      }
    }

    return res;
  };



  HMC hmc(stot, nstep, MKinv_c, OKX);
  Nonexact_HMC hmc2(stot, nstep, MKinv, dMKinv, OKX, dOKX);

  bool is_accept;
  Double dH;

  Obs obs1;
  Obs obs2;

  std::string filename=std::to_string(beta)+"_"+std::to_string(SWITCH)+".log";
  std::ofstream ofs(filename, std::ofstream::trunc);

  for(int n=0; n<nthermalize; n++) hmc(U, is_accept, dH);

  for(int n=0; n<ntraj; n++){
    project_SUNc(U);

    hmc(U, is_accept, dH);

    if(SWITCH==0) hmc(U, is_accept, dH, true); // regular
    else if(SWITCH==1 || SWITCH==2 || SWITCH==10) hmc(U, is_accept, dH, true, true); // pseudo
    else if(SWITCH==3 || SWITCH==4) hmc2(U, is_accept, dH, true); // non-exact

    hmc(U, is_accept, dH);

    if( (n+1)%interval==0 ){
      const Double tr = U.trace().real();

      ofs << std::setprecision(Logprec)
          << std::setw(8) << n << " "
          << std::setw(20) << tr << " "
          << std::setw(8) << is_accept << " "
          << std::setw(20) << dH << " "
          << std::endl;

      obs1.meas( tr );
      obs2.meas( tr*tr );
    }
  }

  obs1.do_it();
  obs2.do_it();

  const Characters ch;
  const Double ch0 = ch.get0();
  const Double ch1 = ch.get1();
  const Double ch2 = ch.get2();
  const Double ch11 = ch.get11();

  std::cout << std::setprecision(Logprec)
            << std::setw(20) << "# <O>" << " "
            << std::setw(20) << "err" << " "
            << std::setw(20) << "exact" << " "
            << std::setw(20) << "sigma" << " "
            << std::setw(10) << "beta" << " "
            << std::setw(10) << "ntraj" << " "
            << std::setw(10) << "interval" << " "
            << std::setw(5) << "switch" << " "
            << std::endl;

  std::cout << std::setprecision(Logprec)
            << std::setw(20) << obs1.mean << " "
            << std::setw(20) << obs1.err << " "
            << std::setw(20) << ch1/ch0 << " "
            << std::setw(20) << (obs1.mean-ch1/ch0)/obs1.err << " "
            << std::setw(10) << beta << " "
            << std::setw(10) << ntraj << " "
            << std::setw(10) << interval << " "
            << std::setw(5) << SWITCH << " "
            << std::endl;

  std::cout << std::setprecision(Logprec)
            << std::setw(20) << obs2.mean << " "
            << std::setw(20) << obs2.err << " "
            << std::setw(20) << ch2/ch0 + ch11/ch0 << " "
            << std::setw(20) << (obs2.mean-(ch2/ch0 + ch11/ch0))/obs2.err << " "
            << std::setw(10) << beta << " "
            << std::setw(10) << ntraj << " "
            << std::setw(10) << interval << " "
            << std::setw(5) << SWITCH << " "
            << std::endl;

  return 0;
}







// std::cout << "double: "
//           << std::numeric_limits<double>::digits10
//           << std::endl
//           << "long double: "
//           << std::numeric_limits<Double>::digits10
//           << std::endl;


// // -- trivial --
// res = MR::Identity(NA, NA);
// std::vector<Double> euler = get_Euler(U);
// MR res(NA, NA), tmp1(NA, NA), tmp2(NA, NA), tmp3(NA, NA);
// assert(Nc==2);
// tmp1 <<
//   std::cos(euler[0]), -std::sin(euler[0]), 0.0,
//   std::sin(euler[0]), std::cos(euler[0]), 0.0,
//   0.0, 0.0, 1.0;
// tmp2 <<
//   1.0, 0.0, 0.0,
//   0.0, std::cos(euler[1]), -std::sin(euler[1]),
//   0.0, std::sin(euler[1]), std::cos(euler[1]);
// tmp3 <<
//   std::cos(euler[2]), 0.0, std::sin(euler[2]),
//   0.0, 1.0, 0.0,
//   -std::sin(euler[2]), 0.0, std::cos(euler[2]);
// res = tmp1; // *tmp2*tmp3;


  // const int aa=0;
  // Double eps = 1.0e-5;
  // MC Up = (MC::Identity(Nc,Nc)+eps*_t[aa])*U;
  // MC Um = (MC::Identity(Nc,Nc)-eps*_t[aa])*U;
  // std::cout << "mkinv(U) numeric deriv: " << std::endl
  //           << ( MKinv(Up)-MKinv(Um) )/(2.0*eps) << std::endl;
  // std::cout << "constructed  : " << std::endl
  //           << dMKinv(U)[aa] << std::endl;
  // // std::cout << "check: " << std::endl
  // //           << OKX(U) << std::endl;
  // std::cout << "OKX(U) numeric deriv: " << std::endl
  //           << ( OKX(Up)-OKX(Um) )/(2.0*eps) << std::endl;
  // auto dokx = dOKX(U);
  // std::cout << "constructed  : " << std::endl
  //           << dokx[aa] << std::endl;

  // VR pi = hmc2.gen_pi(U);
  // std::cout << "H2 numeric deriv: " << std::endl
  //           << ( hmc2.H(pi,Up)-hmc2.H(pi,Um) )/(2.0*eps) << std::endl;
  // std::cout << "constructed  : " << std::endl
  //           << hmc2.dH_dU(pi,U)[aa] << std::endl;

  // VR pp = pi;
  // VR pm = pi;
  // pp[aa] += eps;
  // pm[aa] -= eps;
  // std::cout << "H2 numeric deriv: " << std::endl
  //           << ( hmc2.H(pp,U)-hmc2.H(pm,U) )/(2.0*eps) << std::endl;
  // std::cout << "constructed  : " << std::endl
  //           << hmc2.dH_dp(pi,U)[aa] << std::endl;
