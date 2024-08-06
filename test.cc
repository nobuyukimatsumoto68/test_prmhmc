#include <iostream>
#include <iomanip>
#include <fstream>
#include <complex>
#include <vector>
#include <random>
#include <limits>

#include <Eigen/Core>
#include <eigen-3.4.0/unsupported/Eigen/MatrixFunctions>


const int Nc=2;
const int NA=Nc*Nc-1;
const double beta=2.0;

const int seed = 45;
const double stot = 1.0;
const int nstep = 5;

const int ntraj = 1e6; // 1e4
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


int main(){
  _t = get_ta();
  gen.seed(seed);

  // std::string filename="data.dat";
  // std::ofstream ofs(filename, std::ofstream::app);

  // std::cout << "double: "
  //           << std::numeric_limits<double>::digits10
  //           << std::endl
  //           << "long double: "
  //           << std::numeric_limits<Double>::digits10
  //           << std::endl;

  MC U = MC::Identity(Nc,Nc);


  AR MKinv_c = AR::Zero(NA);
  MKinv_c(0) = 1.0;
  MKinv_c(1) = 0.0;
  MKinv_c(2) = 0.0;

  std::function<AR(const MC&)> MKinv = [](const MC& U) {
    AR res(NA);
    res << 1.0, 1.2, 0.8;
    // res *= U.trace().real()/Nc;
    // res = 2.0 + res;
    return res;
  };

  std::function<std::vector<AR>(const MC&)> dMKinv = [](const MC& U) {
    std::vector<AR> res(NA, AR::Zero(NA));
    // AR tmp(NA);
    // tmp << 1.0, 1.2, 0.8;
    // for(int a=0; a<NA; a++) res[a] = (_t[a]*U).trace().real()/Nc * tmp;
    return res;
  };

  std::function<MR(const MC&)> OKX = [](const MC& U) {

    // // -- trivial --
    // res = MR::Identity(NA, NA);

    // -- rotation --
    // const Double alpha = 0.4;
    const Double alpha = std::acos( U.trace().real()/Nc );
    // const Double alpha = std::acos( U.trace().real()/Nc ) / 2.0;
    MR res(NA, NA);
    assert(Nc==2);
    res <<
      std::cos(alpha), -std::sin(alpha), 0.0,
      std::sin(alpha), std::cos(alpha), 0.0,
      0.0, 0.0, 1.0;

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

    // VR ds = dS(U);
    // ds /= ds.norm();
    // MR res(NA, NA);
    // res <<
    //   ds[0], ds[1], ds[2],
    //   ds[1], 0.0, 0.0,
    //   ds[2], 0.0, 0.0;

    return res;
  };


  std::function<std::vector<MR>(const MC&)> dOKX = [](const MC& U) {
    std::vector<MR> res;
    MR tmp(NA, NA);
    assert(Nc==2);

    // -- rotation --
    const Double x = U.trace().real()/Nc;
    const Double alpha = std::acos( x );
    tmp <<
      -std::sin(alpha), -std::cos(alpha), 0.0,
      std::cos(alpha), -std::sin(alpha), 0.0,
      0.0, 0.0, 0.0;

    const Double dacos = (std::abs(x-1.0)<1.0e-15) ? 0.0 : -1.0/std::sqrt( 1.0-x*x );
    for(int a=0; a<NA; a++) {
      const Double dx = (_t[a]*U).trace().real()/Nc;
      res.push_back( dx * dacos * tmp );
    }

    return res;
  };







  HMC hmc(stot, nstep, MKinv_c, OKX);
  Nonexact_HMC hmc2(stot, nstep, MKinv, dMKinv, OKX, dOKX);

  bool is_accept;
  Double dH;

  Obs obs1;
  Obs obs2;

  for(int n=0; n<nthermalize; n++) hmc(U, is_accept, dH);

  for(int n=0; n<ntraj; n++){
    project_SUNc(U);

    hmc(U, is_accept, dH, true);
    // hmc(U, is_accept, dH, true, false);
    hmc(U, is_accept, dH, true, true);
    // hmc2(U, is_accept, dH, true);
    hmc(U, is_accept, dH, true);

    if( (n+1)%interval ){
      const Double tr = U.trace().real();

      std::cout << std::setprecision(Logprec)
                << std::setw(8) << n << " "
                << std::setw(20) << tr << " "
                << std::setw(8) << is_accept << " "
                << std::setw(20) << dH << " "
                << std::endl;

      obs1.meas( tr );
      obs2.meas( tr*tr );
    }
  }

  // for(Double elem : obs1.v) std::cout << elem << std::endl;
  obs1.do_it();
  obs2.do_it();

  const Characters ch;
  const Double ch0 = ch.get0();
  const Double ch1 = ch.get1();
  const Double ch2 = ch.get2();
  const Double ch11 = ch.get11();

  std::cout << std::setprecision(Logprec)
            << "#" << std::setw(20) << "<Re trU>" << " "
            << std::setw(20) << "err" << " "
            << std::setw(20) << "exact" << " "
            << std::setw(5) << "sigma" << " "
            << std::endl;

  std::cout << std::setprecision(Logprec)
            << std::setw(20) << obs1.mean << " "
            << std::setw(20) << obs1.err << " "
            << std::setw(20) << ch1/ch0 << " "
            << std::setw(5) << (obs1.mean-ch1/ch0)/obs1.err << " "
            << std::endl;

  std::cout << std::setprecision(Logprec)
            << std::setw(20) << obs2.mean << " "
            << std::setw(20) << obs2.err << " "
            << std::setw(20) << ch2/ch0 + ch11/ch0 << " "
            << std::setw(5) << (obs2.mean-(ch2/ch0 + ch11/ch0))/obs2.err << " "
            << std::endl;




  // const int aa=2;
  // Double eps = 1.0e-5;
  // MC Up = (MC::Identity(Nc,Nc)+eps*_t[aa])*U;
  // MC Um = (MC::Identity(Nc,Nc)-eps*_t[aa])*U;
  // std::cout << "numeric deriv: " << std::endl
  //           << ( MKinv(Up)-MKinv(Um) )/(2.0*eps) << std::endl;
  // std::cout << "constructed  : " << std::endl
  //           << dMKinv(U)[aa] << std::endl;
  // std::cout << "numeric deriv: " << std::endl
  //           << ( OKX(Up)-OKX(Um) )/(2.0*eps) << std::endl;
  // std::cout << "constructed  : " << std::endl
  //           << dOKX(U)[aa] << std::endl;

  // VR pi = hmc2.gen_pi(U);
  // std::cout << "numeric deriv: " << std::endl
  //           << ( hmc2.H(pi,Up)-hmc2.H(pi,Um) )/(2.0*eps) << std::endl;
  // std::cout << "constructed  : " << std::endl
  //           << hmc2.dH_dU(pi,U)[aa] << std::endl;

  // VR pp = pi;
  // VR pm = pi;
  // pp[aa] += eps;
  // pm[aa] -= eps;
  // std::cout << "numeric deriv: " << std::endl
  //           << ( hmc2.H(pp,U)-hmc2.H(pm,U) )/(2.0*eps) << std::endl;
  // std::cout << "constructed  : " << std::endl
  //           << hmc2.dH_dp(pi,U)[aa] << std::endl;


  return 0;

}
