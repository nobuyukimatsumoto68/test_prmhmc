#include <cmath>
#include <functional>

std::mt19937 gen;
std::normal_distribution<double> normal(0.0, 1.0);
std::uniform_real_distribution<double> uniform(0.0, 1.0);


std::vector<MC>get_ta(){
  std::vector<MC> _t;
  // generators[0] << 1.0, 0.0, 0.0, 1.0;
  // generators[1] << 0.0, 1.0, 1.0, 0.0;
  // generators[2] << 0.0,  -I,   I, 0.0;
  // generators[3] << 1.0, 0.0, 0.0,-1.0;
  for(int i=0; i<Nc; i++){
    for(int j=i+1; j<Nc; j++){
      MC tmp = MC::Zero(Nc, Nc);
      tmp(i,j)=1.0;
      tmp(j,i)=1.0;
      _t.push_back(tmp);

      tmp = MC::Zero(Nc, Nc);
      tmp(i,j)= -I;
      tmp(j,i)=  I;
      _t.push_back(tmp);}
  }

  for(int m=1; m<Nc; m++){
    MC tmp = MC::Zero(Nc, Nc);
    for(int i=0; i<Nc; i++){
      if(i<m) tmp(i,i) = 1.0;
      else if(i==m) tmp(i,i) = -1.0*m;
      else tmp(i,i) = 0.0;
    }
    tmp *= std::sqrt( 2.0 / (m*(m+1.0)) );
    _t.push_back(tmp);
  }
  for(MC& ta : _t) ta *= -0.5*I;

  return _t;
}


void project_SUNc(MC& U){
  MC log = U.log();
  log = 0.5 * (log - log.adjoint());
  log -= log.trace() * MC::Identity(Nc,Nc)/Nc;
  U = log.exp();
}


Double S(const MC& U){
  return -beta/Nc * U.trace().real();
}

Double dS(const MC& U, const int a){
  return -beta/Nc * (_t[a]*U).trace().real();
}

VR dS(const MC& U){
  VR res = VR::Zero(NA);
  for(int a=0; a<NA; a++){
    res(a) = dS(U,a);
  }
  return res;
}

Double da_db_S(const MC& U, const int a, const int b){
  return -beta/Nc * (_t[b]*_t[a]*U).trace().real();
}

std::vector<VR> d_dS(const MC& U){
  std::vector<VR> res(NA, VR::Zero(NA));
  for(int a=0; a<NA; a++){
    for(int b=0; b<NA; b++){
      res[a](b) = da_db_S(U,a,b);
    }
  }
  return res;
}




std::vector<Double> get_Euler( const MC& U ){
  const Complex a = U(0,0);
  const Complex b = U(0,1);

  const Double theta = 2.0*std::acos( std::abs(a) );
  const Double phi = std::arg( a ) + std::arg( b ) + M_PI;
  const Double psi = std::arg( a ) - std::arg( b ) + M_PI;

  return std::vector<Double>{theta, phi, psi};
}



struct HMC {
  const Double stot;
  const int nstep;
  const Double tau;
  const AR MKinv;
  const std::function<MR(const MC&)> OKX;

  HMC(const Double stot_,
      const Double nstep_,
      const AR& MKinv_=VR::Constant(NA,1.0),
      const std::function<MR(const MC&)>& OKX_=[](const MC&){return MR::Identity(NA,NA);})
    : stot(stot_)
    , nstep(nstep_)
    , tau(stot/nstep)
    , MKinv( MKinv_ )
    , OKX( OKX_ )
  {}

  void find_UH(VR& pK,
               MC& U,
               const int NITER_=1e4,
               const Double TOL_=TOL) const {
    MC UH = U;
    MC UHold = UH;

    Double norm = 1.0;
    for(int i=0; i<NITER_; i++){
      VR pKH = pK - 0.5*tau*OKX(UH)*dS(U);
      VR rhoX = OKX(UH).transpose() * ( pKH.array()*MKinv ).matrix();

      MC shldr = MC::Zero(Nc,Nc);
      for(int a=0; a<NA; a++) shldr += _t[a]*rhoX(a);
      UH = (0.5*tau*shldr).exp() * U;
      norm = (UH-UHold).norm()/Nc;
      // std::cout << "norm = " << norm << std::endl;
      if(norm<TOL_) break;
      else if(i>NITER_-2) assert(false);
      UHold = UH;
    }
    pK = pK - 0.5*tau*OKX(UH)*dS(U);
    U = UH;
  }

  void onestep(VR& pi, MC& U) const {
    pi -= 0.5*tau*dS(U);

    MC shldr = MC::Zero(Nc,Nc);
    for(int a=0; a<NA; a++) shldr += _t[a]*pi(a);
    U = (tau*shldr).exp() * U;

    pi -= 0.5*tau*dS(U);
  }

  void onestep_implicit(VR& pi, MC& U) const {
    find_UH(pi, U);
    const MR OH = OKX(U);
    VR rhoX = OH.transpose() * ( pi.array()*MKinv ).matrix();

    MC shldr = MC::Zero(Nc,Nc);
    for(int a=0; a<NA; a++) shldr += _t[a]*rhoX(a);
    U = (0.5*tau*shldr).exp() * U;
    pi = pi - 0.5*tau*OH*dS(U);
  }


  Double H(const VR& pi, const MC& U,
           const bool is_implicit ) const {
    Double res;
    if(!is_implicit) res = 0.5 * pi.dot(pi) + S(U);
    else res = 0.5 * pi.dot( ( pi.array()*MKinv ).matrix() ) + S(U);
    return res;
  }


  VR gen_pi(const bool is_implicit) const {
    VR pi = VR::Zero(NA);
    for(int a=0; a<NA; a++){
      pi(a) = normal(gen);
      if(is_implicit) {
        const Double c = MKinv(a);
        if(c<TOL) pi(a) = 0.0;
        else pi(a) /= std::sqrt( c );
      }
    }
    return pi;
  }


  void operator()(MC& U, bool& is_accept, Double& dH,
                  const bool is_reverse_test=false,
                  const bool is_implicit=false ) const {
    VR pi = gen_pi(is_implicit);

    const Double Hin = H(pi, U, is_implicit);
    VR pi0 = pi;
    MC U0 = U;

    for(int n=0; n<nstep; n++) {
      if(!is_implicit) onestep(pi, U);
      else onestep_implicit(pi, U);
    }

    if(is_reverse_test) {
      VR pi_rev = -pi;
      MC U_rev = U;

      for(int n=0; n<nstep; n++) {
        if(!is_implicit) onestep(pi_rev, U_rev);
        else onestep_implicit(pi_rev, U_rev);
      }

      Double pi_diff = (pi_rev+pi0).norm() / std::sqrt(NA) / std::sqrt(nstep) / beta;
      Double U_diff = (U_rev-U0).norm() / Nc / std::sqrt(nstep) / beta;
      assert( pi_diff < 10*TOL );
      assert( U_diff < 10*TOL );
    }

    const Double Hfi = H(pi, U, is_implicit);
    dH = Hfi - Hin;

    const Double r = uniform(gen);
    is_accept = false;
    if( r<std::exp(-dH) ) is_accept = true;
    else U = U0;
  }

};




struct Nonexact_HMC {
  const Double stot;
  const int nstep;
  const Double tau;
  const std::function<AR(const MC&)> MKinv;
  const std::function<std::vector<AR>(const MC&)> dMKinv;
  const std::function<MR(const MC&)> OKX;
  const std::function<std::vector<MR>(const MC&)> dOKX;

  Nonexact_HMC(const Double stot_,
               const Double nstep_,
               const std::function<AR(const MC&)> MKinv_,
               const std::function<std::vector<AR>(const MC&)> dMKinv_,
               const std::function<MR(const MC&)>& OKX_,
               const std::function<std::vector<MR>(const MC&)>& dOKX_)
    : stot(stot_)
    , nstep(nstep_)
    , tau(stot/nstep)
    , MKinv( MKinv_ )
    , dMKinv( dMKinv_ )
    , OKX( OKX_ )
    , dOKX( dOKX_ )
  {}

  VR find_pH(const VR& p,
             const MC& U,
             const int NITER_=1e4,
             const Double TOL_=TOL) const {

    VR pH = p - 0.5*tau*dH_dU(p,U);
    VR pHold = pH;

    Double norm = 1.0;
    for(int i=0; i<NITER_; i++){
      pH = p - 0.5*tau*dH_dU(pH,U);
      norm = (pH-pHold).norm()/std::sqrt(NA);
      // std::cout << "norm = " << norm << std::endl;

      if(norm<TOL_) break;
      else if(i>NITER_-2) {
        std::clog << "norm = " << norm << std::endl;
        std::clog << "U = " << std::endl
                  << U << std::endl;
        std::clog << "p = " << std::endl
                  << p.transpose() << std::endl;
        assert(false);
      }
      pHold = pH;
    }

    return pH;
  }


  MC find_Up(const MC& UH,
             const VR& pH,
             const int NITER_=1e4,
             const Double TOL_=TOL) const {
    MC Up = UH;
    MC Upold = Up;

    Double norm = 1.0;
    for(int i=0; i<NITER_; i++){
      VR rho = dH_dp(pH,Up);

      MC shldr = MC::Zero(Nc,Nc);
      for(int a=0; a<NA; a++) shldr += _t[a]*rho(a);
      Up = (0.5*tau*shldr).exp() * UH;
      norm = (Up-Upold).norm()/Nc;

      if(norm<TOL_) break;
      else if(i>NITER_-2) {
        std::clog << "norm = " << norm << std::endl;
        std::clog << "UH = " << std::endl
                  << UH << std::endl;
        std::clog << "pH = " << std::endl
                  << pH.transpose() << std::endl;

        assert(false);
      }
      Upold = Up;
    }
    return Up;
  }


  void onestep(VR& pi, MC& U) const {
    pi = find_pH(pi, U);

    const VR rho = dH_dp(pi, U);
    MC shldr = MC::Zero(Nc,Nc);
    for(int a=0; a<NA; a++) shldr += _t[a]*rho(a);
    U = (0.5*tau*shldr).exp() * U;

    U = find_Up(U, pi);

    pi = pi - 0.5*tau * dH_dU(pi, U);
  }


  Double H(const VR& pX, const MC& U) const {
    const VR pK = OKX(U)*pX;
    Double res = 0.5 * pK.dot( ( pK.array()*MKinv(U) ).matrix() ) + S(U);
    return res;
  }


  VR dH_dU( const VR& pX, const MC& U ) const {
    const std::vector<MR> dokx = dOKX(U);
    std::vector<VR> dpK_dU(NA);
    for(int a=0; a<NA; a++) dpK_dU[a] = dokx[a] * pX;
    const VR pK = OKX(U)*pX;

    std::vector<AR> dmkinv_dU = dMKinv(U);

    VR res = dS(U);
    for(int a=0; a<NA; a++) {
      res(a) += dpK_dU[a].dot( ( pK.array()*MKinv(U) ).matrix() );
      res(a) += 0.5*pK.dot( ( pK.array()*dmkinv_dU[a] ).matrix() );
    }

    return res;
  }

  VR dH_dp( const VR& pX, const MC& U ) const {
    return ( (OKX(U)*pX).array()*MKinv(U) ).matrix();
  }


  VR gen_pi(const MC& U) const {
    VR pK = VR::Zero(NA);
    for(int a=0; a<NA; a++){
      pK(a) = normal(gen);
      const Double c = MKinv(U)(a);
      if(c<TOL) pK(a) = 0.0;
      else pK(a) /= std::sqrt( c );
    }
    const VR pX = OKX(U).transpose() * pK;

    return pX;
  }


  void operator()(MC& U, bool& is_accept, Double& dH,
                  const bool is_reverse_test=false) const {
    VR pi = gen_pi(U);

    const Double Hin = H(pi, U);
    VR pi0 = pi;
    MC U0 = U;

    for(int n=0; n<nstep; n++) onestep(pi, U);

    if(is_reverse_test) {
      VR pi_rev = -pi;
      MC U_rev = U;

      for(int n=0; n<nstep; n++) onestep(pi_rev, U_rev);

      Double pi_diff = (pi_rev+pi0).norm() / std::sqrt(NA) / std::sqrt(nstep) / beta;
      Double U_diff = (U_rev-U0).norm() / Nc / std::sqrt(nstep) / beta;
      assert( pi_diff < 20*TOL );
      assert( U_diff < 20*TOL );
    }

    const Double Hfi = H(pi, U);
    dH = Hfi - Hin;

    const Double r = uniform(gen);
    is_accept = false;
    if( r<std::exp(-dH) ) is_accept = true;
    else U = U0;
  }

};




struct Obs{
  std::vector<Double> v;
  Double mean;
  Double err;

  void meas(const Double c){
    v.push_back(c);
  }

  void do_it(){
    mean = 0.0L;
    for(Double elem : v) mean += elem;
    mean /= v.size();

    Double tmp = 0.0L;
    for(Double elem : v) {
      tmp += (elem-mean)*(elem-mean);
    }
    tmp /= v.size()-1;
    err = std::sqrt(tmp/v.size());
  };
};


struct Characters{
  std::vector<int> lam0;
  std::vector<int> lam1;
  std::vector<int> lam2;
  std::vector<int> lam11;

  Characters()
    : lam0(Nc, 0)
    , lam1(Nc, 0)
    , lam2(Nc, 0)
    , lam11(Nc, 0)
  {
    lam1[0] = 1;

    lam2[0] = 2;

    lam11[0] = 1;
    lam11[1] = 1;
  }

  MR get_matrix( const int Q, const std::vector<int>& lam ) const {
    MR res(Nc, Nc);
    for(int k=0; k<Nc; k++){
      for(int ell=0; ell<Nc; ell++){
        res(k,ell) = std::exp(-beta/Nc) * std::cyl_bessel_i( std::abs(k-ell+Q+lam[ell]), beta/Nc );
      }
    }
    return res;
  }

  Double get_char( const std::vector<int>& lam ) const {
    Double res = get_matrix(0, lam).determinant();

    Double tmp = res;
    for(int AbsQ=1; AbsQ<400; AbsQ++){
      res += get_matrix( AbsQ, lam).determinant();
      res += get_matrix(-AbsQ, lam).determinant();

      if( std::abs(res-tmp)<TOL ) break;
      tmp = res;
    }
    return res;
  }

  Double get0() const {
    return get_char( lam0 );
  }

  Double get1() const {
    return get_char( lam1 );
  }

  Double get2() const {
    return get_char( lam2 );
  }

  Double get11() const {
    return get_char( lam11 );
  }

};



