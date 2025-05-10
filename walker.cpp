/******************** ALL RIGHTS RESERVED ***********************
*****************************************************************
    _/_/_/    _/    _/_/_/   _/_/_/   _/_/_/  Scuola 
   _/        _/    _/       _/       _/  _/   Internazionale
  _/_/_/    _/    _/_/_/   _/_/_/   _/_/_/    Superiore di
     _/    _/        _/       _/   _/  _/     Studi 
_/_/_/    _/    _/_/_/   _/_/_/   _/  _/      Avanzati 
*****************************************************************
*****************************************************************/

#include <iostream>
#include <fstream>
#include "distances.h"
#include "random.h"
#include <math.h>
#include <complex>
#include <algorithm>    // std::min
#include <Eigen>
#include <unsupported/Eigen/CXX11/Tensor>
#include "walker.h"

using Eigen::VectorXcd;
using Eigen::RowVectorXcd;
using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::ArrayXi;
using Eigen::ArrayXXi;
using Eigen::SelfAdjointEigenSolver;
using namespace std;
using namespace std::complex_literals;

//////////////////////////
// Initializing methods //
//////////////////////////

void Walker :: initialize_lattice(ArrayXi & imulti, Eigen::Tensor<int, 3> & ivic){

  _imulti =  imulti;
  _ivic   =  ivic;

  return;
} 

void Walker :: initialize_physical_parameters(int N_e_up, int N_e_do, double P_SpinFlip, double t_hubb, double J, double U_hubb){ 

  _N_e_up     =  N_e_up;
  _N_e_do     =  N_e_do;
  _N_e        = _N_e_up + _N_e_do;

  // Set the model to t-J-U or Heisenberg
  if(_N_e_up==_N_e_do and _N_e_up==(L/2)){
    _actually_t_J = false;
  }
  else{
    _actually_t_J = true;
  }

  _P_SpinFlip = P_SpinFlip;

  _t_hubb     = t_hubb;
  _J          = J;
  _U_hubb     = U_hubb;

  _kel               = ArrayXi::Zero(two_L);
  _iconf             = ArrayXi::Zero(two_L);
  _occupied_sites    = ArrayXi::Zero(_N_e);
  _occupied_sites_up = ArrayXi::Zero(_N_e_up);
  _occupied_sites_do = ArrayXi::Zero(_N_e_do);

  // Generate the initial configuration of the system randomly
  
  int conta_kel = 1;
  int conta_up = 0;

  // Position of the up electrons (randomly chosen)
  while(conta_up < _N_e_up) {
    int index = _rnd.RandInt(0,L);
    if (_iconf[index] == 0) {
      _iconf[index]                = 1;
      _kel[index]                  = conta_kel;
      _occupied_sites[conta_up]    = index;
      _occupied_sites_up[conta_up] = index;
      conta_up += 1;
      conta_kel += 1;
    }
  }

  int conta_do = L-_N_e_up;

  // No double occupancy!!!
  for(int i = 0; i < L; i++) {
    if (_iconf[i] == 0) {
      _iconf[i + L] = -1;
    }
  }

  while(conta_do > _N_e_do) {
    int index_to_be_removed = _rnd.RandInt(0,L);
    if (_iconf[index_to_be_removed + L] == -1) {
      _iconf[index_to_be_removed + L] = 0;
      conta_do -= 1;
    }
  }

  // Position of the down electrons, having exluded double occupancy
  conta_do = 0;
  for(int index = 0; index < L; index++) {
    if (_iconf[index + L] == -1) {
      _kel[index + L]                   = conta_kel;
      _occupied_sites[conta_do+_N_e_up] = index;
      _occupied_sites_do[conta_do]      = index;
      conta_kel += 1;
      conta_do += 1;
    }
  }
  //cout<<endl<<_kel<<endl;
  //cout<<endl<<_occupied_sites<<endl;


  // Initialize W_inv and other matrices
  _W_inv = MatrixXcd::Zero(_N_e_up,_N_e_do); 
  _M_up  = MatrixXcd::Zero(L,_N_e_up);
  _M_up  = MatrixXcd::Zero(_N_e_do,L);

  _C_mat = MatrixXcd::Zero(2,2);
  _appo_Ne_row_vec    = RowVectorXcd::Zero(_N_e_up); 
  _appo_Ne_col_vec    = VectorXcd::Zero(_N_e_do); 

  _appo_f_ij_mat   = MatrixXcd::Zero(L,L);
  _appo_W_inv      = MatrixXcd::Zero(_N_e_up,_N_e_do);

  return;
}

void Walker :: initialize_variational_parameters(MatrixXd & abs_f_ij, MatrixXd & phase_f_ij ){

  _abs_f_ij   = abs_f_ij;
  _phase_f_ij = phase_f_ij;

  _f_ij_mat = MatrixXcd::Zero(L,L);
  build_f_ij_mat(_f_ij_mat);

  // From here on, the small index corresponds to the electron index, the big one to the site index
  // So, the electron "i" is at site _occupied_sites[i], called "R_i"
  // The site "I" is just the site "I"

  _f_mat_Ri_Rj = MatrixXcd::Zero(_N_e_up,_N_e_do);
  for(int i=0;i<_N_e_up;i++){
    for(int j=0;j<_N_e_do;j++){
      _f_mat_Ri_Rj(i,j) = _f_ij_mat(_occupied_sites_up[i],_occupied_sites_do[j]);
    }
  }

  _f_mat_I_Rj  = MatrixXcd::Zero( L ,_N_e_do);
  for(int I=0;I<L;I++){
    for(int j=0;j<_N_e_do;j++){
      _f_mat_I_Rj(I,j) = _f_ij_mat(I,_occupied_sites_do[j]);
    }
  }

  _f_mat_Ri_J  = MatrixXcd::Zero(_N_e_up, L);
  for(int i=0;i<_N_e_up;i++){
    for(int J=0;J<L;J++){
      _f_mat_Ri_J(i,J) = _f_ij_mat(_occupied_sites_up[i],J);
    }
  }

  // Compute the SVD to see if you started from a good configuration
  Eigen::JacobiSVD<Eigen::MatrixXcd> svd(_f_mat_Ri_Rj, Eigen::ComputeThinU | Eigen::ComputeThinV);
  Eigen::VectorXd singular_values = svd.singularValues();
  double lowest_singular_value = singular_values.minCoeff(); // Find the lowest singular value
  // Print the lowest singular value
  if(lowest_singular_value < 1e-10){
    std::cout << "The lowest singular value of f_{R_i,R_j} is: " << lowest_singular_value << std::endl;
  }

  // Compute from scratch the inverse of the matrix f_mat_Ri_Rj and the two products
  _W_inv = (_f_mat_Ri_Rj).fullPivLu().inverse();
  _M_up  = _f_mat_I_Rj * _W_inv;
  _M_do  = _W_inv *_f_mat_Ri_J;

  _count_accept_hopping=0;
  _tot_hopping=0;
  _count_accept_SpinFlip=0;
  _tot_SpinFlip=0;

  return;
}

void Walker :: restart_W_inv_from_scratch(){

  _appo_W_inv = _W_inv;

  // Compute from scratch the inverse of the matrix f_mat_Ri_Rj
  _W_inv = (_f_mat_Ri_Rj).fullPivLu().inverse();

  _M_up  = _f_mat_I_Rj * _W_inv;

  _M_do  = _W_inv *_f_mat_Ri_J;

  // Check the difference between the two inverses
  // If the difference is too big, it means that the fast-update scheme is not working well
  // because W_inv got degraded by the accumulating of numerical errors 
  if( (_appo_W_inv-_W_inv).norm()/_W_inv.norm() > 0.01 ){
    cout<<endl<<"When restarting W_inv from scratch, || W_inv_exact - W_inv_updated || / || W_inv_exact|| = "<<(_appo_W_inv-_W_inv).norm()/_W_inv.norm()<<"; you may want to use a lower N_scra."<<endl;
  }

  return;
}

void Walker :: restart_W_inv_from_scratch_and_check_it(){

  cout<<"W_inv diff = "<<(_W_inv - (_f_mat_Ri_Rj).fullPivLu().inverse()).norm();
  _W_inv = (_f_mat_Ri_Rj).fullPivLu().inverse();
  cout<<"; M_up diff = "<<(_M_up - _f_mat_I_Rj * _W_inv).norm();
  _M_up  = _f_mat_I_Rj * _W_inv;
  cout<<"; M_do diff = "<<(_M_do - _W_inv *_f_mat_Ri_J).norm()<<endl<<endl;
  _M_do  = _W_inv *_f_mat_Ri_J;

  return;
}


// Update the inverse of the matrix f_mat_Ri_Rj after an up-electron hopping
void Walker :: update_W_inv_up_electron_hopping(int l_index_up, int K_new_index_up){

  _M_do += _W_inv.col(l_index_up)*( _f_ij_mat.row(K_new_index_up) -  _M_up.row(K_new_index_up)*_f_mat_Ri_J )/_M_up(K_new_index_up,l_index_up);

  _appo_Ne_row_vec = RowVectorXcd::Zero(_N_e_up);  
  _appo_Ne_row_vec[l_index_up] = complex<double>(1.,0.);

  _appo_Ne_row_vec.noalias() += -_M_up.row(K_new_index_up);
  _appo_Ne_row_vec /= _M_up(K_new_index_up,l_index_up);

  _W_inv += _W_inv.col(l_index_up)*_appo_Ne_row_vec;

  _M_up  += _M_up.col(l_index_up)*_appo_Ne_row_vec;

  return;
}


// Update the inverse of the matrix f_mat_Ri_Rj after an down-electron hopping
void Walker :: update_W_inv_do_electron_hopping(int m_index_do, int I_new_index_do) {

  _M_up += ( _f_ij_mat.col(I_new_index_do) - _f_mat_I_Rj*_M_do.col(I_new_index_do) )*_W_inv.row(m_index_do)/_M_do(m_index_do,I_new_index_do);

  _appo_Ne_col_vec = VectorXcd::Zero(_N_e_do);  
  _appo_Ne_col_vec[m_index_do] = complex<double>(1.,0.);

  _appo_Ne_col_vec.noalias() += -_M_do.col(I_new_index_do);
  _appo_Ne_col_vec /= _M_do(m_index_do,I_new_index_do); 

  _W_inv += _appo_Ne_col_vec*_W_inv.row(m_index_do);

  _M_do  += _appo_Ne_col_vec*_M_do.row(m_index_do);

  return;
}


void Walker :: initRand(int rank){
   //Initializing random number generator, one for each node
   ///////////////////////////////////////
   int seed[4];
   int p1, p2; 
   ifstream Primes("Primes");
   if (Primes.is_open()){
     for(int i=0;i<=rank;i++){             
       Primes >> p1 >> p2 ;
     }   
   } else cerr << "PROBLEM: Unable to open Primes" << endl;
   Primes.close();

   ifstream input("seed.in");
   string property;
   if (input.is_open()){
     while ( !input.eof() ){
       input >> property;
       if( property == "RANDOMSEED" ){
	 input >> seed[0] >> seed[1] >> seed[2] >> seed[3];
	 _rnd.SetRandom(seed,p1,p2);
       }   
     }   
     input.close();
   } else cerr << "PROBLEM: Unable to open seed.in" << endl;
   //////////////////////////////////////

   return;
}


////////////////////////
// Metropolis methods //
////////////////////////

bool Walker :: Metropolis_step_NN_hop(){

  // The following line is useful when testing the fast-update scheme;
  // WARNING: If uncommented, it makes the fast-update efficiency completely useless
  //restart_W_inv_from_scratch_and_check_it(); 

  _tot_hopping += 1;
  _where_to_move = _rnd.RandInt(0,_imulti[0]);
  _l_index = _rnd.RandInt(0,_N_e);
  _K_index = _occupied_sites[_l_index];
  _K_new_index = _ivic(_K_index%L,_where_to_move,0)+L*((int)(_K_index/L));

  // t-J model, no double occupations!!!
  if (_kel[_K_new_index%L]!=0 or _kel[L+(_K_new_index%L)]!=0){
    return false; //The move is not possible
  }
  else{

    if(_l_index < _N_e_up){

      // Compute ratio from fast update and decide to accept or not the hopping
      _ratio_W_inv = _M_up(_K_new_index,_l_index);

      if ( _rnd.Rannyu() <= real(_ratio_W_inv*conj(_ratio_W_inv)) ){
        _count_accept_hopping += 1;

        //Updating W_inv matrices
        update_W_inv_up_electron_hopping(_l_index, _K_new_index);

        //Accept the hopping
        for(int j=0;j<_N_e_do;j++){
          _f_mat_Ri_Rj(_l_index,j) = _f_ij_mat(_K_new_index,_occupied_sites_do[j]);
        }    
        for(int J=0;J<L;J++){
          _f_mat_Ri_J(_l_index,J) = _f_ij_mat(_K_new_index,J);
        }

        _occupied_sites[_l_index]    = _K_new_index;
        _occupied_sites_up[_l_index] = _K_new_index;
        _iconf[_K_new_index]         = _iconf[_K_index]; 
        _iconf[_K_index]             = 0; 
        _kel[_K_new_index]           = _kel[_K_index]; 
        _kel[_K_index]               = 0; 
        
        return true;
      }
      else{
        // Refuse the hopping move
        return false;
      }
    }
    else{

      int m_index_do     = _l_index - _N_e_up;
      //int I_index_do     = _K_index - L;
      int I_new_index_do = _K_new_index - L;

      // Compute ratio from fast update and decide to accept or not the hopping
      _ratio_W_inv = _M_do(m_index_do,I_new_index_do); 

      if ( _rnd.Rannyu() <= real(_ratio_W_inv*conj(_ratio_W_inv)) ){
        _count_accept_hopping += 1;

        //Updating W_inv matrices
        update_W_inv_do_electron_hopping(m_index_do, I_new_index_do);

        // Accept the hopping
        for(int i=0;i<_N_e_up;i++){
          _f_mat_Ri_Rj(i,m_index_do) = _f_ij_mat(_occupied_sites_up[i],I_new_index_do);
        }
        for(int I=0;I<L;I++){
          _f_mat_I_Rj(I,m_index_do) = _f_ij_mat(I,I_new_index_do);
        }

        _occupied_sites[_l_index]      = _K_new_index;
        _occupied_sites_do[m_index_do] = I_new_index_do;
        _iconf[_K_new_index]           = _iconf[_K_index]; 
        _iconf[_K_index]               = 0; 
        _kel[_K_new_index]             = _kel[_K_index]; 
        _kel[_K_index]                 = 0; 

        return true;
      }
      else{
        // Refuse the hopping move
        return false;
      }
    }
  }
}

      
bool Walker :: Metropolis_step_NN_SpinFlip(){

  // The following line is useful when testing the fast-update scheme;
  // WARNING: If uncommented, it makes the fast-update efficiency completely useless
  //restart_W_inv_from_scratch_and_check_it(); 

  _tot_SpinFlip += 1;
  _where_to_move = _rnd.RandInt(0,_imulti[0]);
  _l_index     = _rnd.RandInt(0,_N_e);
  _K_index     = _occupied_sites[_l_index];
  _K_new_index = _ivic(_K_index%L,_where_to_move,0)+L*((int)(_K_index/L));

  _I_index     = (_K_new_index%L) + L*(1-(int)(_K_index/L));  
  _I_new_index = (_K_index%L) + L*(1-(int)(_K_new_index/L));  
  _m_index     = _kel[_I_index]-1;

  int l_index_up =           _l_index*(1-(int)(_K_index/L)) +           _m_index*((int)(_K_index/L));
  int m_index_do = (_m_index-_N_e_up)*(1-(int)(_K_index/L)) + (_l_index-_N_e_up)*((int)(_K_index/L));

  int K_index_up =     _K_index*(1-(int)(_K_index/L)) +     _I_index*((int)(_K_index/L));
  int I_index_do = (_I_index-L)*(1-(int)(_K_index/L)) + (_K_index-L)*((int)(_K_index/L));

  int K_new_index_up =     _K_new_index*(1-(int)(_K_index/L)) +     _I_new_index*((int)(_K_index/L));
  int I_new_index_do = (_I_new_index-L)*(1-(int)(_K_index/L)) + (_K_new_index-L)*((int)(_K_index/L));

  if (_kel[_K_new_index]!=0 or _kel[_I_new_index]!=0 or _m_index==-1){
    return false; //The move is not possible
  }
  else{

    // Compute ratio from fast update
    _C_mat(0,0) = _M_up(K_new_index_up,l_index_up) + _W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) );

    _C_mat(1,1) = _M_do(m_index_do,I_new_index_do) + _W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_index_up,I_index_do) - _f_ij_mat(K_index_up,I_new_index_do) ); 

    _C_mat(1,0) = _W_inv(m_index_do,l_index_up);

    _C_mat(0,1) = complex<double>(0.,0.);
    for(int j=0;j<_N_e_do;j++){
      _C_mat(0,1) += ( _f_mat_I_Rj(K_new_index_up,j) - _f_mat_I_Rj(K_index_up,j) )*( _M_do(j,I_new_index_do) - _M_do(j,I_index_do)  );
    }

    _C_mat(0,1) += -( _M_up(K_new_index_up,l_index_up) -1. )*( _f_ij_mat(K_index_up,I_new_index_do) - _f_ij_mat(K_index_up,I_index_do) );
    _C_mat(0,1) += ( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) )*( _M_do(m_index_do,I_new_index_do) - 1. );
    _C_mat(0,1) += -( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) )*_W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_index_up,I_new_index_do) - _f_ij_mat(K_index_up,I_index_do) );

    _ratio_W_inv = _C_mat(0,0)*_C_mat(1,1)-_C_mat(0,1)*_C_mat(1,0);

    if ( _rnd.Rannyu() <= real(_ratio_W_inv*conj(_ratio_W_inv)) ){
      // The move is accepted
      _count_accept_SpinFlip += 1;

      // Updating W_inv matrices
      // I want to update by performing two subsequent single-electron hopping
      // I could perform first the up-electron hopping and then the down-electron hopping
      // It may happen that the first hopping leads to a zero amplitude
      // In that case, I first move the down-electron and then the up-electron
      if( abs( _M_up(K_new_index_up,l_index_up) ) > 1e-10 ){
        update_W_inv_up_electron_hopping(l_index_up, K_new_index_up);
        update_W_inv_do_electron_hopping(m_index_do, I_new_index_do);
      }
      else{
        if( abs( _M_do(m_index_do,I_new_index_do) ) > 1e-10 ){
          update_W_inv_do_electron_hopping(m_index_do, I_new_index_do);
          update_W_inv_up_electron_hopping(l_index_up, K_new_index_up);
        }
        else{
          // Both amplitudes are 0
          // The fast-update scheme breaks down
          // I have to perform the spin flip and compute W_inv from scratch

          cout<<endl<<"ACHTUNG: an accepted Spin Flip move can not be performed using fast update."<<endl;
          cout<<"The code will perform the Spin Flip and compute W_inv from scratch."<<endl<<endl;

          _occupied_sites_up[l_index_up] = K_new_index_up;
          _occupied_sites_do[m_index_do] = I_new_index_do;	
          for(int j=0;j<_N_e_do;j++){							  
            _f_mat_Ri_Rj(l_index_up,j) = _f_ij_mat(K_new_index_up,_occupied_sites_do[j]);
          }    
          for(int i=0;i<_N_e_up;i++){	
            _f_mat_Ri_Rj(i,m_index_do) = _f_ij_mat(_occupied_sites_up[i],I_new_index_do);
          }    
          for(int J=0;J<L;J++){
            _f_mat_Ri_J(l_index_up,J) = _f_ij_mat(K_new_index_up,J);
            _f_mat_I_Rj(J,m_index_do) = _f_ij_mat(J,I_new_index_do);
          }    
                                  
          _occupied_sites[_l_index] = _K_new_index;
          _occupied_sites[_m_index] = _I_new_index;
          _iconf[_K_new_index]      = _iconf[_K_index]; 
          _iconf[_K_index]          = 0; 
          _iconf[_I_new_index]      = _iconf[_I_index]; 
          _iconf[_I_index]          = 0; 
          _kel[_K_new_index]        = _kel[_K_index]; 
          _kel[_K_index]            = 0; 
          _kel[_I_new_index]        = _kel[_I_index]; 
          _kel[_I_index]            = 0; 
        
          restart_W_inv_from_scratch();

          // It must return here, the move is accepted
          return true;
        }
      }

      // The fast-update scheme works properly

      // Update the f_mat_Ri_Rj matrix
      _occupied_sites_up[l_index_up] = K_new_index_up;
      _occupied_sites_do[m_index_do] = I_new_index_do;

      for(int j=0;j<_N_e_do;j++){
	      _f_mat_Ri_Rj(l_index_up,j) = _f_ij_mat(K_new_index_up,_occupied_sites_do[j]);
      }
      for(int i=0;i<_N_e_up;i++){
	      _f_mat_Ri_Rj(i,m_index_do) = _f_ij_mat(_occupied_sites_up[i],I_new_index_do);
      }    
      for(int J=0;J<L;J++){
        _f_mat_Ri_J(l_index_up,J) = _f_ij_mat(K_new_index_up,J);
        _f_mat_I_Rj(J,m_index_do) = _f_ij_mat(J,I_new_index_do);
      }

      _occupied_sites[_l_index] = _K_new_index;
      _occupied_sites[_m_index] = _I_new_index;
      _iconf[_K_new_index]      = _iconf[_K_index]; 
      _iconf[_K_index]          = 0; 
      _iconf[_I_new_index]      = _iconf[_I_index]; 
      _iconf[_I_index]          = 0; 
      _kel[_K_new_index]        = _kel[_K_index]; 
      _kel[_K_index]            = 0; 
      _kel[_I_new_index]        = _kel[_I_index]; 
      _kel[_I_index]            = 0; 

      return true;
    }
    else{
      // Refuse the spin flip 
      return false;
    }
  }
}


bool Walker :: Metropolis_step(){

  // Choose between the two possible moves
  if(_rnd.Rannyu() < _P_SpinFlip){
    return Metropolis_step_NN_SpinFlip(); 
  }
  else {
    return Metropolis_step_NN_hop();
  }
}

double Walker :: U_Hubb_energy(){
  double U_local_energy = 0.;
  // U_hub, diagonal part of local energy
  for(int site=0;site<L;site++){
    if (_kel[site]!=0 and _kel[site+L]!=0 ){
      U_local_energy += _U_hubb;
    }
  }   
  return U_local_energy;
}

complex<double> Walker :: t_energy(){
  // t_hub, hopping part of local energy
  complex<double> t_local_energy = 0.;

  // Moving only spin-up electrons
  for(_l_index=0;_l_index<_N_e_up;_l_index++){
    _K_index = _occupied_sites[_l_index];

    for(int NN_move=0;NN_move<_imulti[0];NN_move++){
      _K_new_index=_ivic(_K_index%L,NN_move,0); 

      // n_i n_j term for the t-J model (included only here, not in the spin-down part)
      t_local_energy += -0.25*_J*(_iconf[_K_index%L]-_iconf[(_K_index%L)+L])*(_iconf[_K_new_index%L]-_iconf[(_K_new_index%L)+L]);

      if (_kel[_K_new_index]==0 and _kel[_K_new_index+L]==0){

        // Computing virtual hopping contribution using the fast-update
        _ratio_W_inv = _M_up(_K_new_index,_l_index);

        t_local_energy += _t_hubb*_ratio_W_inv;
      }
    }
  }

  // Moving only spin-down electrons
  for(_l_index=_N_e_up;_l_index<_N_e;_l_index++){

    _K_index = _occupied_sites[_l_index];
    int m_index_do     = _l_index - _N_e_up;
    //int I_index_do     = _K_index - L;

    for(int NN_move=0;NN_move<_imulti[0];NN_move++){
      _K_new_index       = _ivic(_K_index%L,NN_move,0) + L; 
      int I_new_index_do = _K_new_index - L;

      if (_kel[_K_new_index%L]==0 and _kel[(_K_new_index%L)+L]==0){

        // Computing virtual hopping contribution using the fast-update 
        _ratio_W_inv = _M_do(m_index_do,I_new_index_do);
        
        t_local_energy += _t_hubb*_ratio_W_inv;
      }
    }
  }

  return t_local_energy;
}

complex<double> Walker :: J_energy(){
  // J, spin-flip part of local energy
  complex<double> J_local_energy = 0. + 0i;

  for(_l_index=0;_l_index<_N_e_up;_l_index++){
    _K_index = _occupied_sites[_l_index];

    for(int NN_move=0;NN_move<_imulti[0];NN_move++){
      _K_new_index=_ivic(_K_index%L,NN_move,0); 

      _I_index     = _K_new_index + L;
      _I_new_index = _K_index     + L;
      _m_index     = _kel[_I_index]-1;

      int l_index_up = _l_index;
      int m_index_do = _m_index-_N_e_up;

      int K_index_up = _K_index;
      int I_index_do = _I_index-L;

      int K_new_index_up = _K_new_index;
      int I_new_index_do = _I_new_index-L;

      //Diagonal part Sz_i-Sz_j
      J_local_energy += 0.25*_J*(_iconf[_K_index%L]+_iconf[(_K_index%L)+L])*(_iconf[_I_index%L]+_iconf[(_I_index%L)+L]);

      if (_kel[_K_new_index]==0 and _kel[_I_new_index]==0 and _m_index!=-1){

        // Computing virtual spin-flip contribution using the fast-update 
        _C_mat(0,0) = _M_up(K_new_index_up,l_index_up) + _W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) );

        _C_mat(1,1) = _M_do(m_index_do,I_new_index_do) + _W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_index_up,I_index_do) - _f_ij_mat(K_index_up,I_new_index_do) );

        _C_mat(1,0) = _W_inv(m_index_do,l_index_up);

        _C_mat(0,1) = complex<double>(0.,0.);
        for(int j=0;j<_N_e_do;j++){
          _C_mat(0,1) += ( _f_mat_I_Rj(K_new_index_up,j) - _f_mat_I_Rj(K_index_up,j) )*( _M_do(j,I_new_index_do) - _M_do(j,I_index_do)  );
        }

        _C_mat(0,1) += -( _M_up(K_new_index_up,l_index_up) -1. )*( _f_ij_mat(K_index_up,I_new_index_do) - _f_ij_mat(K_index_up,I_index_do) );
        _C_mat(0,1) += ( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) )*( _M_do(m_index_do,I_new_index_do) - 1. );
        _C_mat(0,1) += -( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) )*_W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_index_up,I_new_index_do) - _f_ij_mat(K_index_up,I_index_do) );

        _ratio_W_inv = _C_mat(0,0)*_C_mat(1,1)-_C_mat(0,1)*_C_mat(1,0);

        J_local_energy += -0.5*_J*_ratio_W_inv;
      }
    }
  }
  return J_local_energy;
}

void Walker :: local_energy_Hubbard(){

  _local_energy = 0.;

  if(_actually_t_J==true){
    _local_energy += U_Hubb_energy();
    _local_energy += t_energy();
  }

  _local_energy += J_energy();

  return;
}

complex<double> Walker :: get_energy(){
   return _local_energy;
}

double Walker :: get_NN_hopping_acceptance(){
  return _count_accept_hopping/max(1.*_tot_hopping,1.);
}

double Walker :: get_NN_SpinFlip_acceptance(){
  return _count_accept_SpinFlip/max(1.*_tot_SpinFlip,1.);
}


//////////////////////////
// Measurements methods //
//////////////////////////

void Walker :: add_Si_Sj(MatrixXcd & Si_Sj){

  for(_l_index=0;_l_index<_N_e_up;_l_index++){
    _K_index = _occupied_sites_up[_l_index];

    for(int _K_new_index=0;_K_new_index<L;_K_new_index++){

      if (_K_new_index==_K_index){
        Si_Sj(_K_index,_K_new_index) += 0.75; // S*(S+1) = 3/4
        continue;
      }

      _I_index     = _K_new_index + L;
      _I_new_index = _K_index     + L;
      _m_index     = _kel[_I_index]-1;

      int l_index_up = _l_index;
      int m_index_do = _m_index-_N_e_up;

      int K_index_up = _K_index;
      int I_index_do = _I_index-L;

      int K_new_index_up = _K_new_index;
      int I_new_index_do = _I_new_index-L;

      //Diagonal part Sz_i-Sz_j
      Si_Sj(_K_index,_K_new_index) += 0.25*_J*(_iconf[_K_index%L]+_iconf[(_K_index%L)+L])*(_iconf[_I_index%L]+_iconf[(_I_index%L)+L]);
      Si_Sj(_K_new_index,_K_index) += 0.25*_J*(_iconf[_K_index%L]+_iconf[(_K_index%L)+L])*(_iconf[_I_index%L]+_iconf[(_I_index%L)+L]);

      if (_kel[_K_new_index]==0 and _kel[_I_new_index]==0 and _m_index!=-1){

        // Computing virtual spin-flip contribution using the fast-update 
        _C_mat(0,0) = _M_up(K_new_index_up,l_index_up) + _W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) );

        _C_mat(1,1) = _M_do(m_index_do,I_new_index_do) + _W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_index_up,I_index_do) - _f_ij_mat(K_index_up,I_new_index_do) );

        _C_mat(1,0) = _W_inv(m_index_do,l_index_up);

        _C_mat(0,1) = complex<double>(0.,0.);
        for(int j=0;j<_N_e_do;j++){
          _C_mat(0,1) += ( _f_mat_I_Rj(K_new_index_up,j) - _f_mat_I_Rj(K_index_up,j) )*( _M_do(j,I_new_index_do) - _M_do(j,I_index_do)  );
        }

        _C_mat(0,1) += -( _M_up(K_new_index_up,l_index_up) -1. )*( _f_ij_mat(K_index_up,I_new_index_do) - _f_ij_mat(K_index_up,I_index_do) );
        _C_mat(0,1) += ( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) )*( _M_do(m_index_do,I_new_index_do) - 1. );
        _C_mat(0,1) += -( _f_ij_mat(K_new_index_up,I_new_index_do) - _f_ij_mat(K_new_index_up,I_index_do) )*_W_inv(m_index_do,l_index_up)*( _f_ij_mat(K_index_up,I_new_index_do) - _f_ij_mat(K_index_up,I_index_do) );

        _ratio_W_inv = _C_mat(0,0)*_C_mat(1,1)-_C_mat(0,1)*_C_mat(1,0);

        Si_Sj(_K_index,_K_new_index) += -0.5*_J*_ratio_W_inv;
        Si_Sj(_K_new_index,_K_index) += -0.5*_J*_ratio_W_inv;
      }
    }
  }
  return;
}


void Walker :: print_iconf(){
  cout<<endl<<"kel"<<endl; 
  for(int i=0; i<L; i++){
    cout<<_kel[i]<<", ";
  }
  cout<<" | ";
  for(int i=L; i<two_L; i++){
    cout<<_kel[i]<<", ";
  }
  cout<<endl;
  cout<<endl<<"iconf"<<endl; 
  for(int i=0; i<L; i++){
    cout<<_iconf[i]<<", ";
  }
  cout<<" | ";
  for(int i=L; i<two_L; i++){
    cout<<_iconf[i]<<", ";
  }
  cout<<endl;
  cout<<endl<<"occupied_list"<<endl; 
  for(int i=0; i<_N_e; i++){
    cout<<_occupied_sites[i]<<", ";
  }
  cout<<endl<<endl;

  cout<<endl<<"Check kel"<<endl; 
  for(int i=0; i<L; i++){
    cout<<max(_kel[i],_kel[i+L])<<", ";
  }
  cout<<endl;
  cout<<endl<<"Check iconf"<<endl; 
  for(int i=0; i<L; i++){
    cout<<_iconf[i]+_iconf[i+L]<<", ";
  }
  cout<<endl;
  cout<<endl<<"Check occupied_list"<<endl; 
  for(int i=0; i<_N_e_up; i++){
    cout<<_occupied_sites[i]-_occupied_sites_up[i]<<", ";
  }
  for(int i=_N_e_up; i<_N_e_do; i++){
    cout<<_occupied_sites[i]-_occupied_sites_do[i-_N_e_up]-_N_e_up<<", ";
  }
  cout<<endl<<endl;

  return;
}


void Walker :: write_info_output_file(std::ofstream& info_out_file ) const {

  info_out_file << "Simulating t-J-U model on Pyrocglore Lattice\n";
  info_out_file << "L = " << L << std::endl;
  info_out_file << "N_up = " << _N_e_up << ", N_do = " << _N_e_do << std::endl;
  info_out_file << "\nPhysical parameters: \n";
  info_out_file << "t = " << _t_hubb << std::endl;
  info_out_file << "U = " << _U_hubb << std::endl;
  info_out_file << "J = " << _J << std::endl;
  info_out_file << "\n\n---------------------------\n\nVariational parameters\n";

  info_out_file << "# abs_f_ij_NN: "<<L*L<< std::endl;
  info_out_file << "# phase_f_ij_NN: "<<L*L<< std::endl;

  info_out_file << "\n\n\n";
  return;
};


void Walker :: get_kel(ArrayXi & kel_init){
    for (int ind_i = 0; ind_i < two_L; ind_i++) {
      kel_init[ind_i] = _kel[ind_i];
    }
    return;
}


///////////////////////////////////////////////////////////////
// Amplitude of wavefunction !!!                             //
// Derivatives wrt variational parameters                    //
// (using Adjoint/Backward Algorithmic Differentiation) !!!  //
///////////////////////////////////////////////////////////////

// Derivatives with respect to the variational parameters: top functions

int Walker :: compute_O_x_ALL_parameters(VectorXcd & O_x_in){
  
  O_x_in *= 0.;
  int index_starting = 0;
  index_starting = compute_O_x_f_ij(O_x_in, index_starting);
  index_starting = compute_log_amp_NO_det_adjoint(O_x_in, index_starting);

  return index_starting;
}

// ###
// ### Amplitudes and derivatives for NOT-f_ij part 
// ###

// Compute the amplitude of the wave function, EXCLUDING the determinant

complex<double> Walker :: compute_log_amp_NO_det(){
  complex<double> log_amp = 0. + 0i;
  return log_amp;
}

// Derivatives with respect to the parameters of the wave function, EXCLUDING the determinant
// O_x_in represents a vector containing parameters_adjoints

int Walker :: compute_log_amp_NO_det_adjoint(VectorXcd & O_x_in, int index_starting){
  return index_starting; 
}


// ###
// ### Amplitudes and derivatives for f_ij part 
// ###

int Walker :: compute_O_x_f_ij(VectorXcd & O_x_in, int index_starting){

  _appo_f_ij_mat.setZero();
  for(int i=0;i<_N_e_up;i++){
    for(int j=0;j<_N_e_do;j++){
	    _appo_f_ij_mat(_occupied_sites_up[i],_occupied_sites_do[j]) = _W_inv(j,i);
    }
  }

  index_starting = build_f_ij_mat_adjoint(_appo_f_ij_mat,O_x_in,index_starting);

  return index_starting;
}

void Walker ::  build_f_ij_mat(MatrixXcd & f_ij_matrix){

  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      f_ij_matrix(i,j) = _abs_f_ij(i,j)*exp( 1i*_phase_f_ij(i,j) );
    }
  }

  return;
}

int Walker :: build_f_ij_mat_adjoint(MatrixXcd & f_ij_matrix_adjoint, VectorXcd & O_x_in, int index_starting){

  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      O_x_in[index_starting] = f_ij_matrix_adjoint(i,j)*exp( 1i*_phase_f_ij(i,j) ); 
      index_starting += 1;
    }
  }
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {
      O_x_in[index_starting] = f_ij_matrix_adjoint(i,j)* 1i * _f_ij_mat(i,j); 
      index_starting += 1;
    }
  }

  return index_starting;
}


// Checking with finite difference that the derivatives are correct

void Walker :: DEBUG_compute_O_x_f_ij(){

  restart_W_inv_from_scratch();

  cout<<endl<<"Checking logarithmic derivatives with respect to variational parameters using finite differences!!!"<<endl<<endl;

  VectorXcd O_x_f_ij_test = VectorXcd::Zero(2*L*L);

  compute_O_x_f_ij(O_x_f_ij_test, 0);

  complex<double> log_det_amp = log((_f_mat_Ri_Rj).fullPivLu().determinant());

  int count_OP = 0;
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {

      _abs_f_ij(i,j) += 0.000001;

      MatrixXcd f_ij_matrix_CHECK = MatrixXcd::Zero(L,L);
      build_f_ij_mat(f_ij_matrix_CHECK);

      MatrixXcd f_mat_Ri_Rj_CHECK = MatrixXcd::Zero(_N_e_up,_N_e_do);
      for(int i=0;i<_N_e_up;i++){
        for(int j=0;j<_N_e_do;j++){
          f_mat_Ri_Rj_CHECK(i,j) = f_ij_matrix_CHECK(_occupied_sites_up[i],_occupied_sites_do[j]);
        }
      }

      complex<double> log_det_amp_CHECK = log((f_mat_Ri_Rj_CHECK).fullPivLu().determinant());

      cout<<"Ox abs_f_ij_bonds: "<<O_x_f_ij_test[count_OP]<<"  "<<(log_det_amp_CHECK-log_det_amp)/0.000001<<endl;

      _abs_f_ij(i,j) += -0.000001;
      count_OP += 1;
    }   
  }
  for (int i = 0; i < L; ++i) {
    for (int j = 0; j < L; ++j) {

      _phase_f_ij(i,j) += 0.000001;

      MatrixXcd f_ij_matrix_CHECK = MatrixXcd::Zero(L,L);
      build_f_ij_mat(f_ij_matrix_CHECK);

      MatrixXcd f_mat_Ri_Rj_CHECK = MatrixXcd::Zero(_N_e_up,_N_e_do);
      for(int i=0;i<_N_e_up;i++){
        for(int j=0;j<_N_e_do;j++){
          f_mat_Ri_Rj_CHECK(i,j) = f_ij_matrix_CHECK(_occupied_sites_up[i],_occupied_sites_do[j]);
        }
      }

      complex<double> log_det_amp_CHECK = log((f_mat_Ri_Rj_CHECK).fullPivLu().determinant());

      cout<<"Ox phase_f_ij_bonds: "<<O_x_f_ij_test[count_OP]<<"  "<<(log_det_amp_CHECK-log_det_amp)/0.000001<<endl;

      _phase_f_ij(i,j) += -0.000001;
      count_OP += 1;
    }   
  }   

  if(count_OP != O_x_f_ij_test.size() ){
    cout<<"DEBUG_compute_O_x_f_ij_test function!!!"<<endl<<"There may be a problem with the number of parameters of the f_ij!"<<endl<<endl;
  }
  cout<<endl;

  return;
}


void Walker :: DEBUG_compute_O_x_NO_det(){
  return;
}


Walker :: Walker(){}

Walker :: ~Walker(){
}



/******************** ALL RIGHTS RESERVED ***********************
*****************************************************************
    _/_/_/    _/    _/_/_/   _/_/_/   _/_/_/  Scuola 
   _/        _/    _/       _/       _/  _/   Internazionale
  _/_/_/    _/    _/_/_/   _/_/_/   _/_/_/    Superiore di
     _/    _/        _/       _/   _/  _/     Studi 
_/_/_/    _/    _/_/_/   _/_/_/   _/  _/      Avanzati 
*****************************************************************
*****************************************************************/
