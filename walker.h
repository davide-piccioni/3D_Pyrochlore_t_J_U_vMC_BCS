/******************** ALL RIGHTS RESERVED ***********************
*****************************************************************
    _/_/_/    _/    _/_/_/   _/_/_/   _/_/_/  Scuola 
   _/        _/    _/       _/       _/  _/   Internazionale
  _/_/_/    _/    _/_/_/   _/_/_/   _/_/_/    Superiore di
     _/    _/        _/       _/   _/  _/     Studi 
_/_/_/    _/    _/_/_/   _/_/_/   _/  _/      Avanzati 
*****************************************************************
*****************************************************************/

#ifndef __Walker__
#define __Walker__


using namespace std;
using Eigen::VectorXcd;
using Eigen::RowVectorXcd;
using Eigen::VectorXd;
using Eigen::MatrixXcd;
using Eigen::MatrixXd;
using Eigen::ArrayXi;
using Eigen::ArrayXXi;


class Walker {

private:

  //Distances and ivic
  ArrayXi _imulti; // How many neighbors at each distance
  Eigen::Tensor<int, 3> _ivic; // Table of distances and directions

  //Random number generator
  Random _rnd;

  //Physical parameters
  double _t_hubb;
  double _J;
  double _U_hubb;

  //State of the system
  int _N_e_up; // Number of up electrons
  int _N_e_do; // Number of down electrons
  int _N_e; // Total number of electrons

  ArrayXi _iconf; // Configuration of the system, 0 for empty, 1 for up, -1 for down
  ArrayXi _kel; // Configuration of the system, including the order with which construction operators act on vacuum
  ArrayXi _occupied_sites; // Sites occupied by electrons, ordered as their index in kel
  ArrayXi _occupied_sites_up; // Sites occupied by up electrons
  ArrayXi _occupied_sites_do; // Sites occupied by down electrons
  bool _actually_t_J; // true if t-J-U model, false if Heisenberg model

  complex<double> _local_energy; // Local energy of the system

  ////////////////////////////
  // Variational parameters //
  ////////////////////////////

  // f_ij 
  MatrixXd _abs_f_ij; // Amplitude of the f_ij
  MatrixXd _phase_f_ij; // Phase of the f_ij

  ///////////////////////////
  // Addiational variables //
  ///////////////////////////

  complex<double> _log_amp_NO_det, _log_amp_NO_det_new; //If you may want to add a non Slater determinant part to the wavefunction

  //Matrices for slater determinant
  MatrixXcd _f_ij_mat; // Matrix containing the f_ij
  MatrixXcd _f_mat_Ri_Rj; // Matrix containing the f_ij for the occupied sites

  //Matrices useful for fast-update and for derivatives
  MatrixXcd _f_mat_I_Rj;
  MatrixXcd _f_mat_Ri_J;
  MatrixXcd _W_inv; // Inverse of the _f_mat_Ri_Rj matrix
  MatrixXcd _M_up;  // Product _f_mat_I_Rj * _W_inv
  MatrixXcd _M_do;  // Product _W_inv * _f_mat_Ri_J
  MatrixXcd _appo_f_ij_mat; // Auxiliary matrix
  MatrixXcd _appo_W_inv; // Auxiliary matrix
  MatrixXcd _C_mat; // Auxiliary 2x2 matrix 
  complex<double> _ratio_W_inv; // Ratio of the determinants of successive configurations given by the fast-update scheme
  RowVectorXcd _appo_Ne_row_vec; // Auxiliary vector
  VectorXcd _appo_Ne_col_vec; // Auxiliary vector

  // Variables to perform hopping and spin flip with Metropolis
  int _where_to_move; // Random number for the direction of the hopping
  int _l_index; // Random number for the index of the electron to be moved
  int _K_index; // Actual position of the electron to be moved
  int _K_new_index; // New proposed position of the electron to be moved
  int _m_index; // Random number for the index of the other electron to be moved
  int _I_index; // Actual position of the other electron to be moved
  int _I_new_index; // New proposed position of the other electron to be moved

  int _count_accept_hopping; // Number of accepted hopping moves
  int _tot_hopping; // Total number of hopping moves
  int _count_accept_SpinFlip; // Number of accepted spin-flip moves
  int _tot_SpinFlip; // Total number of spin-flip moves
  double _P_SpinFlip; // Probability of performing a spin-flip move

protected:

public:
  // constructors
  Walker();
  // destructor
  ~Walker();
  // methods

  //////////////////////////
  // Initializing methods //
  //////////////////////////
  void initialize_lattice(ArrayXi & imulti, Eigen::Tensor<int, 3> & ivic);
  void initialize_physical_parameters(int N_e_up, int N_e_do, double P_SpinFlip, double t_hubb, double J, double U_hubb); 

  void initialize_variational_parameters(MatrixXd & abs_f_ij, MatrixXd & phase_f_ij );  
  void initRand(int rank);

  /////////////////////////
  // Fast update methods //
  /////////////////////////

  // During the fast update, W_inv gets degraded by the accumulating of numerical errors
  // This function is used to restart W_inv from scratch, and check the fast update scheme
  void restart_W_inv_from_scratch(); // Restart W_inv from scratch

  
  void update_W_inv_up_electron_hopping(int l_index_up, int K_new_index_up);  // Update W_inv after an up-electron hopping
  void update_W_inv_do_electron_hopping(int m_index_do, int I_new_index_do);  // Update W_inv after a down-electron hopping

  void restart_W_inv_from_scratch_and_check_it();
  
  ////////////////////////
  // Metropolis methods //
  ////////////////////////

  bool Metropolis_step_NN_hop(); // Hopping move
  bool Metropolis_step_NN_SpinFlip(); // Spin flip move
  bool Metropolis_step(); // Metropolis step, choose between hopping and spin flip
  double get_NN_hopping_acceptance(); // Get the acceptance ratio of the hopping moves
  double get_NN_SpinFlip_acceptance(); // Get the acceptance ratio of the spin-flip moves

  double U_Hubb_energy(); // U_hub, diagonal part of local energy
  complex<double> t_energy(); // t_hub, hopping part of local energy
  complex<double> J_energy(); // J_hub, Spin-flip part of local energy
  void local_energy_Hubbard(); // Compute the local energy of the system
  complex<double> get_energy(); // Get the local energy of the system

  //////////////////////////
  // Measurements methods //
  //////////////////////////

  void add_Si_Sj(MatrixXcd & Si_Sj); //Measurement of spin-spin correlations 
  void write_info_output_file(ofstream& info_out_file) const;

  void get_kel(ArrayXi & kel_init); // For debug
  void print_iconf(); // For debug

  ///////////////////////////////////////////////////////////////
  // Amplitude of wavefunction and                             //
  // Logarithmic derivatives of the wave function
  // with respect to the variational parameters                //
  // (using Adjoint/Backward Algorithmic Differentiation) !!!  //
  ///////////////////////////////////////////////////////////////

  // With the notation "O_x" we mean the logarithmic derivative of the wave function with respect to the variational parameters
  
  // O_x_in represents a vector containing parameters_adjoints

  // Derivatives with respect to the variational parameters: top functions
  int compute_O_x_ALL_parameters(VectorXcd & O_x_in);
  int compute_O_x_f_ij(VectorXcd & O_x_in, int index_starting);

  // Checking with finite difference methods
  void DEBUG_compute_O_x_f_ij();
  void DEBUG_compute_O_x_NO_det();

  // ###
  // ### Amplitudes and derivatives for NOT-Slater determinant part 
  // ###

  // Compute the amplitude of the wave function, EXCLUDING the determinant
  complex<double> compute_log_amp_NO_det();

  // Derivatives with respect to the parameters of the wave function, EXCLUDING the determinant

  int compute_log_amp_NO_det_adjoint(VectorXcd & O_x_in, int index_starting);

  // ###
  // ### Amplitudes and derivatives for f_ij part 
  // ###

  // Building auxiliary matrix for Slater determinant
  void build_f_ij_mat(MatrixXcd & f_ij_matrix); //2

  // Adjoints of previous functions to perform Adjoint (backward) Automatic Differentiation
  
  int build_f_ij_mat_adjoint(MatrixXcd & f_ij_matrix_adjoint, VectorXcd & O_x_in, int index_starting); //2 adjoint

};



#endif // __Walker__

/******************** ALL RIGHTS RESERVED ***********************
*****************************************************************
    _/_/_/    _/    _/_/_/   _/_/_/   _/_/_/  Scuola 
   _/        _/    _/       _/       _/  _/   Internazionale
  _/_/_/    _/    _/_/_/   _/_/_/   _/_/_/    Superiore di
     _/    _/        _/       _/   _/  _/     Studi 
_/_/_/    _/    _/_/_/   _/_/_/   _/  _/      Avanzati 
*****************************************************************
*****************************************************************/
