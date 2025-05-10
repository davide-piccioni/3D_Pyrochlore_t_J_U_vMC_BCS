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
#include <string>
#include <vector>
#include <array>
#include <cmath>
#include <complex>
#include <iomanip>  // std::setprecision()
#include <sys/stat.h>
#include <unistd.h>
#include <Eigen>
//#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/CXX11/Tensor>
#include <mpi.h>

#include "distances.h"
#include "random.h"
#include "walker.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using Eigen::VectorXcd;
using Eigen::MatrixXcd;
using Eigen::ArrayXi;
using Eigen::SelfAdjointEigenSolver;
using namespace std;


int read_ivic(ArrayXi & imulti, Eigen::Tensor<int, 3> & ivic);

void read_f_ij_params(const std::string& input_folder, MatrixXd & f_ij_mat);

int print_and_change_f_ij_params(ofstream & file_out_params, MatrixXd & f_ij_mat, int conta_pars, VectorXd & d_alpha, double gamma, double const_mult = 1. ); 



int main (int argc, char *argv[]){

   MPI_Init(&argc,&argv);
   int numprocs, rank;
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs); 
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);

   const clock_t begin_time = clock();

   if(rank==0){
     cout<<"Starting"<<endl<<endl;
     cout<<setprecision(15);
   }

   ///////////////////////////
   // Defining input folder //
   ///////////////////////////
   
   string input_folder = "input";

   if (argc > 1) {
     string folder_suffix = argv[1];
     string try_input_folder = "input_" + folder_suffix;
     struct stat info;
     if (stat(try_input_folder.c_str(), &info) == 0 && (info.st_mode & S_IFDIR)) {
       // Folder exists, you can use the input_folder variable here
       input_folder = try_input_folder;
       std::cout << "Input folder exists: " << input_folder << std::endl<<std::endl;
     }
     else{
       std::cout << "Input folder does not exists! "<<endl<<"The folder <<"<< input_folder <<" will be used as input."<< std::endl<<std::endl;
     }
   }

   /////////////////////
   // Reading lattice //
   /////////////////////

   ArrayXi imulti(N_distances);
   Eigen::Tensor<int, 3> ivic(L, Imaxmulti, N_distances);
   
   if(rank==0){
     cout<<"Reading distances (ivic)"<<endl<<endl;

     int check_read_ivic = read_ivic(imulti, ivic);

     if( check_read_ivic != 0 ){
       return 1;
     }
   }

   MPI_Bcast(imulti.data(),N_distances,MPI_INT,0,MPI_COMM_WORLD);
   MPI_Bcast(ivic.data(),L*Imaxmulti*N_distances,MPI_INT,0,MPI_COMM_WORLD);

   ///////////////////////////////////
   // Reading simulation parameters //
   ///////////////////////////////////

   if(rank==0){
     cout<<"Reading inputs"<<endl<<endl;
   }

   // Optimization parameters 
   bool starting_from_previous_params;
   int N, M, nsparse_ave, N_therm, N_scra, n_iterations_SR;
   double gamma, epsilon_SR;

   // Physical and simulation parameters to be provided to the wave funtion
   int N_e_up, N_e_do;
   double t_hubb, J, U_hubb;
   double P_SpinFlip ;

   // Give to the program: 
   //
   // N_e_up (# of electrons with spin up), 
   // N_e_do (# of electrons with spin down), 
   // t_hubb (t - hopping parameter),
   // J (exchange term J),
   // U_hubb (U - Hubbard interaction parameter),
   // P_SpinFlip (The probability of choosing to try to perform a hopping is 1. - P_SPIN_FLIP),
   //
   // starting_from_previous_params (if true, try starting the simulation from the last saved parameters)
   // N (# of total steps in unit of nsparse_ave for each walker), 
   // M (# of walkers), 
   // nsparse_ave (number of tried Metropolis steps between two measurements), 
   // N_therm (# of termalization steps in unit of nsparse_ave for each walker), 
   // N_scra (# of steps in unit of nsparse_ave after which the fast-update matrices are recomputed from scratch)
   // gamma (learning rate), 
   // n_iterations_SR (iterations of Stochastic reconfigurations);
   // epsilon_SR (regularization of S-Matrix)

   ifstream file_in(input_folder+"/input.dat");
   file_in>>N_e_up>>N_e_do>>t_hubb>>J>>U_hubb>>P_SpinFlip>>starting_from_previous_params>>N>>M>>nsparse_ave>>N_therm>>N_scra>>gamma>>n_iterations_SR>>epsilon_SR;
   file_in.close();

   if(M%numprocs==0){
     M = M / numprocs;
   }
   else{
     M = 1 + (M / numprocs); 
   }

   ////////////////////////////////////
   // Reading variational parameters //
   ////////////////////////////////////

   // abs of f_ij amplitudes
   MatrixXd abs_f_ij(L, L);
   read_f_ij_params(input_folder + "/input_abs_f_ij.dat", abs_f_ij ); 
   cout<<abs_f_ij(0,0)*0; //DEBUG

   // phase of f_ij amplitudes
   MatrixXd phase_f_ij(L, L);
   read_f_ij_params(input_folder + "/input_phase_f_ij.dat", phase_f_ij ); 
   cout<<phase_f_ij(0,0)*0; //DEBUG

   /////////////////////////////////////////////////////
   // Restarting from previous parameters if asked to //
   /////////////////////////////////////////////////////

   int SR_parameters = 2*L*L;

   if(starting_from_previous_params==true){ // Try restarting from the last saved parameters

     VectorXd appo_params = VectorXd::Zero(SR_parameters);
     std::ifstream file_in_restart(input_folder+"/restarting_pars.dat");

     int conta_pars=0;
     if (file_in_restart.is_open()) {
       while ( file_in_restart.good() and conta_pars<SR_parameters ){
        file_in_restart>>appo_params[conta_pars];
        conta_pars += 1;
       }
       file_in_restart.close();
     }

     if (conta_pars==SR_parameters) {

       ofstream garbage_file("./.garbage.txt");

       conta_pars = 0;
       conta_pars = print_and_change_f_ij_params(garbage_file, abs_f_ij, conta_pars, appo_params, 1., 0.);
       conta_pars = print_and_change_f_ij_params(garbage_file, phase_f_ij, conta_pars, appo_params, 1., 0.);

       garbage_file.close();

     }
     else{
       cout<<endl<<"There may be a problem with the number of parameters you want to restart from... The code is starting from the input ones..."<<endl<<endl;
     }

   }

   MPI_Barrier(MPI_COMM_WORLD);

   if(rank==0){
     cout<<endl<<endl<<"All input read!!!"<<endl<<endl;
   }

   /////////////////////////////
   // Starting the simulation //
   /////////////////////////////

   vector <Walker> wlkrs(M);
   for(int j=0; j<M; j++){
     (wlkrs[j]).initRand(rank*M+j);
     (wlkrs[j]).initialize_lattice(imulti, ivic);
     (wlkrs[j]).initialize_physical_parameters(N_e_up, N_e_do, P_SpinFlip, t_hubb, J, U_hubb );
   }

   // According to n_iterations_SR you will either perform the stochastic reconfiguration (and optimizate the variational parameters) 
   // or you will just measure the observables at fixed variational parameters

   if (n_iterations_SR<1){ //You will NOT perform SR!

     if(rank==0){
       cout<<"Computing mean energy and other observables at fixed variational parameters"<<endl<<endl;
     }

     for(int j=0; j<M; j++){
       (wlkrs[j]).initialize_variational_parameters(abs_f_ij, phase_f_ij ); 
     }

     if(rank==0){
       cout<<"Inizialized!!!"<<endl<<endl;
     } 
 
     //Thermalization
     int count_scra  = 0;
     for(int j=0; j<M; j++){
       for(int i=0; i<N_therm*nsparse_ave; i++){
        (wlkrs[j]).Metropolis_step();

        count_scra += 1;
        if(count_scra > N_scra*nsparse_ave){
          // Restart W_inv from scratch
          (wlkrs[j]).restart_W_inv_from_scratch();
          count_scra = 0;
        }

       }

       // Restart W_inv from scratch after the thermalization
       (wlkrs[j]).restart_W_inv_from_scratch();
       count_scra = 0;
     }

     if(rank==0){
       cout<<"Thermalized!!!"<<endl<<endl;
     }

     std::string result;
     if (argc == 1) {
       result = "./output/";
     } else if (argc > 1) {
       result = std::string("./output_") + argv[1] + std::string("/");
     }

     ofstream file_out_energy(result + std::string("energy_raw.dat"), std::ofstream::app);
     ofstream file_out_energy_squared(result + std::string("energy_squared_raw.dat"), std::ofstream::app);
     ofstream file_out_Si_Sj(result + std::string("Si_Sj_raw.dat"), std::ofstream::app);

     if (rank==0){
       file_out_energy<<std::fixed << setprecision(14) << endl;
       file_out_energy_squared<<std::fixed << setprecision(14) << endl;
       file_out_Si_Sj<<std::fixed << setprecision(14) << endl;
     }

     complex<double> appo_energy = std::complex<double>(0.0, 0.0);
     complex<double> energy_measured = std::complex<double>(0.0, 0.0);
     complex<double> tot_energy_measured = std::complex<double>(0.0, 0.0);
     double energy_squared_measured = 0.; 
     double tot_energy_squared_measured = 0.; 

     MatrixXcd Si_Sj_measured = MatrixXcd::Zero(L,L);
     MatrixXcd tot_Si_Sj_measured = MatrixXcd::Zero(L,L);

     int print_index = N/20;

     // Here we will measure the observables
     for(int i=0; i<N; i++){
       if(rank==0 and i%(print_index)==0){
	      cout<<i<<" of "<<N<<endl;
       }

       energy_measured = std::complex<double>(0.0, 0.0);
       tot_energy_measured = std::complex<double>(0.0, 0.0);
       energy_squared_measured = 0.; 
       tot_energy_squared_measured = 0.; 

       Si_Sj_measured.setZero();
       tot_Si_Sj_measured.setZero();

       for(int j=0; j<M; j++){
          for(int single_step=0; single_step<nsparse_ave; single_step++){
            (wlkrs[j]).Metropolis_step();
          }

          (wlkrs[j]).local_energy_Hubbard();

          appo_energy = (wlkrs[j]).get_energy();
          energy_measured += appo_energy;
          energy_squared_measured += real(appo_energy*conj(appo_energy));

          (wlkrs[j]).add_Si_Sj(Si_Sj_measured);

          count_scra += nsparse_ave;
          if(count_scra > N_scra*nsparse_ave){
            (wlkrs[j]).restart_W_inv_from_scratch();
            count_scra = 0;
          }
       }

       MPI_Reduce(&energy_measured,&tot_energy_measured,2,MPI_DOUBLE,MPI_SUM,0.,MPI_COMM_WORLD);
       MPI_Reduce(&energy_squared_measured,&tot_energy_squared_measured,1,MPI_DOUBLE,MPI_SUM,0.,MPI_COMM_WORLD);
       MPI_Reduce(Si_Sj_measured.data(),tot_Si_Sj_measured.data(),2*L*L,MPI_DOUBLE,MPI_SUM,0.,MPI_COMM_WORLD);

       if(rank==0){
          file_out_energy<<tot_energy_measured/(1.*M*numprocs)<<endl;
          file_out_energy_squared<<tot_energy_squared_measured/(1.*M*numprocs)<<endl;
          file_out_Si_Sj<<tot_Si_Sj_measured/(1.*M*numprocs)<<endl<<endl;
       }
     }

     if(rank==0){
       cout<<endl<<"Acceptance hopping : ";
       cout<<(wlkrs[0]).get_NN_hopping_acceptance()<<endl;		 
       cout<<endl<<"Acceptance spin-flip : ";
       cout<<(wlkrs[0]).get_NN_SpinFlip_acceptance()<<endl<<endl;		 
     }

     file_out_energy.close();
     file_out_energy_squared.close();
     file_out_Si_Sj.close();

   }
   else{ //In this case you DO perform Stochastic reconfiguration!

     if(rank==0){
       cout<<"Optimizing the wave function"<<endl<<endl;
     }

     VectorXcd O_local = VectorXcd::Zero(SR_parameters);
     VectorXcd O_mean = VectorXcd::Zero(SR_parameters);
     VectorXcd e_O_mean = VectorXcd::Zero(SR_parameters);
     MatrixXcd S_mean = MatrixXcd::Zero(SR_parameters,SR_parameters);
     VectorXcd global_O_mean = VectorXcd::Zero(SR_parameters);
     VectorXcd global_e_O_mean = VectorXcd::Zero(SR_parameters);
     MatrixXcd global_S_mean = MatrixXcd::Zero(SR_parameters,SR_parameters);

     MatrixXd real_S_mat = MatrixXd::Zero(SR_parameters,SR_parameters);
     VectorXd d_alpha    = VectorXd::Zero(SR_parameters);

     std::string result;
     if (argc == 1) {
       result = "./output/";
     } else if (argc > 1) {
       result = std::string("./output_") + argv[1] + std::string("/");
     }

     ofstream file_out_energy(result+"energy_SR.dat", std::ofstream::app);
     ofstream file_out_params(result+"params_SR.dat", std::ofstream::app);

     for(int time_step=0; time_step<n_iterations_SR and (float( clock () - begin_time )/CLOCKS_PER_SEC) < 42000.; time_step++){

       int count_scra = 0;

       for(int j=0; j<M; j++){
	        (wlkrs[j]).initialize_variational_parameters(abs_f_ij, phase_f_ij ); 
       }

       //Thermalization
       for(int j=0; j<M; j++){
          for(int i=0; i<N_therm*nsparse_ave; i++){
            (wlkrs[j]).Metropolis_step();

            count_scra += 1;
            if(count_scra > N_scra*nsparse_ave){
              (wlkrs[j]).restart_W_inv_from_scratch();
              count_scra = 0;
            }
          }
       }

       complex<double> e_mean = 0.;
       O_mean.setZero();
       e_O_mean.setZero();
       S_mean.setZero();
       complex<double> global_e_mean = 0.;
       global_O_mean.setZero();
       global_e_O_mean.setZero();
       global_S_mean.setZero();

       // Here we will measure the observables needed for the SR
       for(int i=0; i<N; i++){
          for(int j=0; j<M; j++){
            for(int single_step=0; single_step<nsparse_ave; single_step++){
              (wlkrs[j]).Metropolis_step();
            }

            (wlkrs[j]).local_energy_Hubbard();
            (wlkrs[j]).compute_O_x_ALL_parameters(O_local);
            
            e_mean   += (wlkrs[j]).get_energy();
            O_mean   += O_local;
            e_O_mean += O_local*conj( (wlkrs[j]).get_energy() );
            S_mean   += O_local*O_local.adjoint();

            count_scra += nsparse_ave;
            if(count_scra > N_scra*nsparse_ave){
              (wlkrs[j]).restart_W_inv_from_scratch();
              count_scra = 0;
            }
          }
       }

       MPI_Reduce(&e_mean,&global_e_mean,2,MPI_DOUBLE,MPI_SUM,0.,MPI_COMM_WORLD);
       MPI_Reduce(O_mean.data(),global_O_mean.data(),2*SR_parameters,MPI_DOUBLE,MPI_SUM,0.,MPI_COMM_WORLD);
       MPI_Reduce(e_O_mean.data(),global_e_O_mean.data(),2*SR_parameters,MPI_DOUBLE,MPI_SUM,0.,MPI_COMM_WORLD);
       MPI_Reduce(S_mean.data(),global_S_mean.data(),2*(SR_parameters)*(SR_parameters),MPI_DOUBLE,MPI_SUM,0.,MPI_COMM_WORLD);

       // Invert the S matrix and compute the update of the parameters
       if(rank==0){
          global_e_mean = global_e_mean / (1.*N*M*numprocs);
          global_O_mean = global_O_mean / (1.*N*M*numprocs);
          global_e_O_mean = global_e_O_mean / (1.*N*M*numprocs);
          global_S_mean = global_S_mean / (1.*N*M*numprocs);

          global_e_O_mean = -global_e_O_mean + global_O_mean*conj( global_e_mean );
          global_S_mean = global_S_mean - global_O_mean*global_O_mean.adjoint();

          real_S_mat    = global_S_mean.real() + epsilon_SR*MatrixXd::Identity(SR_parameters, SR_parameters);

          d_alpha = real_S_mat.llt().solve(global_e_O_mean.real());

          cout<<time_step+1<<" / "<<n_iterations_SR<<endl;
          double norm_d_alpha = d_alpha.norm();
          cout<<"d_alpha actual norm = "<<norm_d_alpha<<endl;

          double norm_cutoff = 0.25*sqrt(SR_parameters); // In case the norm of d_alpha is too big, we will rescale it
          norm_d_alpha = max(norm_d_alpha, norm_cutoff);

          d_alpha = d_alpha * (norm_cutoff / norm_d_alpha);
          cout<<"d_alpha used   norm = "<<d_alpha.norm()<<endl;

          //////////////
          //SelfAdjointEigenSolver<MatrixXd> eigensol_Smat(global_S_mean);
          //VectorXd eigs_Smat = eigensol_Smat.eigenvalues();
          //cout<<"Eigenvalues of S matrix : "<<endl<<eigs_Smat<<endl;
          //////////////

          file_out_energy<<global_e_mean<<endl;

          int conta_pars = 0;
          conta_pars = print_and_change_f_ij_params(file_out_params, abs_f_ij, conta_pars, d_alpha, gamma);
          conta_pars = print_and_change_f_ij_params(file_out_params, phase_f_ij, conta_pars, d_alpha, gamma);

          file_out_params<<endl;

          cout<<"Acceptance hopping / spin-flip : ";
          cout<<(wlkrs[0]).get_NN_hopping_acceptance()<<" / "<<(wlkrs[0]).get_NN_SpinFlip_acceptance()<<endl<<endl;
       }

       // Broadcast the updated parameters to all processes
       MPI_Bcast(abs_f_ij.data(), L*L, MPI_DOUBLE, 0, MPI_COMM_WORLD);
       MPI_Bcast(phase_f_ij.data(), L*L, MPI_DOUBLE, 0, MPI_COMM_WORLD);
     }

     file_out_energy.close();
     file_out_params.close();

     ofstream file_out_info(result+"info.dat");
     if (rank==0) {
       (wlkrs[0]).write_info_output_file(file_out_info);
     }
     file_out_info.close();

     // Printing restarting parameters
     if( rank == 0 ){
       VectorXd appo_params = VectorXd::Zero(SR_parameters);
       ofstream restart_file(input_folder+"/restarting_pars.dat");

       int conta_pars = 0;
       conta_pars = print_and_change_f_ij_params(restart_file, abs_f_ij, conta_pars, appo_params, 0.);
       conta_pars = print_and_change_f_ij_params(restart_file, phase_f_ij, conta_pars, appo_params, 0.);
       restart_file.close();
     }

   }


   if(rank==0){
     cout<<"Finished"<<endl<<endl;
     cout << float( clock () - begin_time ) /  CLOCKS_PER_SEC << endl << endl<< endl;
   }

   if(rank==0){
     (wlkrs[0]).print_iconf();
     (wlkrs[0]).local_energy_Hubbard();
     cout<<endl<<"Energy = "<<(wlkrs[0]).get_energy()<<endl<<endl;
     (wlkrs[0]).DEBUG_compute_O_x_f_ij();
     (wlkrs[0]).DEBUG_compute_O_x_NO_det();
   }

   MPI_Barrier(MPI_COMM_WORLD);
   MPI_Finalize();
   return 0;
}



int read_ivic(ArrayXi & imulti, Eigen::Tensor<int, 3> & ivic){ 

   std::ifstream ivict_file("include/ivic.dat");
   if (!ivict_file.is_open()) {
     std::cerr << "Unable to open ivic file" << std::endl;
     return 1; // Return an error code	
   }

   int value;
   for (int i = 0; i < L; ++i) {
     for (int j = 0; j < Imaxmulti; ++j) {
       for (int k = 0; k < N_distances; ++k) {
       	 if (!(ivict_file >> value)) {
          std::cerr << "Error reading values from ivic.dat" << std::endl;
          return 1; // Return an error code	
        }
        ivic(i, j, k) = value;
       }
     }
   }

   for(int i=0;i<N_distances;i++){
     if (!(ivict_file >> value)) {
       std::cerr << "Error reading values from ivic.dat" << std::endl;
       return 1; // Return an error code		     
     }
     imulti[i] = value;
   }

   ivict_file.close();

   return 0;
}



// Function to read the file and store the data
void read_f_ij_params(const std::string& input_folder, MatrixXd & f_ij_mat) {

    std::string filename = input_folder; 
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(1);
    }

    int count = 0;
    double value;

    for (int i = 0; i < L; ++i) {
      for (int j = 0; j < L; ++j) {
        if (!(file >> value)) {
          std::cerr << "Error: Not enough values to fill a " << L << "x" << L << " matrix." << std::endl;
          exit(1);
        }
        f_ij_mat(i, j) = value;
        ++count;
      }
    }

    // Check if the file contains too many numbers
    if (file >> value) {
      std::cerr << "Warning: File "<<filename<<" contains more than " << L*L << " values. Extra values will be ignored." << std::endl;
    }

    file.close();

    return;
}


int print_and_change_f_ij_params(ofstream & file_out_params,  MatrixXd & f_ij_mat, int conta_pars,  VectorXd & d_alpha,  double gamma,  double const_mult){ 

   for (int i = 0; i < L; ++i) {
     for (int j = 0; j < L; ++j) {
       file_out_params<<f_ij_mat(i,j)<<"  ";
       f_ij_mat(i,j) = f_ij_mat(i,j)*const_mult + d_alpha[conta_pars]*gamma;
       conta_pars += 1;
     }
   }

   return conta_pars;
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
