#define NDEBUG

#include <complex.h>
#include <fftw3.h>
#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <algorithm>
#include <string>
#include <chrono>
#include <thread>

using namespace Eigen;
using namespace std;

ArrayXXd K,L,KKLL;
ArrayXXcd ikU,ilU,ikW,ilW,ikT,ilT,ikS,ilS;
ArrayXXcd grU,grW,grT,grS;

double Pr,Rrho,tho,dt;
int restart;
string method;
fftw_plan plan;
fftw_plan iplan;
complex<double> II(0,1);


void write_binary(const char* filename, ArrayXXcd matrix){
    std::ofstream out(filename, std::ios::out | std::ios::binary | std::ios::trunc);
    typename ArrayXXcd::Index rows=matrix.rows(), cols=matrix.cols();
    out.write((char*) (&rows), sizeof(typename ArrayXXcd::Index));
    out.write((char*) (&cols), sizeof(typename ArrayXXcd::Index));
    out.write((char*) matrix.data(), rows*cols*sizeof(typename ArrayXXcd::Scalar) );
    out.close();
}

ArrayXXcd read_binary(const char* filename, ArrayXXcd matrix){
    std::ifstream in(filename, std::ios::in | std::ios::binary);
    typename ArrayXXcd::Index rows=0, cols=0;
    in.read((char*) (&rows),sizeof(typename ArrayXXcd::Index));
    in.read((char*) (&cols),sizeof(typename ArrayXXcd::Index));
    matrix.resize(rows, cols);
    in.read( (char *) matrix.data() , rows*cols*sizeof(typename ArrayXXcd::Scalar) );
    in.close();
    return matrix;
}

ArrayXXcd f_U(ArrayXXcd U,ArrayXXcd W){
	ikU = II*K*U;
	ilU = II*L*U;
	ArrayXXcd ifU = U;
	ArrayXXcd ifW = W;
	ArrayXXcd ifdxU = U;
	ArrayXXcd ifdyU = U;
	ArrayXXcd NLU = U;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(U.data()), reinterpret_cast<fftw_complex*>(ifU.data()));
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(W.data()), reinterpret_cast<fftw_complex*>(ifW.data()));
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(ikU.data()), reinterpret_cast<fftw_complex*>(ifdxU.data()));
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(ilU.data()), reinterpret_cast<fftw_complex*>(ifdyU.data()));
	grU = (ifU*ifdxU)+(ifW*ifdyU);
	fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(grU.data()), reinterpret_cast<fftw_complex*>(NLU.data()));
	
	ArrayXXcd U_dt = -Pr*KKLL*U - NLU;
	
	

return U_dt;
}

ArrayXXcd f_W(ArrayXXcd U,ArrayXXcd W,ArrayXXcd T,ArrayXXcd S){
	ArrayXXcd ifU = U;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(U.data()), reinterpret_cast<fftw_complex*>(ifU.data()));
	ArrayXXcd ifW = W;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(W.data()), reinterpret_cast<fftw_complex*>(ifW.data()));
	ArrayXXcd ifdxW = W;
	ikW = II*K*W;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(ikW.data()), reinterpret_cast<fftw_complex*>(ifdxW.data()));
	ArrayXXcd ifdyW = W;
	ilW = II*L*W;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(ilW.data()), reinterpret_cast<fftw_complex*>(ifdyW.data()));
	ArrayXXcd NLW = W;
	grW = (ifU*ifdxW)+(ifW*ifdyW);
	fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(grW.data()), reinterpret_cast<fftw_complex*>(NLW.data()));
	ArrayXXcd W_dt = -Pr*(-T+S+KKLL*W) - NLW;
	
	
return W_dt;
}

ArrayXXcd f_T(ArrayXXcd U,ArrayXXcd W,ArrayXXcd T){
	ArrayXXcd ifU =U;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(U.data()), reinterpret_cast<fftw_complex*>(ifU.data()));
	ArrayXXcd ifW = W;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(W.data()), reinterpret_cast<fftw_complex*>(ifW.data()));
	ArrayXXcd ifdxT = T;
	ikT = II*K*T;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(ikT.data()), reinterpret_cast<fftw_complex*>(ifdxT.data()));
	ArrayXXcd ifdyT = T;
	ilT = II*L*T;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(ilT.data()), reinterpret_cast<fftw_complex*>(ifdyT.data()));
	grT = (ifU*ifdxT)+(ifW*ifdyT);
	ArrayXXcd NLT = T;
	fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(grT.data()), reinterpret_cast<fftw_complex*>(NLT.data()));
	ArrayXXcd T_dt = -T*KKLL - W  -NLT;
return T_dt;
}

ArrayXXcd f_S(ArrayXXcd U,ArrayXXcd W,ArrayXXcd S){
	ArrayXXcd ifU =U;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(U.data()), reinterpret_cast<fftw_complex*>(ifU.data()));
	ArrayXXcd ifW = W;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(W.data()), reinterpret_cast<fftw_complex*>(ifW.data()));
	ArrayXXcd ifdxS = S;
	ikS = II*K*S;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(ikS.data()), reinterpret_cast<fftw_complex*>(ifdxS.data()));
	ArrayXXcd ifdyS = S;
	ilS = II*L*S;
	fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(ilS.data()), reinterpret_cast<fftw_complex*>(ifdyS.data()));
	grS = (ifU*ifdxS)+(ifW*ifdyS);
	ArrayXXcd NLS = S;
	fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(grS.data()), reinterpret_cast<fftw_complex*>(NLS.data()));
	ArrayXXcd S_dt = -tho*S*KKLL - W/Rrho - NLS;
return S_dt;
}


int main(){

//~ ############ Reading of .par file
int kmax;
int lmax;
double iter;
std::ifstream cFile ("DDC.par");
if (cFile.is_open())
{
	std::string line;
	while(getline(cFile, line)){
		line.erase(std::remove_if(line.begin(), line.end(), ::isspace),
							 line.end());
		if(line[0] == '#' || line.empty())
			continue;
		auto delimiterPos = line.find("=");
		auto name = line.substr(0, delimiterPos);
		auto value = line.substr(delimiterPos + 1);
		if ( name.compare("Rrho")  == 0) {
			Rrho = stod(value);
			
		}
		else if (name.compare("kmax") == 0) {
			kmax = stod(value);
		}
		else if (name.compare("lmax") == 0) {
			lmax = stod(value);
		}
		else if (name.compare("tho") == 0) {
			tho = stod(value);
		}
		else if (name.compare("Pr") == 0) {
			Pr = stod(value);
		}
		else if (name.compare("dt") == 0) {
			dt = stod(value);
		}
		else if (name.compare("iter") == 0) {
			iter = stod(value);
		}
		else if (name.compare("restart") == 0) {
			restart = stod(value);
		}
		else if (name.compare("method") == 0) {
			method = value;
		}
	}
	
}
else {
	std::cerr << "Couldn't open config file for reading.\n";
}
//~ ############

//~ ############ Prep K,L
ArrayXd vk;
ArrayXd vl;
ArrayXd v2k;
ArrayXd v2l;
vk = ArrayXd::LinSpaced(kmax+1, 0, kmax);
vl = ArrayXd::LinSpaced(lmax+1, 0, lmax);
v2k = ArrayXd::LinSpaced(kmax, -kmax, -1);
v2l = ArrayXd::LinSpaced(lmax, -lmax, -1);

//~ ArrayXcd VK(vk.size()+v2k.size()+1); 
//~ VK << vk,0,v2k;
//~ ArrayXcd VL(vl.size()+v2l.size()+1); 
//~ VL << vl,0,v2l;

ArrayXd VK(vk.size()+v2k.size());
VK << vk,v2k;
ArrayXd VL(vl.size()+v2l.size()); 
VL << vl,v2l;

//~ I have replaced all kmax*2+2 with kmax*2+1, lamx too

L = VL.replicate(1,kmax*2+1);
K = VK.replicate(1,lmax*2+1).transpose();
KKLL = K*K+L*L;
//~ ############

//~ ############ Initialisation fields

ArrayXXcd U,W,T,S,TT;

	U = ArrayXXd::Random(lmax*2+1,kmax*2+1)*1e-6;
	TT = ArrayXXd::Random(lmax*2+1,kmax*2+1)*1e-6;
	W = ArrayXXcd::Constant(lmax*2+1,kmax*2+1,0);
	S = ArrayXXcd::Constant(lmax*2+1,kmax*2+1,0);
	T = TT;
	
//~ ############ Prepartion for FFT
int n[] = {kmax*2+1, lmax*2+1};
int howmany = 1;
int idist = 0;
int odist = 0;
int istride = 1;
int ostride = 1; 
int *inembed = n, *onembed = n;
ArrayXXcd Ufft = U;

plan =  fftw_plan_many_dft(2,n,howmany,
                             reinterpret_cast<fftw_complex*>(U.data()),inembed,
                             istride, idist,
                             reinterpret_cast<fftw_complex*>(Ufft.data()), onembed,
                             ostride, odist,
                             FFTW_FORWARD,FFTW_PATIENT);
iplan =  fftw_plan_many_dft(2,n,howmany,
                             reinterpret_cast<fftw_complex*>(U.data()),inembed,
                             istride, idist,
                             reinterpret_cast<fftw_complex*>(Ufft.data()), onembed,
                             ostride, odist,
                             FFTW_BACKWARD,FFTW_PATIENT);
                             
U = ArrayXXcd::Constant(lmax*2+1,kmax*2+1,0);

fftw_execute_dft(plan, reinterpret_cast<fftw_complex*>(TT.data()), reinterpret_cast<fftw_complex*>(T.data()));
/////////
write_binary("U0.dat",U);
write_binary("W0.dat",W);
write_binary("T0.dat",T);
write_binary("S0.dat",S);
//~ ############ Restart option
if (restart == 1){
	ArrayXXcd Uf,Wf,Tf,Sf;
	Uf = read_binary("U.dat",Uf);
	Wf = read_binary("W.dat",Wf);
	Tf = read_binary("T.dat",Tf);
	Sf = read_binary("S.dat",Sf);
	U = Uf;
	W = Wf;
	T = Tf;
	S = Sf;
	cout << "restart from previous field" << endl;
	
}

if (restart == 2){
	ArrayXXcd Uf,Wf,Tf,Sf;
	Uf = read_binary("U0.dat",Uf);
	Wf = read_binary("W0.dat",Wf);
	Tf = read_binary("T0.dat",Tf);
	Sf = read_binary("S0.dat",Sf);
	U = Uf;
	W = Wf;
	T = Tf;
	S = Sf;
	cout << "restart from previous field" << endl;
	
}
//~ ############

//~ ############ Def for projection

ArrayXXd ProjKK = (K*K)/(K*K+L*L);
ArrayXXd ProjKL = (K*L)/(K*K+L*L);
ArrayXXd ProjLL = (L*L)/(K*K+L*L);
ProjKK = (ProjKK.isFinite()).select(ProjKK,0);
ProjLL = (ProjLL.isFinite()).select(ProjLL,0);
ProjKL = (ProjKL.isFinite()).select(ProjKL,0);

//~ ############


//~ ############ Runge_Kutta 4
if ( method.compare("RK4")  == 0){
cout << "Pr= " << Pr << endl << "tau= " << tho << endl << "rrho= " << Rrho << endl;
ArrayXXcd k1_U,k1_W,k1_T,k1_S,
		k2_U,k2_W,k2_T,k2_S,
		k3_U,k3_W,k3_T,k3_S,
		k4_U,k4_W,k4_T,k4_S,
		ndivU,ndivW;

std::ofstream energy;
	energy.open ("energy.txt");
	energy << "Energy, grow \n";



for (int i=0; i<iter; ++i){
	double co = (static_cast<double> (i)+1)/ static_cast<double> (iter) * 100;
	ArrayXXd ener = sqrt(U.real()*U.real()+W.real()*W.real());
	

	k1_U = dt * f_U(U,W);
	k1_W = dt * f_W(U,W,T,S);
	k1_T = dt * f_T(U,W,T);
	k1_S = dt * f_S(U,W,S);

	k2_U = dt * f_U(U + k1_U/2,W + k1_W/2);
	k2_W = dt * f_W(U + k1_U/2,W + k1_W/2,T + k1_T/2, S + k1_S/2);
	k2_T = dt * f_T(U + k1_U/2,W + k1_W/2,T + k1_T/2);
	k2_S = dt * f_S(U + k1_U/2,W + k1_W/2,S + k1_S/2);

	k3_U = dt * f_U(U + k2_U/2,W + k2_W/2);
	k3_W = dt * f_W(U + k2_U/2,W + k2_W/2,T + k2_T/2, S + k2_S/2);
	k3_T = dt * f_T(U + k2_U/2,W + k2_W/2,T + k2_T/2);
	k3_S = dt * f_S(U + k2_U/2,W + k2_W/2,S + k2_S/2);

	k4_U = dt * f_U(U + k3_U,W + k3_W);
	k4_W = dt * f_W(U + k3_U,W + k3_W,T+ k3_T, S+ k3_S);
	k4_T = dt * f_T(U + k3_U,W + k3_W,T+ k3_T);
	k4_S = dt * f_S(U + k3_U,W + k3_W,S+ k3_S);

	U = U +(k1_U+2*k2_U+2*k3_U+k4_U)/6;
	W = W +(k1_W+2*k2_W+2*k3_W+k4_W)/6;
	T = T +(k1_T+2*k2_T+2*k3_T+k4_T)/6;
	S = S +(k1_S+2*k2_S+2*k3_S+k4_S)/6;
	
	ndivU = U - (ProjKK*U + ProjKL*W);
	ndivW = W - (ProjKL*U + ProjLL*W);
	
	U = ndivU;
	W = ndivW;
	double divu = (U*K+W*L).real().sum();
	ArrayXXd ener2 = sqrt(U.real()*U.real()+W.real()*W.real());
	double grow = (log(ener2.sum()/ener.sum())/dt);
	energy << ener2.sum() << "," << grow << "\n" ;
	cout << "\r" << co  << "%" << "   Energy : " << ener2.sum()  <<    "   Grow : " << grow << "   div : " << divu << "   dt:" << dt << flush;

}
}
//~ ############

//~ ############ Runge Kutta 2
else if ( method.compare("RK2")  == 0){
	cout << "Pr= " << Pr << endl << "tau= " << tho << endl << "rrho= " << Rrho << endl;
ArrayXXcd k1_U,k1_W,k1_T,k1_S,
		k2_U,k2_W,k2_T,k2_S,
		ndivU,ndivW;

std::ofstream energy;
	energy.open ("energy.txt");
	energy << "Energy, grow \n";

ArrayXXcd Ut,Wt,Tt,St,k2_Wt;
ArrayXXd tU,tW,tT,tS;
double tol = 1e-7;
double time = 0;
double co,ener,ener2,grow;
cout << "RK2 method " << endl;
int i;
int cont = 0;
double fin = iter;
//~ for(i=iter; i--; ){
while(cont == 0){
	co =  time/fin * 100;
	ener = (sqrt(abs2(U)+abs2(W))).sum();
	
	k1_U = f_U(U,W);
	k1_W = f_W(U,W,T,S);
	k1_T = f_T(U,W,T);
	k1_S = f_S(U,W,S);

	k2_U = f_U(U + dt * 2*k1_U/3,W+ dt * 2*k1_W/3);
	k2_W = f_W(U + dt * 2*k1_U/3,W+ dt * 2*k1_W/3,T +dt * 2*k1_T/3, S + dt * 2*k1_S/3);
	k2_T = f_T(U + dt * 2*k1_U/3,W+ dt * 2*k1_W/3,T +dt * 2*k1_T/3);
	k2_S = f_S(U + dt * 2*k1_U/3,W+ dt * 2*k1_W/3,S +dt * 2*k1_S/3);
	
	k2_Wt = f_W(U + dt *k1_U/3,W+ dt *k1_W/3,T +dt *k1_T/3, S + dt *k1_S/3);
	Wt = W + dt/2 * (0.25*k1_W+0.75*k2_Wt);
	
	U = U + dt * (0.25*k1_U+0.75*k2_U);
	W = W + dt * (0.25*k1_W+0.75*k2_W);
	T = T + dt * (0.25*k1_T+0.75*k2_T);
	S = S + dt * (0.25*k1_S+0.75*k2_S);
	
	//~ ####### Computing of stepsize
	tW = (Wt-W).real();
	tW = tW.abs();
	dt = 0.9*dt*sqrt(tol/(2*tW.maxCoeff()));
	
	//~ #############
	
	ndivU = U - (ProjKK*U + ProjKL*W);
	ndivW = W - (ProjKL*U + ProjLL*W);
	
	U = ndivU;
	W = ndivW;
	
	time += dt;
	//~ double divu = (U*K+W*L).real().sum();
	ener2 = (sqrt(abs2(U)+abs2(W))).sum();
	grow = (log(ener2/ener)/dt);
	energy << ener2 << "," << grow << "\n" ;
	cout << "\r" << co  << "%" << "   Energy : " << ener2  <<    "   Grow : " << grow << "    dt = " << dt << "    time = " << time << flush;
	if(time>iter){
		cont = 1;
	}
}
}

//~ ############

//~ ############ Saving files
ArrayXXcd Ts,Ss,Ws,Us;
Ts= T;
Ss = S;
Ws = W;
Us = U;
fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(T.data()), reinterpret_cast<fftw_complex*>(Ts.data()));
fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(S.data()), reinterpret_cast<fftw_complex*>(Ss.data()));
fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(W.data()), reinterpret_cast<fftw_complex*>(Ws.data()));
fftw_execute_dft(iplan, reinterpret_cast<fftw_complex*>(U.data()), reinterpret_cast<fftw_complex*>(Us.data()));
write_binary("U.dat",U);
write_binary("W.dat",W);
write_binary("T.dat",T);
write_binary("S.dat",S);
cout << endl << "Saved Successfully" << endl;

std::ofstream file("T.txt");
if (file.is_open()){
	file << Ts ;
}

std::ofstream fileS("S.txt");
if (fileS.is_open()){
	fileS << Ss;
}

std::ofstream file4("W.txt");
if (file4.is_open()){
	file4 << Ws;
}
//~ ############

return 0;
}


