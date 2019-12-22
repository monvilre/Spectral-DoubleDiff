//~ #define NDEBUG

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

ArrayXXcd K,L,KKLL;
double Pr,Rrho,tho,dt;
int restart;
string method;
fftw_plan plan;
fftw_plan iplan;

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

ArrayXXcd f_U(ArrayXXcd U,ArrayXXcd W,ArrayXXcd T,ArrayXXcd S){
	ArrayXXcd U_dt;
	ArrayXXcd ifU =U;
	fftw_execute_dft(iplan, (fftw_complex*) U.data(), (fftw_complex*) ifU.data());
	ArrayXXcd ifW = W;
	fftw_execute_dft(iplan, (fftw_complex*) W.data(), (fftw_complex*) ifW.data());
	ArrayXXcd ifdxU = U;
	ArrayXXcd ikU = I*K*U;
	fftw_execute_dft(iplan, (fftw_complex*) ikU.data(), (fftw_complex*) ifdxU.data());
	ArrayXXcd ifdyU = U;
	ArrayXXcd ilU = I*L*U;
	fftw_execute_dft(iplan, (fftw_complex*) ilU.data(), (fftw_complex*) ifdyU.data());
	ArrayXXcd NLU = U;
	ArrayXXcd grU = -ifU*ifdxU- ifW*ifdyU;
	fftw_execute_dft(plan, (fftw_complex*) grU.data(), (fftw_complex*) NLU.data());
	U_dt = -Pr*KKLL*U + NLU;
	//~ U_dt = -Pr*KKLL*U;
return U_dt;
}

ArrayXXcd f_W(ArrayXXcd U,ArrayXXcd W,ArrayXXcd T,ArrayXXcd S){
	ArrayXXcd W_dt;
	ArrayXXcd ifU_w = W;
	fftw_execute_dft(iplan, (fftw_complex*) U.data(), (fftw_complex*) ifU_w.data());
	ArrayXXcd ifW = W;
	fftw_execute_dft(iplan, (fftw_complex*) W.data(), (fftw_complex*) ifW.data());
	ArrayXXcd ifdxW = W;
	ArrayXXcd ikW = I*K*W;
	fftw_execute_dft(iplan, (fftw_complex*) ikW.data(), (fftw_complex*) ifdxW.data());
	ArrayXXcd ifdyW = W;
	ArrayXXcd ilW = I*L*W;
	fftw_execute_dft(iplan, (fftw_complex*) ilW.data(), (fftw_complex*) ifdyW.data());
	ArrayXXcd NLW = W;
	ArrayXXcd grW = -ifU_w*ifdxW- ifW*ifdyW;
	fftw_execute_dft(plan, (fftw_complex*) grW.data(), (fftw_complex*) NLW.data());
	W_dt = -Pr*(-T+S+KKLL*W) + NLW;
return W_dt;
}

ArrayXXcd f_T(ArrayXXcd U,ArrayXXcd W,ArrayXXcd T,ArrayXXcd S){
	ArrayXXcd T_dt;
	ArrayXXcd ifU =U;
	fftw_execute_dft(iplan, (fftw_complex*) U.data(), (fftw_complex*) ifU.data());
	ArrayXXcd ifW = W;
	fftw_execute_dft(iplan, (fftw_complex*) W.data(), (fftw_complex*) ifW.data());
	ArrayXXcd ifdxT = T;
	ArrayXXcd ikT = I*K*T;
	fftw_execute_dft(iplan, (fftw_complex*) ikT.data(), (fftw_complex*) ifdxT.data());
	ArrayXXcd ifdyT = T;
	ArrayXXcd ilT = I*L*T;
	fftw_execute_dft(iplan, (fftw_complex*) ilT.data(), (fftw_complex*) ifdyT.data());
	ArrayXXcd grT = -ifU*ifdxT- ifW*ifdyT;
	ArrayXXcd NLT = T;
	fftw_execute_dft(plan, (fftw_complex*) grT.data(), (fftw_complex*) NLT.data());
	T_dt = -T*KKLL - W +NLT;
return T_dt;
}

ArrayXXcd f_S(ArrayXXcd U,ArrayXXcd W,ArrayXXcd T,ArrayXXcd S){
	ArrayXXcd S_dt;
	ArrayXXcd ifU =U;
	fftw_execute_dft(iplan, (fftw_complex*) U.data(), (fftw_complex*) ifU.data());
	ArrayXXcd ifW = W;
	fftw_execute_dft(iplan, (fftw_complex*) W.data(), (fftw_complex*) ifW.data());
	ArrayXXcd ifdxS = S;
	ArrayXXcd ikS = I*K*S;
	fftw_execute_dft(iplan, (fftw_complex*) ikS.data(), (fftw_complex*) ifdxS.data());
	ArrayXXcd ifdyS = S;
	ArrayXXcd ilS = I*L*S;
	fftw_execute_dft(iplan, (fftw_complex*) ilS.data(), (fftw_complex*) ifdyS.data());
	ArrayXXcd grS = -ifU*ifdxS- ifW*ifdyS;
	ArrayXXcd NLS = S;
	fftw_execute_dft(plan, (fftw_complex*) grS.data(), (fftw_complex*) NLS.data());
	S_dt = -tho*S*KKLL - W/Rrho + NLS;
return S_dt;
}


int main(){

int kmax;
int lmax;
int iter;
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


ArrayXcd vk;
ArrayXcd vl;
ArrayXcd v2k;
ArrayXcd v2l;
vk = ArrayXd::LinSpaced(kmax+1, 0, kmax);
vl = ArrayXd::LinSpaced(lmax+1, 0, lmax);
v2k = ArrayXd::LinSpaced(kmax, -kmax, -1);
v2l = ArrayXd::LinSpaced(lmax, -lmax, -1);

ArrayXcd VK(vk.size()+v2k.size()+1); 
VK << vk,0,v2k;
ArrayXcd VL(vl.size()+v2l.size()+1); 
VL << vl,0,v2l;



L = VL.replicate(1,kmax*2+2);
K = VK.replicate(1,lmax*2+2).transpose();
KKLL = K*K+L*L;
//~ Initialisation

ArrayXXcd U,W,T,S,TT;

if (restart == 0){
	U = ArrayXXcd::Constant(lmax*2+2,kmax*2+2,0);
	TT = ArrayXXd::Random(lmax*2+2,kmax*2+2);
	
	W = ArrayXXcd::Constant(lmax*2+2,kmax*2+2,0);
	//~ T = ArrayXXcd::Constant(lmax*2+2,kmax*2+2,0);
	//~ W = ArrayXXcd::Random(lmax*2+2,kmax*2+2)*1e-2;
	S = ArrayXXcd::Constant(lmax*2+2,kmax*2+2,0);
	//~ TT.col(kmax+1) = 0;
	//~ TT.row(lmax+1) = 0;
	T = TT;
	
std::ofstream fileS0("S0.txt");
if (fileS0.is_open()){
	fileS0 << T;
}
	
}
else if (restart == 1){
	U = read_binary("U.dat",U);
	W = read_binary("W.dat",W);
	T = read_binary("T.dat",T);
	S = read_binary("S.dat",S);
	cout << "restart from previous field" << endl;
}

ArrayXXcd ProjKK = (K*K)/(K*K+L*L);
ArrayXXcd ProjKL = (K*L)/(K*K+L*L);
ArrayXXcd ProjLL = (L*L)/(K*K+L*L);
ProjKK = (ProjKK.isFinite()).select(ProjKK,0);
ProjLL = (ProjLL.isFinite()).select(ProjLL,0);
ProjKL = (ProjKL.isFinite()).select(ProjKL,0);

// Prepartion for FFT
int n[] = {kmax*2+2, lmax*2+2};
int howmany = 1;
int idist = 0;
int odist = 0;
int istride = 1;
int ostride = 1; 
int *inembed = n, *onembed = n;
int sign = -1; // -1 for fft 1 for ifft
ArrayXXcd Ufft = U;

plan =  fftw_plan_many_dft(2,n,howmany,
                             (fftw_complex*) U.data(),inembed,
                             istride, idist,
                             (fftw_complex*) Ufft.data(), onembed,
                             ostride, odist,
                             sign,FFTW_MEASURE);
iplan =  fftw_plan_many_dft(2,n,howmany,
                             (fftw_complex*) U.data(),inembed,
                             istride, idist,
                             (fftw_complex*) Ufft.data(), onembed,
                             ostride, odist,
                             -sign,FFTW_MEASURE);
                             

/////////
U = ArrayXXcd::Constant(lmax*2+2,kmax*2+2,0);

fftw_execute_dft(plan, (fftw_complex*) TT.data(), (fftw_complex*) T.data());

//~ Runge_Kutta 4
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
	double co = ((double)i+1)/ (double) iter * 100;
	ArrayXXcd ener = sqrt(U*U+W*W).real();
	
	
	
	k1_U = f_U(U,W,T,S);
	k1_W = f_W(U,W,T,S);
	k1_T = f_T(U,W,T,S);
	k1_S = f_S(U,W,T,S);
	
	k2_U = f_U(U + 0.5 * dt * k1_U,W+ 0.5 * dt * k1_W,T+ 0.5 * dt * k1_T, S+ 0.5 * dt * k1_S);
	k2_W = f_W(U + 0.5 * dt * k1_U,W+ 0.5 * dt * k1_W,T+ 0.5 * dt * k1_T, S+ 0.5 * dt * k1_S);
	k2_T = f_T(U + 0.5 * dt * k1_U,W+ 0.5 * dt * k1_W,T+ 0.5 * dt * k1_T, S+ 0.5 * dt * k1_S);
	k2_S = f_S(U + 0.5 * dt * k1_U,W+ 0.5 * dt * k1_W,T+ 0.5 * dt * k1_T, S+ 0.5 * dt * k1_S);

	k3_U = f_U(U + 0.5 * dt * k2_U,W+ 0.5 * dt * k2_W,T+ 0.5 * dt * k2_T, S+ 0.5 * dt * k2_S);
	k3_W = f_W(U + 0.5 * dt * k2_U,W+ 0.5 * dt * k2_W,T+ 0.5 * dt * k2_T, S+ 0.5 * dt * k2_S);
	k3_T = f_T(U + 0.5 * dt * k2_U,W+ 0.5 * dt * k2_W,T+ 0.5 * dt * k2_T, S+ 0.5 * dt * k2_S);
	k3_S = f_S(U + 0.5 * dt * k2_U,W+ 0.5 * dt * k2_W,T+ 0.5 * dt * k2_T, S+ 0.5 * dt * k2_S);

	k4_U = f_U(U + dt * k3_U,W+ dt * k3_W,T+ dt * k3_T, S+ dt * k3_S);
	k4_W = f_W(U + dt * k3_U,W+ dt * k3_W,T+ dt * k3_T, S+ dt * k3_S);
	k4_T = f_T(U + dt * k3_U,W+ dt * k3_W,T+ dt * k3_T, S+ dt * k3_S);
	k4_S = f_S(U + dt * k3_U,W+ dt * k3_W,T+ dt * k3_T, S+ dt * k3_S);

	U = U +(dt/6)*(k1_U+2*k2_U+2*k3_U+k4_U);
	W = W +(dt/6)*(k1_W+2*k2_W+2*k3_W+k4_W);
	T = T +(dt/6)*(k1_T+2*k2_T+2*k3_T+k4_T);
	S = S +(dt/6)*(k1_S+2*k2_S+2*k3_S+k4_S);
	
	ndivU = U - (ProjKK*U + ProjKL*W);
	ndivW = W - (ProjKL*U + ProjLL*W);
	
	U = ndivU;
	W = ndivW;
	double divu = (U*K+W*L).real().sum();
	ArrayXXcd ener2 = sqrt(U*U+W*W).real();
	double grow = (log(ener2/ener)/dt).real().maxCoeff();
	energy << ener2.real().sum() << "," << grow << "\n" ;
	cout << "\r" << co  << "%" << "  Energy : " << ener2.sum()  << "Grow : " << grow << "   div : " << divu << flush;

}
}

//~ else if ( method.compare("RK2")  == 0){
	//~ cout << "Pr= " << Pr << endl << "tau= " << tho << endl << "rrho= " << Rrho << endl;
	//~ ArrayXXcd ndivU,ndivW,kU,kW,kT,kS;
	
	//~ std::ofstream energy;
	//~ energy.open ("energy.txt");
	//~ energy << "Energy, grow \n";
	
	//~ for (int i=0; i<iter; ++i){
		//~ double co = ((double)i+1)/ (double) iter * 100;
		//~ ArrayXXcd ener = (W*W+U*U);
		
		//~ kU = U + dt/2*f_U(U,W,T,S);
		//~ kW = W + dt/2*f_W(U,W,T,S);
		//~ kT = T + dt/2*f_T(U,W,T,S);
		//~ kS = S + dt/2*f_S(U,W,T,S);
		//~ U = U+dt*f_U(kU,kW,kT,kS);
		//~ W = W+dt*f_W(kU,kW,kT,kS);
		//~ T = T+dt*f_T(kU,kW,kT,kS);
		//~ S = S+dt*f_S(kU,kW,kT,kS);
		
		//~ ndivU = U - (ProjKK*U + ProjKL*W);
		//~ ndivW = W - (ProjKL*U + ProjLL*W);
		//~ U = ndivU;
		//~ W = ndivW;
		//~ double divu = (U*K+W*L).sum();
		//~ ArrayXXcd ener2 = (W*W+U*U);
		//~ double grow = (log((ener2-ener)/ener)/dt).sum();
		//~ energy << ener2 << "," << grow << "\n" ;
		//~ cout << "\r" << co  << "%" << "  Energy : " << ener2.sum()  << "Grow : " << grow << "   div : " << divu << flush;
//~ }
//~ }


write_binary("U.dat",U);
write_binary("W.dat",W);
write_binary("T.dat",T);
write_binary("S.dat",S);
cout << endl << "Saved Successfully" << endl;

std::ofstream file("T.txt");
if (file.is_open()){
	file << T ;
}

std::ofstream fileS("S.txt");
if (fileS.is_open()){
	fileS << S;
}

std::ofstream file4("W.txt");
if (file4.is_open()){
	file4 << W;
}

return 0;
}


