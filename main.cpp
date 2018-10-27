#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <math.h>

using namespace std;

/*
** Dvect is a class set up to store and manipulate a double precision vector.
*/
class Dvect {
public:
  Dvect(void);                     // default constructor
  Dvect(int);                      // alternate constructor
  Dvect(const Dvect&);             // copy constructor
  ~Dvect();                        // destructor

  double element(int) const;       // gets the value of a specific element in the vector
  void   setall(double);           // sets all elements of a vector to the argument value
  void   set(int,double);          // sets a specific element to the argument value
  int    size(void) const;         // gets the size of the vector

  Dvect& operator*=(const double); // multiplies the vector by a constant
  Dvect  operator*(const double);  // multiplies a vector by a constant
  Dvect& operator*=(const Dvect&); // multiplies the vector element-by-element with another
  Dvect  operator*(const Dvect&);  // multiplies two vectors element-by-element
  Dvect& operator+=(const Dvect&); // adds the vector element-by-element with another
  Dvect  operator+(const Dvect&);  // adds two vectors together element-by-element
  Dvect& operator-=(const Dvect&); // subtracts another vector element-by-element from this one  
  Dvect  operator-(const Dvect&);  // subtracts two vectors element-by-element
  Dvect& operator=(const double);  // sets all elements of a vector to a specific value
  Dvect& operator=(const Dvect&);  // sets the elements to the same as those of another
  double operator[](int) const;    // allows accessing an individual element via brackets
  double& operator[](int);         // allows setting an individual element via brackets

  bool   resize(int);              // discards the data and sets the vector size to a new value
  bool   copy(const Dvect&);       // copies the data from an input vector to this one
  double sum(void);                // returns the summation of all elements of this vector
  void   exp_elem(void);           // takes the exponential function of every element
  void   apply_threshold(double);  // sets values >= threshold to 1 and < threshold to 0

  friend ostream& operator<<(ostream&,const Dvect&); // outputs all elements to a stream
  friend istream& operator>>(istream&, Dvect&);      // inputs n elements from a stream

private:
  double *a;                       // the array for the vector data
  int     sz;                      // the size of the vector
};

#define TP 0 // true positive
#define FP 1 // false positive
#define FN 2 // false negative
#define TN 3 // true negative

// read_input reads the input files and stores the data in the Dvect class
bool   read_input(char **, Dvect&, Dvect&, Dvect**, bool=true);
// calculates the gradient from a weight, y-observations, and features
void   grad(Dvect&, Dvect&, Dvect&, Dvect*);
void   gradest(Dvect&, Dvect&, Dvect&, Dvect*);
// calculates the objective function
void   LLcomp(Dvect&, Dvect&, Dvect&, Dvect*);
double LL(Dvect&, Dvect&, Dvect*);
// calculate the next set of weights
void   getnext(Dvect&, Dvect&, double);
// iterate to a solution: 
// getsoln(weights, y-objserved, features, delta-criteria, max-iterations)
int    getsoln(Dvect&, Dvect&, Dvect*, double=0.001, int=100);
// calculates True Positives (TP), False Positives (FP), True Negatives(TN),
// and False Negatives (FN)
void   calc_conf(Dvect&, Dvect&, Dvect&);
// randomly splits the x-matrix (features) into two subsets of the data
// xvec_split(input-xmatrix, input-ymatrix, fraction-to-matrix1, input-size, 
//            x-matrix1-output, x-matrix2-output, y-matrix1-output,
//            y-matrix2-output)
int xmat_split(Dvect*, Dvect&, double, Dvect**, Dvect**, Dvect&, Dvect&);
// outputs the FPR/TPR data in a format that will allow the custom html file
// to display it as a graphic
void outdata(Dvect&, Dvect&, string); 

int main(int argv, char **argc) {

  cout << endl << "Executing logr with command line arguments: " << endl << "  ";
  for (int i=0; i<argv; i++)
	cout << argc[i] << " ";
  cout << endl;

  Dvect    wvec;            // weights input file
  Dvect    yvec;            // observed y-values
  Dvect    yvec1;           // observations subset #1
  Dvect    yvec2;           // observations subset #2
  Dvect   *xvec  = nullptr; // features (array size is number of examples supported)
  Dvect   *xvec1 = nullptr; // features subset #1
  Dvect   *xvec2 = nullptr; // features subset #2
  Dvect    lvec;            // reults (liklihood) with threshold applied
  Dvect    llvec;           // log liklihood components
  Dvect    cvec;            // confusion vector
  int      niter;           // number of iterations used
  int      n1;              // the size of the subset matrix #1
  double   logl;            // the aggregate log-liklihood
  double   thr;             // threshold
  double   tpr, fpr;        // true positive rate, false positive rate
  double   optTPR, optFPR;  // optimal tpr/fpr pair
  double   optTHR;          // optimal threshold
  double   dist, optDIST;   // distance from optimal

  if (argv == 4) {
	if (read_input(argc, wvec, yvec, &xvec, false)) {
	  n1 = xmat_split(xvec, yvec, 0.333, &xvec1, &xvec2, yvec1, yvec2);
	  cout << "  Training data: " << (yvec.size()-n1) << " examples" << endl;
	  cout << "  Testing data : " <<  n1 << " examples" << endl;
	  niter = getsoln(wvec, yvec2, xvec2, 0.001, 10000);
	  cout << endl << "Solution after " << niter << " iterations: " << endl;
	  cout << wvec << endl << endl;

	  cout << "Observed training y-values:" << endl 
		   << setprecision(0) << fixed << yvec2 << endl << endl;
	  cout << "Training Results (liklihood):" << endl;
	  LLcomp(llvec, wvec, yvec2, xvec2);
	  lvec = llvec;
	  logl = lvec.sum();
	  lvec.exp_elem();
	  lvec.apply_threshold(0.999);
	  cout << setprecision(0) << fixed << lvec << endl;
	  calc_conf(cvec,yvec2,lvec);
	  cout << endl << "Training Confusion numbers:" << endl;
	  cout << setprecision(1) << fixed;
	  cout << "  TP: " << setw(5) << 100*(cvec[TP]/(cvec[TP]+cvec[FP])) 
		   << "% (" << (int)cvec[TP] << ")" << endl;
	  cout << "  FP: " << setw(5) << 100*(cvec[FP]/(cvec[TP]+cvec[FP])) 
		   << "% (" << (int)cvec[FP] << ")" << endl;
	  cout << "  TN: " << setw(5) << 100*(cvec[TN]/(cvec[TN]+cvec[FN])) 
		   << "% (" << (int)cvec[TN] << ")" << endl;
	  cout << "  FN: " << setw(5) << 100*(cvec[FN]/(cvec[TN]+cvec[FN])) 
		   << "% (" << (int)cvec[FN] << ")" << endl;
	  cout << "             =====" << endl;
	  cout << "              " << (int)(cvec[TP]+cvec[FP]+cvec[FN]+cvec[TN]) << endl << endl;

	  LLcomp(llvec, wvec, yvec1, xvec1);

	  cout << "  *********** TESTING ROC CURVE DATA ************" << endl;
	  cout << "  ***********************************************" << endl << endl;
	  cout << "  Threshold   FPR    TPR    Distance From Optimal" << endl;
	  cout << "  =========   =====  =====  =====================" << endl;
	  cout << "      0.000   1.00   1.00   1.00" << endl;

	  thr = 0.0;
	  optDIST = 1.0;
	  for (int i=0; i<50; i++) {
		thr += 0.01998;
		lvec = llvec;
		lvec.exp_elem();
		lvec.apply_threshold(thr);
		calc_conf(cvec,yvec1,lvec);
		tpr = (cvec[TP]/(cvec[TP]+cvec[FN]));
		fpr = (cvec[FP]/(cvec[FP]+cvec[TN]));
		dist = (tpr - 1.0)*(tpr - 1.0) + fpr*fpr;
		if (dist < optDIST) 
		  { optDIST = dist; optTPR = tpr; optFPR = fpr; optTHR = thr; }

		cout << setprecision(3)
			 << "  " << setw(9) << thr
			 << setprecision(2)
			 << "  " << setw(5) << fpr
			 << "  " << setw(5) << tpr
			 << "  " << setw(5) << dist << endl; 
	  }
	  cout << "      1.000   0.00   0.00   1.00" << endl;
	  cout << endl;

	  outdata(llvec, yvec1, "setdata.js");

	  cout << "Optimal threshold: " << setprecision(3) << optTHR << " (TPR = " 
		   << optTPR << ", FPR = " << setprecision(2) << optFPR << ")" << endl; 

	  lvec = llvec;
	  lvec.exp_elem();
	  lvec.apply_threshold(optTHR);
	  calc_conf(cvec,yvec1,lvec);
	  cout << endl;
	  cout << "Observed testing y-values:" << endl 
		   << setprecision(0) << fixed << yvec1 << endl << endl;
	  cout << "Optimal Testing Results (liklihood):" << endl;
	  cout << setprecision(0) << fixed << lvec << endl;
	  cout << endl << "Optimal Testing Confusion numbers:" << endl;
	  cout << setprecision(1) << fixed;
	  cout << "  TP: " << setw(5) << 100*(cvec[TP]/(cvec[TP]+cvec[FP])) 
		   << "% (" << (int)cvec[TP] << ")" << endl;
	  cout << "  FP: " << setw(5) << 100*(cvec[FP]/(cvec[TP]+cvec[FP])) 
		   << "% (" << (int)cvec[FP] << ")" << endl;
	  cout << "  TN: " << setw(5) << 100*(cvec[TN]/(cvec[TN]+cvec[FN])) 
		   << "% (" << (int)cvec[TN] << ")" << endl;
	  cout << "  FN: " << setw(5) << 100*(cvec[FN]/(cvec[TN]+cvec[FN])) 
		   << "% (" << (int)cvec[FN] << ")" << endl;
	  cout << "             =====" << endl;
	  cout << "              " << (int)(cvec[TP]+cvec[FP]+cvec[FN]+cvec[TN]) << endl << endl;
	}
  }
  else {
	cout << "Usage:  ./logr [Initial Weights] [y-data] [x-data]" << endl;
  }

  return 0;
}

/*
** The getsoln() function iterates to a solution until either the objective
** function change from one iteration to the next is less than espsilon, or the
** max number of iterations is reached.
*/
int getsoln(Dvect &w, Dvect &y, Dvect *x, double epsilon, int maxiter) {
  int    i;            // counter
  double ll, ll_old;   // objective function values
  double alpha = 0.001;// speed at which to converge using the gradient
  Dvect  dk(w.size()); // temp variable for the gradient

  ll = ll_old = 0.0;
  for (i=0; i<maxiter; i++) {
	grad(dk, w, y, x);
	ll_old = ll;
	ll     = LL(w, y, x);
    if (fabs(ll_old-ll) < epsilon) break;
	getnext(w, dk, alpha);
  }

  return i;

}

/*
** The gen_conf_matrix() function calculates and displays (to stdout) a
** confusion matrix using the observed y values and the calculated y-values
** as inputs.
*/
void calc_conf(Dvect &conf, Dvect &yo, Dvect &yc) {
  conf.resize(4);
  if (yo.size() == yc.size()) {
	for (int i=0; i<yo.size(); i++) {
	  if ((yo[i] == 1) && (yc[i] == 1)) conf[TP]++; // true positives
	  if ((yo[i] == 0) && (yc[i] == 1)) conf[FP]++; // false positives
	  if ((yo[i] == 1) && (yc[i] == 0)) conf[FN]++; // false negatives
	  if ((yo[i] == 0) && (yc[i] == 0)) conf[TN]++; // true negatives
	}
  }
}

/*
** The xmat_split() function takes an input matrix and randomly splits 
** it into two subsets of the data.  The fraction that goes to the first
** output matrix is given by the second argument.
*/
int xmat_split(Dvect *x_input, Dvect &y_input, double fract1, 
			   Dvect **xout1, Dvect **xout2, Dvect &yout1,
			   Dvect &yout2) {
  int    n1, n2;  // number of examples for matrix #1 & #2
  int    nsize;   // size of input
  int    i, j, k; // counters
  double d;       // discriminator

  // random number generator
  default_random_engine generator; 
  uniform_real_distribution<double> distrib(0.0,1.0);

  if ((fract1 >= 0.0) && (fract1 <= 1.0)) {
	nsize  = y_input.size();
	n1     = (int)(fract1 * nsize);
	n2     = (int)(nsize - n1);

	*xout1 = new Dvect[n1];
	*xout2 = new Dvect[n2];
	yout1.resize(n1);
	yout2.resize(n2);

	j = k = 0;
	for (int i=0; i < nsize; i++) {
	  d = distrib(generator);
	  if ((d < fract1) && (j < n1)) 
		{ (*xout1)[j] = x_input[i]; yout1[j] = y_input[i]; j++; }
	  else 
		{ (*xout2)[k] = x_input[i]; yout2[k] = y_input[i]; k++; }
	} // end for loop (i)

  } // end if (fract1) 

  return n1;
} // end xmat_split()

/*
** The read_input() function reads the input files and stores the data in
** the Dvect class for use in the solution convergence algorithm.
*/
bool read_input(char **argc, Dvect &w, Dvect &y, Dvect **x, bool verbose) {
  ifstream infile;
  int      nFeatures, nExamples;

  // reading in initial weights file
  infile.open(argc[1]);
  if (infile.is_open()) {
	if (verbose) cout << "Reading in data files..." << endl;
	
	infile >> nFeatures >> nExamples;
	cout << "  (" << nFeatures << " features, " 
		 << nExamples << " examples)" << endl;
	*x = new Dvect[nExamples];
	for (int i=0; i<nExamples; i++) (*x)[i].resize(nFeatures);
	w.resize(nFeatures);
	y.resize(nExamples);

	infile >> w;
	infile.close();
	if (verbose) cout << "Initial Weights = " << w << endl;
	  
	infile.open(argc[2]);
	if (infile.is_open()) {
	  infile >> y;
	  infile.close();
	  if (verbose) cout << "Observed y-values: " << endl << y << endl;

	  infile.open(argc[3]);
	  if (infile.is_open()) {
		for (int i=0; i<nExamples; i++) infile >> (*x)[i];
		infile.close();
		if (verbose) cout << "Features:" << endl;
		if (verbose) 
		  for (int i=0; i<nExamples; i++) 
			cout << setw(5) << i << ": " << (*x)[i] << endl;
		}
	  else 
		{ cerr << "Bad input file name (x-data)." << endl; return false; }
	}
	else 
	  { cerr << "Bad input file name (y-data)." << endl; return false; }
  }
  else 
	{ cerr << "Bad input file name (weights)." << endl; return false; }
  
  return true;
}

void outdata(Dvect &llvec, Dvect &yvec, string fname) {
  ofstream outfile;    // output file
  double   thr = 0.0;  // threshold
  double   tpr, fpr;   // temp variables for true/false postive rates
  double   dist;       // distance from optimal
  Dvect    lvec, cvec; // liklihood and counter vectors

  outfile.open(fname);
  if (outfile.is_open()) {
    outfile << "function setdata() {" << endl;
    outfile << "var inputvar = " << endl;
    outfile << "[ [ 1.000, 1.000, 1.000 ]," << endl;
    for (int i=0; i<50; i++) {
      thr += 0.01998;
      lvec = llvec;
      lvec.exp_elem();
      lvec.apply_threshold(thr);
      calc_conf(cvec,yvec,lvec);
      tpr = (cvec[TP]/(cvec[TP]+cvec[FN]));
      fpr = (cvec[FP]/(cvec[FP]+cvec[TN]));
      dist = (tpr - 1.0)*(tpr - 1.0) + fpr*fpr;
      outfile << setprecision(3) << fixed
	      << "  [ " << setw(5) << fpr  << ", " << setw(5) << tpr 
	      << ", " << setw(5) << dist << " ]," << endl;
    }
    outfile << "  [ 0.000, 0.000, 1.000 ] ];" << endl;
    outfile << "return inputvar;" << endl;
    outfile << "}" << endl;
    outfile.close();
  } // end if (outfile) 

} // end outdata()


void grad(Dvect &ret, Dvect &w, Dvect &y, Dvect *x) {
  double wTx, f;
  Dvect  a, c;

  ret.resize(w.size());
  for (int i=0; i<y.size(); i++) {
	a    = x[i] * y[i];
	wTx  = (w * x[i]).sum();
	f    = exp(wTx)/(1+exp(wTx));
	c    = x[i] * f;
	ret += (a - c);
  }

}

void gradest(Dvect &ret, Dvect &w, Dvect &y, Dvect *x) {
  double wTx, alpha, x1, x2, y1, y2, l1, l2;
  Dvect  w1, w2;

  alpha = 0.001;
  w1 = w;
  ret.resize(w.size());

  for (int i=0; i<w.size(); i++) {
	w2    = w1;
	w2[i] = w1[i]*(1-alpha);
	l1 = LL(w1,y,x);
	l2 = LL(w2,y,x);

	// calculating slope
	x1 = w1[i];
	x2 = w2[i];
	y1 = l1;
	y2 = l2;
	ret[i] = (y2 - y1)/(x2 - x1);
  }

}

/*
** Finding the next values for weights by extrapolating each element of the
** gradient to zero, and then moving the current weight in that direction
** at an input speed.  A speed of 1.0 would apply the entire extrapolation,
** while a speed of 0.5 would be half speed, and so on.
*/
void getnext(Dvect &w, Dvect &dk, double speed) {
  Dvect  wold(w);

  // each element of dk is a slope pointed toward the minimum objective
  // function value
  w = wold + dk*speed;

}

/*
** Calculate the components of the Log Liklihood (LL) objective function.
*/
void LLcomp(Dvect &l, Dvect &w, Dvect &y, Dvect *x) {
  double wTx, a, b;

  l.resize(y.size());
  for (int i=0; i<l.size(); i++) {
	wTx  = (w * x[i]).sum();
	l[i] = y[i] * wTx - log(1 + exp(wTx));
  }

}

/*
** The Log Liklihood (LL) objective function.
*/
double LL(Dvect &w, Dvect &y, Dvect *x) {
  Dvect ret;
  LLcomp(ret, w, y, x);
  return ret.sum();
}

/*
******************************************************************************
******************** Dvect CLASS DEFINITION BELOW HERE ***********************
******************************************************************************
*/

/*
** Overloading the "<<" operator allows outputing the elements to an output stream.
*/
ostream& operator<<(ostream& os, const Dvect& v) {
  os << "[ ";
  for (int i=0; i<v.size(); i++) os << v.element(i) << " ";
  os << "]";
  return os;
}

/*
** Overloading the ">>" operator allows inputing the elements from an input stream.
*/
istream& operator>>(istream& is, Dvect& v) {
  for (int i=0; i<v.size(); i++) is >> v[i];
  return is;
}

/*
** Default constructor.
*/
Dvect::Dvect(void) { 
  a  = nullptr;
  sz = 0;
}

/*
** Alternate constructor.
*/
Dvect::Dvect(int n) { 
  // setting "a" to nullptr so that the resize function does not attempt to delete it
  a = nullptr;
  // using the resize function to initialize the vector
  resize(n);
}

/*
** Copy constructor.
*/
Dvect::Dvect(const Dvect &v) {
  // setting "a" to nullptr so that the resize function does not attempt to delete it
  a = nullptr;
  // using the resize function to initialize the vector, if it works copy the input
  if (resize(v.size())) copy(v);
}

Dvect::~Dvect(void) {
  // deallocating the memory set aside for the vector
  if (a != nullptr) delete a;
}

/*
** The following inline functions are simple get/set functions.
*/
inline double Dvect::element(int i) const { return a[i]; }
inline void   Dvect::setall(double d)     { for (int i=0; i<sz; i++) a[i] = d; }
inline void   Dvect::set(int i,double d)  { if (i < sz) a[i] = d; }
inline int    Dvect::size(void) const     { return sz; }

/*
** The "*=" operator when used with two vectors multiplies each of the vectors
** together element-by-element.  This does not correspond to a true matrix multiplication.
** If the vectors are not of equal size, it does nothing.
*/
Dvect& Dvect::operator*=(const Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] *= v.a[i]; }
  return *this;
}

/*
** This version of the "*=" unary operator simply multiplies every element in the
** vector by a constant.
*/
Dvect& Dvect::operator*=(const double d) {
  for (int i=0; i<sz; i++) a[i] *= d;
  return *this;
}

/*
** This version of the "*" operator multiplies a vector by a constant.
*/
Dvect Dvect::operator*(const double d) {
  Dvect vreturn(*this);
  vreturn *= d;
  return vreturn;
}

/*
** This version of the  "*" operator multiplies two vectors together element-by-element. 
** If the vectors are not of equal size, it returns the vector on the lhs of the "*".
*/
Dvect Dvect::operator*(const Dvect &v) {
  Dvect vreturn(*this);
  vreturn *= v;
  return vreturn;
}

/*
** The "+=" operator when used with two vectors adds another vector element-by-element.
** to this one. If the vectors are not of equal size, it does nothing.
*/
Dvect& Dvect::operator+=(const Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] += v.a[i]; }
  return *this;
}

/*
** The "+" operator adds two vectors together element-by-element. If the vectors are
** not of equal size, it returns the vector on the lhs of the "+".
*/
Dvect Dvect::operator+(const Dvect &v) {
  Dvect vreturn(*this);
  vreturn += v;
  return vreturn;
}

/*
** The "-=" operator when used with two vectors subtracts another vector element-by-element.
** from this one. If the vectors are not of equal size, it does nothing.
*/
Dvect& Dvect::operator-=(const Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i] -= v.a[i]; }
  return *this;
}

/*
** The "-" operator subtracts two vectors element-by-element. If the vectors are
** not of equal size, it returns the vector on the lhs of the "-".
*/
Dvect Dvect::operator-(const Dvect &v) {
  Dvect vreturn(*this);
  vreturn -= v;
  return vreturn;
}

/*
** This assignment operator uses the copy() function to copy from one vector to another
** as long as they are the same size.  Otherwise it does nothing.
*/
Dvect& Dvect::operator=(const Dvect &v) {
  resize(v.size());
  copy(v);
  return *this;
}

/*
** This assignment operator uses the setall() function to copy a double to every element
** in the vector.
*/
Dvect& Dvect::operator=(const double d) {
  setall(d);
  return *this;
}

/*
** The bracket ("[]") operator allows accessing an individual element in the vector.
*/
double Dvect::operator[](int i) const{
  if (i < sz) return a[i]; else return a[sz-1];
}

double& Dvect::operator[](int i) {
  if (i < sz) return a[i]; else return a[sz-1];
}


/*
** The resize() function resizes the vectors and destroys the data (sets to zero).
*/
bool Dvect::resize(int n) {
  // if the array is already allocated, deallocate it
  if (a != nullptr) delete a;
  // allocating a new vector ("a" for array)
  a = new double[n];
  // if the allocation was a success, the size is stored in "size"
  // otherwise, size is set to -1
  if (a != nullptr) sz = n; else sz = -1;
  // initializing the new vector with all zeroes
  for (int i=0; i<n; i++) a[i]=0;

  if (sz == -1) return false; else return true;
}

/*
** The copy() function copies the contents of one vector to another and returns "true"
** if they are the same size.  Otherwise, it does nothing and returns "false".
*/
bool Dvect::copy(const Dvect &v) {
  if (v.size() == sz) {	for (int i=0; i<sz; i++) a[i]=v.a[i]; return true;  }
  else                {                                       return false; }
}

/*
** The sum() function returns a summation of all elements in a vector.
*/
double Dvect::sum(void) {
  double sum=0.0;
  for (int i=0; i<sz; i++) sum+=a[i];
  return sum;
}

/*
** The exp() function takes the exponential function of every element.
*/
void Dvect::exp_elem(void) {
  for (int i=0; i<sz; i++) a[i] = exp(a[i]);
}

/*
** The apply_threshold() function sets values greater than or equal to
** the threshold to one and values less than the threshold to zero.
*/
void Dvect::apply_threshold(double d) {
  for (int i=0; i<sz; i++) a[i] = (a[i] >= d)?1.0:0.0;
}
