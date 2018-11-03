/*
** Created by: Jason Orender
** (c) all rights reserved
**
** This program implements a logistic regression algorithm using a sparse vector format. For
** very small vectors this incurs a fairly massive penalty (about 22x).  However, for extremely
** large vectors, there is both a speed and memory advantage if most of the entries in the
** vector are zeroes.
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <math.h>
#include <list>
#include <cstring>

using namespace std;

// cache size for Svect
#define CSZ 32

/*
** Datapoint is a struct that contains the data for Svect.
*/
struct Datapoint {
public:
  Datapoint()         { d = 0.0; }  // default constructor initializes as zero
  Datapoint(double f) { d = f; }    // alternate constructor initializes with supplied value
  ~Datapoint()        { }           // destructor does nothing

  double d;                         // the double precision element of the data
  int    i;                         // the index of the data (like an array index)
  
};

/*
** Svect is a class set up to store and manipulate a double precision vector.
*/
class Svect {
public:
  Svect(void);                       // default constructor
  Svect(int);                        // alternate constructor
  Svect(const Svect&);               // copy constructor
  ~Svect();                          // destructor

  double  element(int) const;        // gets the value of a specific element in the vector
  double& element_c(int);            // returns a reference to an element's data
  void    setall(double);            // sets all elements of a vector to the argument value
  void    sete(int,double);          // sets a specific element to the argument value
  void    sete(list<Datapoint>::iterator&, double);
  int     size(void) const;          // gets the size of the vector

  Svect& operator*=(const double);   // multiplies the vector by a constant
  Svect  operator*(const double);    // multiplies a vector by a constant
  Svect& operator*=(const Svect&);   // multiplies the vector element-by-element with another
  Svect  operator*(const Svect&);    // multiplies two vectors element-by-element
  Svect& operator+=(const Svect&);   // adds the vector element-by-element with another
  Svect  operator+(const Svect&);    // adds two vectors together element-by-element
  Svect& operator-=(const Svect&);   // subtracts another vector element-by-element from this one  
  Svect  operator-(const Svect&);    // subtracts two vectors element-by-element
  Svect& operator=(const double);    // sets all elements of a vector to a specific value
  Svect& operator=(const Svect&);    // sets the elements to the same as those of another
  double operator[](int) const;      // allows accessing an individual element via brackets
  double& operator[](int);           // allows setting an individual element via brackets

  bool   is_explicit(int) const;     // returns whether an element is explicitly present
  int    count_explicit(void) const; // returns the number of explicit entries in the list
  void   remove(int);                // removes an explicit element (sets it to zero)
  list<Datapoint>::iterator remove(list<Datapoint>::iterator&);
  bool   resize(int);                // discards the data and sets the vector size to a new value
  bool   copy(const Svect&);         // copies the data from an input vector to this one
  double sum(void);                  // returns the summation of all elements of this vector
  void   exp_elem(void);             // takes the exponential function of every element
  void   apply_threshold(double);    // sets values >= threshold to 1 and < threshold to 0

  friend ostream& operator<<(ostream&,const Svect&); // outputs all elements to a stream
  friend istream& operator>>(istream&, Svect&);      // inputs n elements from a stream

private:
  list<Datapoint> a;                 // the list containing the data for the vector
  int     sz;                        // the size of the vector
  list<Datapoint>::iterator cache_data[CSZ];  // cache for speedier access
  int                       cache_index[CSZ];
};

#define TP 0 // true positive
#define FP 1 // false positive
#define FN 2 // false negative
#define TN 3 // true negative

// read_input reads the input files and stores the data in the Svect class
bool   read_input(char **, Svect&, Svect&, Svect**, bool=true);
// calculates the gradient from a weight, y-observations, and features
void   grad(Svect&, Svect&, Svect&, Svect*);
void   gradest(Svect&, Svect&, Svect&, Svect*);
// calculates the objective function
void   LLcomp(Svect&, Svect&, Svect&, Svect*);
double LL(Svect&, Svect&, Svect*);
// calculate the next set of weights
void   getnext(Svect&, Svect&, double);
// iterate to a solution: 
// getsoln(weights, y-objserved, features, delta-criteria, max-iterations)
int    getsoln(Svect&, Svect&, Svect*, double=0.001, int=100);
// the predictive function
void   pred(Svect&, Svect&, Svect*);
// calculates True Positives (TP), False Positives (FP), True Negatives(TN),
// and False Negatives (FN)
void   calc_conf(Svect&, Svect&, Svect&);
// randomly splits the x-matrix (features) into two subsets of the data
// xvec_split(input-xmatrix, input-ymatrix, fraction-to-matrix1, input-size, 
//            x-matrix1-output, x-matrix2-output, y-matrix1-output,
//            y-matrix2-output)
int xmat_split(Svect*, Svect&, double, Svect**, Svect**, Svect&, Svect&);
// outputs the FPR/TPR data in a format that will allow the custom html file
// to display it as a graphic
void outdata(Svect&, Svect&, string); 

int main(int argv, char **argc) {

  cout << endl << "Executing logr with command line arguments: " << endl << "  ";
  for (int i=0; i<argv; i++)
	cout << argc[i] << " ";
  cout << endl;

  /*
  Svect test1(5), test2(5);

  cout << "enter 5 values: ";
  cin  >> test1;
  cout << "you entered: " << test1 << endl;
  test2 = test1;
  cout << "after assignment, test2 = " << test2 << endl;
  test1[2] = 234;
  test1[4] = 432;
  cout << "after changing, test1 = " << test1 << endl;
  */  
  Svect    wvec;            // weights input file
  Svect    yvec;            // observed y-values
  Svect    yvec1;           // observations subset #1
  Svect    yvec2;           // observations subset #2
  Svect   *xvec  = nullptr; // features (array size is number of examples supported)
  Svect   *xvec1 = nullptr; // features subset #1
  Svect   *xvec2 = nullptr; // features subset #2
  Svect    lvec;            // reults (liklihood) with threshold applied
  Svect    llvec;           // log liklihood components
  Svect    cvec;            // confusion vector
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
      cout << "  Training data: " << yvec2.size() << " examples" << endl;
      cout << "  Testing data : " <<  yvec1.size() << " examples" << endl;
      niter = getsoln(wvec, yvec2, xvec2, 0.001, 10000);
      cout << endl << "Solution after " << niter << " iterations: " << endl;
      cout << wvec << endl << endl;

      cout << "Observed training y-values:" << endl 
	   << setprecision(0) << fixed << yvec2 << endl << endl;
      cout << "Training Results (liklihood):" << endl;

      lvec.resize(yvec2.size());
      pred(lvec, wvec, xvec2);
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

      cout << "  *********** TESTING ROC CURVE DATA ************" << endl;
      cout << "  ***********************************************" << endl << endl;
      cout << "  Threshold   FPR    TPR    Distance From Optimal" << endl;
      cout << "  =========   =====  =====  =====================" << endl;
      cout << "      0.000   1.00   1.00   1.00" << endl;

      llvec.resize(yvec1.size());
      pred(llvec, wvec, xvec1);

      thr = 0.0;
      optDIST = 1.0;
      for (int i=0; i<50; i++) {
	thr += 0.01998;
	lvec = llvec;
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
int getsoln(Svect &w, Svect &y, Svect *x, double epsilon, int maxiter) {
  int    i;            // counter
  double ll, ll_old;   // objective function values
  double alpha = 0.001;// speed at which to converge using the gradient
  Svect  dk(w.size()); // temp variable for the gradient

  ll = ll_old = 0.0;
  for (i=0; i<maxiter; i++) {
    grad(dk, w, y, x);
    //cout << "dk = " << dk << endl;
    //cout << "w = " << w <<endl;
    //cout << "y = " << y << endl;
    //cout << "x = " << x[i] << endl;
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
void calc_conf(Svect &conf, Svect &yo, Svect &yc) {
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

int xmat_split(Svect *x_input, Svect &y_input, double fract1, 
			   Svect **xout1, Svect **xout2, Svect &yout1,
			   Svect &yout2) {
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

	*xout1 = new Svect[n1];
	*xout2 = new Svect[n2];
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
** the Svect class for use in the solution convergence algorithm.
*/
bool read_input(char **argc, Svect &w, Svect &y, Svect **x, bool verbose) {
  ifstream infile;
  int      nFeatures, nExamples;

  // reading in initial weights file
  infile.open(argc[1]);
  if (infile.is_open()) {
	if (verbose) cout << "Reading in data files..." << endl;
	
	infile >> nFeatures >> nExamples;
	cout << "  (" << nFeatures << " features, " 
		 << nExamples << " examples)" << endl;
	*x = new Svect[nExamples];
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


void outdata(Svect &llvec, Svect &yvec, string fname) {
  ofstream outfile;    // output file
  double   thr = 0.0;  // threshold
  double   tpr, fpr;   // temp variables for true/false postive rates
  double   dist;       // distance from optimal
  Svect    lvec, cvec; // liklihood and counter vectors

  outfile.open(fname);
  if (outfile.is_open()) {
    outfile << "function setdata() {" << endl;
    outfile << "var inputvar = " << endl;
    outfile << "[ [ 1.000, 1.000, 1.000 ]," << endl;
    for (int i=0; i<50; i++) {
      thr += 0.01998;
      lvec = llvec;
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

void grad(Svect &ret, Svect &w, Svect &y, Svect *x) {
  double wTx, f;
  Svect  a, c;

  ret.resize(w.size());
  for (int i=0; i<y.size(); i++) {
    a    = x[i] * y[i];
    wTx  = (w * x[i]).sum();
    f    = exp(wTx)/(1+exp(wTx));
    c    = x[i] * f;
    ret += (a - c);
  }
}

void gradest(Svect &ret, Svect &w, Svect &y, Svect *x) {
  double wTx, alpha, x1, x2, y1, y2, l1, l2;
  Svect  w1, w2;

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
void getnext(Svect &w, Svect &dk, double speed) {
  Svect  wold(w);

  // each element of dk is a slope pointed toward the minimum objective
  // function value
  w = wold + dk*speed;
  //cout << "wold = " << wold << endl;
  //cout << "dk = " << dk << endl;
  //cout << "speed = " << speed << endl;
  //cout << "w = " << w << endl;

}

/*
** Calculate the components of the Log Liklihood (LL) objective function.
*/
void LLcomp(Svect &l, Svect &w, Svect &y, Svect *x) {
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
double LL(Svect &w, Svect &y, Svect *x) {
  Svect ret;
  LLcomp(ret, w, y, x);
  return ret.sum();
}

/*
** The predictive function. "y" is the output in this case.
*/
void pred(Svect &y, Svect &w, Svect *x) {
  double wTx;

  for (int i=0; i<y.size(); i++) {
    wTx  = (w * x[i]).sum();
    y[i] = exp(wTx)/(1 + exp(wTx));
  }
}

/*
******************************************************************************
******************** Svect CLASS DEFINITION BELOW HERE ***********************
******************************************************************************
*/

/*
** Overloading the "<<" operator allows outputing the elements to an output stream.
*/

ostream& operator<<(ostream& os, const Svect& v) {
  os << "[ ";
  for (int i=0; i<v.size(); i++) os << v.element(i) << " ";
  os << "]";
  return os;
}


/*
** Overloading the ">>" operator allows inputing the elements from an input stream.
*/

istream& operator>>(istream& is, Svect& v) {
  double d;
  for (int i=0; i<v.size(); i++) {
    is >> d;                           // getting the input value
    if (d != 0.0) v.element_c(i) =  d; // only explicitly add the value if it is nonzero
  }
  return is;
}


/*
** Default constructor.
*/
inline Svect::Svect(void) { resize(0); }

/*
** Alternate constructor.
*/
inline Svect::Svect(int n) { resize(n); }

/*
** Copy constructor.
*/
inline Svect::Svect(const Svect &v) { copy(v); }

inline Svect::~Svect(void) {
  // explicitly erasing the list holding the data before destroying the object
  a.erase(a.begin(),a.end());
 }

/*
** The is_explicit() function returns whether the element is explicitly present in the
** list.  It will only be explicitly present if the value is nonzero.
*/
bool Svect::is_explicit(int n) const {
  list<Datapoint>::const_iterator it;
  it = a.begin();
  while (it != a.end()) { if ((*it).i == n) return true; it++; } // return true if the index is found
  return false;                                                  // return false if it is not
}

/*
** The count_explicit() function returns the number of elements that are explicitly 
** present in the list.
*/
int Svect::count_explicit(void) const { return a.size(); }

/*
** The remove() function removes an explicit element from the list, which essentially
** sets it to zero since when that element is referenced in the future a zero will
** be returned. If the element doesn't exist, it does nothing.
*/
void Svect::remove(int n) {
  int i=0;
  list<Datapoint>::iterator it;

  if (cache_index[n%CSZ] == n) { 
    a.erase(cache_data[n%CSZ]); 
    cache_index[n%CSZ] = -1; 
    return; 
  } // end if (cache_index)
  else {
    it = a.begin();
    while (it != a.end()) { 
      if ((*it).i == n) { a.erase(it); return; }
      it++; 
    } // end while (it)

  } // end else (cache_index)

} // end remove()

// this version takes an iterator as input
list<Datapoint>::iterator Svect::remove(list<Datapoint>::iterator &it) {
  if (cache_index[(*it).i%CSZ] == (*it).i) cache_index[(*it).i%CSZ] = -1;
  return a.erase(it);
}

/*
** The element() function extracts an element of a specific ordinal from the list.  It
** is meant to simulate the behavior of an array from the user perspective.
*/
double Svect::element(int n) const { 
  list<Datapoint>::const_iterator it;

  if (cache_index[n%CSZ] == n) return (*(cache_data[n%CSZ])).d;
  else {
    it = a.begin();
    while (it != a.end()) { 
      if ((*it).i == n) {
	return (*it).d; 
      } // end if (it)
      it++; 
    } // end while (it)
  } // end else (cache_index)

  return 0.0; // always returns zero if a corresponding index was not found
} // end element()

/*
** The element_c() function is like the element() function, but it creates a list entry
** if one was not found and passes back the reference.
*/
double& Svect::element_c(int n) { 
  list<Datapoint>::iterator it;
  Datapoint                 dp;

  if (cache_index[n%CSZ] == n) return (*(cache_data[n%CSZ])).d;
  else {
    it = a.begin();
    while (it != a.end()) { 
      if ((*it).i == n) {
	cache_index[n%CSZ] = n;
	cache_data[n%CSZ]  = it;
	return (*it).d;
      } // end if (it)
      it++; 
    } // end while (it)

    dp.i = n;
    a.push_front(dp);
    cache_index[n%CSZ] = n;
    cache_data[n%CSZ]  = a.begin();
    return ((*a.begin()).d); // returns a new element if the index was not found

  } // end else (cache_index)

} // end element_c()

/*
** The setall() function sets every element explicitly to an input value. Note that this eliminates
** sparsity.
*/
void Svect::setall(double d) {
  Datapoint dp;
  resize(sz);           // the easiest way is to simply start from scratch since none of the data
                        // will be retained

  if (d != 0) {         // this only needs to be done if the input value is nonzero
    dp.d = d;
    for (int i=0; i<sz; i++) {
      dp.i = i;
      a.push_front(dp); // need to create a new element for each entry (eliminates sparsity)
      cache_index[i%CSZ] = i;
      cache_data[i%CSZ] = a.begin();
    }
  }
}

/*
** The sete() function sets a specific element to an input value.  If the subscript is out of range,
** it does nothing.
*/
inline void Svect::sete(int i,double d) { if (i < sz) element_c(i) = d; }

/*
** This version of set uses an iterator instead of an index.
*/
void Svect::sete(list<Datapoint>::iterator &it, double d) {
  (*it).d = d;
  cache_index[(*it).i%CSZ] = (*it).i;
  cache_data[(*it).i%CSZ]  = it;
}


/*
** The size() function returns the nominal size of the vector.
*/
inline int Svect::size(void) const { return sz; }

/*
** The "*=" operator when used with two vectors multiplies each of the vectors
** together element-by-element.  This does not correspond to a true matrix multiplication.
** If the vectors are not of equal size, it does nothing.
*/
Svect& Svect::operator*=(const Svect &v) {
  double d;
  int    i;
  list<Datapoint>::iterator it;

  if (v.size() == sz) {	
    it = a.begin();
    while (it != a.end()) {
      i = (*it).i;
      d = (*it).d * v[i];
      //if (d != 0.0) { (*it).d = d; it++; }
      //else            it = a.erase(it);

      if (d != 0.0) { sete(it,d); it++; }
      else            it = remove(it);
    } // end while(it)
  } // end if (v)

  return *this;
}

/*
** This version of the "*=" unary operator simply multiplies every element in the
** vector by a constant.
*/
Svect& Svect::operator*=(const double f) {
  list<Datapoint>::iterator it;
  double d;

  if (f != 0.0) {
    it = a.begin();
    while (it != a.end()) { 
      d = (*it).d * f;
      sete(it, d); 
      it++; 
    }
  } // end if (f)
  else resize(sz); // this is the same as removing all explicit elements

  return *this;
} // end "*=" operator definition

/*
** This version of the "*" operator multiplies a vector by a constant.
*/
Svect Svect::operator*(const double d) {
  Svect vreturn(*this);
  vreturn *= d;
  return vreturn;
} // end "*" operator definition


/*
** This version of the  "*" operator multiplies two vectors together element-by-element. 
** If the vectors are not of equal size, it returns the vector on the lhs of the "*".
*/
Svect Svect::operator*(const Svect &v) {
  Svect vreturn(*this);
  vreturn *= v;
  return vreturn;
} // end "*" operator definition

/*
** The "+=" operator when used with two vectors adds another vector element-by-element.
** to this one. If the vectors are not of equal size, it does nothing.
*/
Svect& Svect::operator+=(const Svect &v) {
  int    i;
  list<Datapoint>::const_iterator itc;

  if (v.size() == sz) {	
    itc = v.a.begin();
    while (itc != v.a.end()) {
      i = (*itc).i;
      element_c(i) += (*itc).d;
      itc++;
    } // end while (it)

  } // end if (v)

  return *this;
} // end "+=" operator definition

/*
** The "+" operator adds two vectors together element-by-element. If the vectors are
** not of equal size, it returns the vector on the lhs of the "+".
*/
Svect Svect::operator+(const Svect &v) {
  Svect vreturn(*this);
  vreturn += v;
  return vreturn;
} // end "+" operator defnition

/*
** The "-=" operator when used with two vectors subtracts another vector element-by-element.
** from this one. If the vectors are not of equal size, it does nothing.
*/
Svect& Svect::operator-=(const Svect &v) {
  int i;
  list<Datapoint>::const_iterator itc;

  if (v.size() == sz) {	
    itc = v.a.begin();
    while (itc != v.a.end()) {
      i = (*itc).i;
      element_c(i) -= (*itc).d;
      itc++;
    } // end while (it)

  } // end if (v)

  return *this;
} // end "-=" operator definition

/*
** The "-" operator subtracts two vectors element-by-element. If the vectors are
** not of equal size, it returns the vector on the lhs of the "-".
*/
Svect Svect::operator-(const Svect &v) {
  Svect vreturn(*this);
  vreturn -= v;
  return vreturn;
} // end "-" operator definition

/*
** This assignment operator uses the copy() function to copy from one vector to another
** as long as they are the same size.  Otherwise it does nothing.
*/
Svect& Svect::operator=(const Svect &v) { copy(v); return *this; }

/*
** This assignment operator uses the setall() function to copy a double to every element
** in the vector.
*/
Svect& Svect::operator=(const double d) { setall(d); return *this; }

/*
** The bracket ("[]") operator allows accessing an individual element in the vector. The first
** version is the "get" function, and the second version is the "set" function.
*/
double Svect::operator[](int i) const{
  if (i < sz) return element(i); else return element(sz-1);
} // end "[]" (get) operator definition
double& Svect::operator[](int i) {
  if (i < sz) return element_c(i); else return element_c(sz-1);
} // end "[]" (set) operator definition


/*
** The resize() function resizes the vectors and destroys the data (sets to zero).
*/
bool Svect::resize(int n) {
  // if ensure that the list is empty
  a.erase(a.begin(),a.end());
  // set the new size
  sz = n;
  // zero out the cache
  memset(cache_index, -1, CSZ*sizeof(int));

  return true; // this basic case always returns true
} // end resize()

/*
** The copy() function copies the data of one vector to another and returns "true"
** if they are the same size.  Otherwise, it does nothing and returns "false".
*/
bool Svect::copy(const Svect &v) {
  list<Datapoint>::const_iterator it;
  Datapoint dp;

  // resetting this vector size to the new vector size
  resize(v.sz);

  // copying the list data
  it = v.a.begin();
  while (it != v.a.end()) { 
    dp.i = (*it).i; 
    dp.d = (*it).d;     
    a.push_front(dp);
    cache_index[dp.i%CSZ] = dp.i;
    cache_data[dp.i%CSZ]  = a.begin();
    it++; 
  }

  // copying the cache data
  //memcpy(cache_index, v.cache_index, CSZ*sizeof(int));
  //memcpy(cache_data, v.cache_data, CSZ*sizeof(list<Datapoint>::iterator));
  /*
  for (int i=0; i<CSZ; i++) {
    cache_index[i] = v.cache_index[i];
    cache_data[i]  = v.cache_data[i];
  }
  */
  return true;  

} // end copy()

/*
** The sum() function returns a summation of all elements in a vector.
*/
double Svect::sum(void) {
  list<Datapoint>::iterator it;
  double sum=0.0;

  it = a.begin();
  while (it != a.end()) {
    sum += (*it).d;
    it++;
  } // end while (it)

  return sum;
} // end sum()

/*
** The exp() function takes the exponential function of every element.  Note that
** zeroes will potentially become explicit elements.
*/
void Svect::exp_elem(void) {
  double d;

  for (int i=0; i<sz; i++) {
    d = exp(element(i));
    if (d > 0.00001) element_c(i) = d; else remove(i);
  } // end for (i)

} // exp_elem()

/*
** The apply_threshold() function sets values greater than or equal to
** the threshold to one and values less than the threshold to zero. The 
** threshold supplied must be greater than zero and less than or equal
** to one.
*/
void Svect::apply_threshold(double f) {
  list<Datapoint>::iterator it;
  double d;

  if ((f > 0.0) && (f <= 1.0)) {

    it = a.begin();
    while (it != a.end()) {
      //if ((*it).d >= f) { (*it).d = 1.0; it++; }
      //else                it = a.erase(it);
      if ((*it).d >= f) { sete(it, 1.0); it++; }
      else                it = remove(it);
    } // end while (it)

  } // end if (f)

} // end apply_threshold()
