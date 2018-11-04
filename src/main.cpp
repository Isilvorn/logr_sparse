/*
** Created by: Jason Orender
** (c) 2018 all rights reserved
**
** This program implements a logistic regression algorithm using a sparse vector format. Since
** it uses the list container to ensure that vectors of virtually any size can be used and resized
** at will, it incurs a fairly massive penalty (about 15x) over the static version for smaller
** vectors.  However, for extremely large vectors, there is both a speed and memory advantage if 
** most of the entries in the vector are zeroes.  The cache system also reduces some of the speed 
** disadvantage for smaller vectors.
*/

#include <iostream>
#include <iomanip>
#include <fstream>
#include <random>
#include <math.h>
/*
#include <list>
#include <cstring>
*/
#include "../include/vect.h"

using namespace std;

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
  int    szw;          // the size of the w vector
  double ll, ll_old;   // objective function values
  double wTx, f;
  double alpha = 0.001;// speed at which to converge using the gradient
  Svect  dk(w.size()); // temp variable for the gradient

  ll = ll_old = 0.0;
  szw = w.size();
  for (i=0; i<maxiter; i++) {
    ll_old = ll;
    // calculating the gradient of the logistic function
    dk.resize(szw);
    for (int i=0; i<y.size(); i++) {
      wTx  = (w * x[i]).sum();
      f    = exp(wTx)/(1+exp(wTx));
      dk  += (x[i] * (y[i] - f));            // gradient update
      ll  += y[i] * wTx - log(1 + exp(wTx)); // log-liklihood
    }

    if (fabs(ll_old-ll) < epsilon) break;
    w += dk*alpha;
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

/*
** The grad() function calculates the logistic function gradient with respect to "w".
** It has been optimized to reduce the number of vector operations.
*/
void grad(Svect &ret, Svect &w, Svect &y, Svect *x) {
  double wTx, f;

  ret.resize(w.size());
  for (int i=0; i<y.size(); i++) {
    wTx  = (w * x[i]).sum();
    f    = exp(wTx)/(1+exp(wTx));
    ret += (x[i] * (y[i] - f));
  }
}

/*
** The gradest() function is an alternative to the grad() function, which estimates
** the gradient by calculating a slope between two points on the logistic function
** separated by a very small differential.  This is just used to test the results
** of the grad() function since the grad function is much faster and more precise.
*/
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

