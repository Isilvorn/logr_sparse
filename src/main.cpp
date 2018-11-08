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

class Datamodule {
public:
  Datamodule();                                                   // default constructor
  ~Datamodule();                                                  // destructor

  bool   read_input(char **, bool = true);                        // read_input reads input files
  void   LLcomp(Svect&, Svect&, Svect&, Svect*);                  // calc objective function components
  double LL(Svect&, Svect&, Svect*);                              // returns the objective function sum
  int    getsoln(double = 0.001, int = 100);                      // iterate to a solution: 
  void   pred(void);                                              // the predictive function
  void   apply_threshold(double = 0.999);                         // apply a threshold limiter to results
  void   calc_conf(double* = nullptr);                            // calculate confusion numbers
  void   outdata(string);                                         // writes TPR/FPR data to a file
  void   set_weights(Datamodule&);                                // copies the weights

  int    examples(void);                                          // returns the number of examples in the dataset
  void   display_weights(int = 4);                                // display the current weights
  void   display_observations(void);                              // display the observations vector
  void   display_features(int = 4);                               // display the features matrix
  void   display_results(void);                                   // display the results vector
  void   display_confusion(void);                                 // display the confusion matrix

  friend int xmat_split(Datamodule&, double, Datamodule&, Datamodule&);

private:
  Svect  wvec;   // weights vector
  Svect *xvec;   // features vector
  Svect  yvec;   // observations vector
  Svect  rvec;   // results vector
  Svect  lvec;   // objective function components
  Svect  tvec;   // threshold limited vector
  Svect  cvec;   // confusion matrix components
};

// randomly splits the matrices (weights, observed, features) into two subsets of the data
// xvec_split(input-data, fraction-to-out1, matrix-out1, matrix-out2)
int xmat_split(Datamodule&, double, Datamodule&, Datamodule&);

int main(int argv, char **argc) {

  cout << endl << "Executing logr with command line arguments: " << endl << "  ";
  for (int i=0; i<argv; i++)
	cout << argc[i] << " ";
  cout << endl;

  /*
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
  double   logl;            // the aggregate log-liklihood
  */

  int      n1;              // the size of the subset matrix #1
  int      niter;           // number of iterations used
  double   thr;             // threshold
  double   dist, optDIST;   // distance from optimal
  double   tpr, fpr;        // true positive rate, false positive rate
  double   optTPR, optFPR;  // optimal tpr/fpr pair
  double   optTHR;          // optimal threshold

  Datamodule entireDS;
  Datamodule trainingDS, testingDS;
  double     res[4];

  if (argv == 4) {
    if (entireDS.read_input(argc, false)) {
      n1 = xmat_split(entireDS, 0.333, testingDS, trainingDS);
      cout << "  Training data: " << trainingDS.examples() << " examples" << endl;
      cout << "  Testing data : " << testingDS.examples()  << " examples" << endl;
      niter = trainingDS.getsoln(0.001, 10000);
      cout << endl << "Solution after " << niter << " iterations: " << endl;
      trainingDS.display_weights();
      cout << endl;
      trainingDS.display_observations();

      trainingDS.pred();
      cout << endl << "*** Training Finished ***" << endl;
      trainingDS.display_results();
      trainingDS.display_confusion();
      
      cout << endl;
      cout << "  *********** TESTING ROC CURVE DATA ************" << endl;
      cout << "  ***********************************************" << endl << endl;
      cout << "  Threshold   FPR    TPR    Distance From Optimal" << endl;
      cout << "  =========   =====  =====  =====================" << endl;
      cout << "      0.000   1.00   1.00   1.00" << endl;

      testingDS.set_weights(trainingDS);
      testingDS.pred();

      thr = 0.0;
      optDIST = 1.0;
      for (int i=0; i<50; i++) {
	thr += 0.01998;
	testingDS.apply_threshold(thr);
	testingDS.calc_conf(res);
	tpr = (res[TP]/(res[TP]+res[FN]));
	fpr = (res[FP]/(res[FP]+res[TN]));
	dist = (tpr - 1.0)*(tpr - 1.0) + fpr*fpr;
	if (dist < optDIST) 
	  { optDIST = dist; optTPR = tpr; optFPR = fpr; optTHR = thr; }

	cout << setprecision(3)
	     << "  " << setw(9) << thr
	     << setprecision(2)
	     << "  " << setw(5) << fpr
	     << "  " << setw(5) << tpr
	     << "  " << setw(5) << dist << endl; 
      } // end for (i)

      cout << "      1.000   0.00   0.00   1.00" << endl;
      cout << endl;

      testingDS.outdata("setdata.js");
      
      cout << "Optimal threshold: " << setprecision(3) << optTHR << " (TPR = " 
	   << optTPR << ", FPR = " << setprecision(2) << optFPR << ")" << endl; 

      testingDS.apply_threshold(optTHR);
      cout << endl;
      testingDS.display_observations();
      cout << endl;
      testingDS.display_results();
      cout << endl;
      testingDS.display_confusion();
    } // end if (entireDS)
  } // end if (argv)
  else {
	cout << "Usage:  ./logr [Initial Weights] [y-data] [x-data]" << endl;
  }

  return 0;
}

Datamodule::Datamodule() {
  xvec = nullptr;
} // end Datamodule()

Datamodule::~Datamodule () {
} // end ~Datamodule()

/*
** The getsoln() function iterates to a solution until either the objective
** function change from one iteration to the next is less than espsilon, or the
** max number of iterations is reached.
*/
int Datamodule::getsoln(double epsilon, int maxiter) {
  int    i;             // counter
  int    szw;           // the size of the w vector
  double ll, ll_old;    // objective function values
  double wTx, f;
  double alpha = 0.001; // speed at which to converge using the gradient
  Svect  dk;            // temp variable for the gradient

  ll = ll_old = 0.0;
  szw = wvec.size();
  for (i=0; i<maxiter; i++) {
    ll_old = ll;
    // calculating the gradient of the logistic function
    dk.resize(szw);
    for (int i = 0; i < examples(); i++) {
      wTx  = (wvec * xvec[i]).sum();
      f    = exp(wTx)/(1+exp(wTx));
      dk  += (xvec[i] * (yvec[i] - f));         // gradient update
      ll  += yvec[i] * wTx - log(1 + exp(wTx)); // log-liklihood
    }

    if (fabs(ll_old-ll) < epsilon) break;
    wvec += dk*alpha;
  }

  return i;

}

/*
** The gen_conf_matrix() function calculates and displays (to stdout) a
** confusion matrix using the observed y values and the calculated y-values
** as inputs.
*/
void Datamodule::calc_conf(double *results) {
  cvec.resize(4);
  if (rvec.size() == 0) pred();
  if (tvec.size() == 0) apply_threshold();
  if (tvec.size() == yvec.size()) {
    for (int i=0; i<tvec.size(); i++) {
      if ((yvec[i] == 1) && (tvec[i] == 1)) cvec[TP]++; // true positives
      if ((yvec[i] == 0) && (tvec[i] == 1)) cvec[FP]++; // false positives
      if ((yvec[i] == 1) && (tvec[i] == 0)) cvec[FN]++; // false negatives
      if ((yvec[i] == 0) && (tvec[i] == 0)) cvec[TN]++; // true negatives
    } // end for (i)
    if (results != nullptr) {
      results[TP] = cvec[TP];
      results[FP] = cvec[FP];
      results[FN] = cvec[FN];
      results[TN] = cvec[TN];
    } // end if (results)
  } // end if (tvec.size())
} // end calc_conf()

/*
** The xmat_split() function takes an input matrix and randomly splits 
** it into two subsets of the data.  The fraction that goes to the first
** output matrix is given by the second argument.
*/
int xmat_split(Datamodule &inputDS, double fract1, Datamodule &out1DS,
	       Datamodule &out2DS) {
  int    n1, n2;  // number of examples for matrix #1 & #2
  int    nsize;   // size of input
  int    i, j, k; // counters
  double d;       // discriminator

  // random number generator
  default_random_engine generator; 
  uniform_real_distribution<double> distrib(0.0,1.0);

  if ((fract1 >= 0.0) && (fract1 <= 1.0)) {
	nsize  = inputDS.yvec.size();
	n1     = (int)(fract1 * nsize);
	n2     = (int)(nsize - n1);

	out1DS.xvec = new Svect[n1];
	out2DS.xvec = new Svect[n2];
	out1DS.yvec.resize(n1);
	out2DS.yvec.resize(n2);
	out1DS.wvec = inputDS.wvec;
	out2DS.wvec = inputDS.wvec;

	j = k = 0;
	for (int i=0; i < nsize; i++) {
	  d = distrib(generator);
	  if ((d < fract1) && (j < n1)) 
		{ out1DS.xvec[j] = inputDS.xvec[i]; out1DS.yvec[j] = inputDS.yvec[i]; j++; }
	  else 
		{ out2DS.xvec[k] = inputDS.xvec[i]; out2DS.yvec[k] = inputDS.yvec[i]; k++; }
	} // end for loop (i)

  } // end if (fract1) 
 
  return n1;
} // end xmat_split()

/*
** The read_input() function reads the input files and stores the data in
** the Svect class for use in the solution convergence algorithm.
*/
bool Datamodule::read_input(char **argc, bool verbose) {
  ifstream infile;
  int      nFeatures, nExamples;

  // reading in initial weights file
  infile.open(argc[1]);
  if (infile.is_open()) {
	if (verbose) cout << "Reading in data files..." << endl;
	
	infile >> nFeatures >> nExamples;
	cout << "  (" << nFeatures << " features, " 
		 << nExamples << " examples)" << endl;
	xvec = new Svect[nExamples];
	for (int i=0; i<nExamples; i++) xvec[i].resize(nFeatures);
	wvec.resize(nFeatures);
	yvec.resize(nExamples);

	infile >> wvec;
	infile.close();
	if (verbose) cout << "Initial Weights = " << wvec << endl;
	  
	infile.open(argc[2]);
	if (infile.is_open()) {
	  infile >> yvec;
	  infile.close();
	  if (verbose) cout << "Observed y-values: " << endl << yvec << endl;

	  infile.open(argc[3]);
	  if (infile.is_open()) {
		for (int i=0; i<nExamples; i++) infile >> xvec[i];
		infile.close();
		if (verbose) cout << "Features:" << endl;
		if (verbose) 
		  for (int i=0; i<nExamples; i++) 
			cout << setw(5) << i << ": " << xvec[i] << endl;
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


void Datamodule::outdata(string fname) {
  ofstream outfile;    // output file
  double   thr = 0.0;  // threshold
  double   tpr, fpr;   // temp variables for true/false postive rates
  double   dist;       // distance from optimal

  outfile.open(fname);
  if (outfile.is_open()) {
    outfile << "function setdata() {" << endl;
    outfile << "var inputvar = " << endl;
    outfile << "[ [ 1.000, 1.000, 1.000 ]," << endl;
    for (int i=0; i<50; i++) {
      thr += 0.01998;
      apply_threshold(thr);
      calc_conf();
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
** Set the weights from another Datamodule instance to this one.
*/
void Datamodule::set_weights(Datamodule &dm) { wvec = dm.wvec; }

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
void Datamodule::pred(void) {
  double wTx;

  tvec.resize(0);
  rvec.resize(yvec.size());

  for (int i=0; i<yvec.size(); i++) {
    wTx  = (wvec * xvec[i]).sum();
    rvec[i] = exp(wTx)/(1 + exp(wTx));
  }
}

/*
** The apply_threshold() function applies a threshold to the results vector
** and stores it in the threshold limited vector.
*/
void Datamodule::apply_threshold(double thr) {
  tvec = rvec;
  tvec.apply_threshold(thr);
}


/*
** The examples() function returns the number of examples in the dataset.
*/
int Datamodule::examples(void) { return yvec.size(); }

/*
** The display_weights() function sends the current weights vector to stdout.
*/
void Datamodule::display_weights(int dec) { 
  cout << setprecision(dec) << fixed;
  cout << "Weights: " << wvec << endl; 
}

/*
** The display_observations() function sends the current observations vector to stdout.
*/
void Datamodule::display_observations(void) { 
  cout << setprecision(0) << fixed;
  cout << "Observations: " << yvec << endl; 
}

/*
** The display_features() function sends the current features vector to stdout.
*/
void Datamodule::display_features(int dec) {
  cout << setprecision(dec) << fixed;
  cout << endl << "Features:" << endl;
  for (int i=0; i<yvec.size(); i++)
    cout << xvec[i] << endl;
  cout << endl;
} // end display_features()

/*
** The display_results() function sends the results vector to stdout.  If there
** are no results, it prints "no results".
*/
void Datamodule::display_results(void) { 
  if (tvec.size() == 0) apply_threshold();
  if (tvec.size() == yvec.size()) {
    cout << setprecision(0) << fixed;
    cout << "Results (liklihood): " << tvec << endl; 
  }
  else {
    cout << "** No Results **" << endl;
  }
}

/*
** The display_confusion() function calculates then displays the confusion matrix.
*/
void Datamodule::display_confusion(void) { 
  int tp, fp, tn, fn;
  calc_conf();

  if (tvec.size() == yvec.size()) {
    tp = cvec[TP];
    fp = cvec[FP];
    tn = cvec[TN];
    fn = cvec[FN];

    cout << setprecision(1) << fixed << endl;
    cout << "                  Training Confusion Matrix" << endl;
    cout << "        +-----------------------------------------------+" << endl;
    cout << "        |             |             actual              |" << endl;
    cout << "        +-------------+---------------------------------+" << endl;
    cout << "        |  predicted  |      TRUE      |     FALSE      |" << endl;
    cout << "        +-------------+----------------+----------------+" << endl;
    cout << "        |    TRUE     | " << setw(5) << tp << " (" << setw(5) << (100.0*tp)/(tp+fp) << "%) | " 
 	                       << setw(5) << fp << " (" << setw(5) << (100.0*fp)/(fp+tp) << "%) |" << endl;
    cout << "        +-------------+----------------+----------------+" << endl;
    cout << "        |   FALSE     | " << setw(5) << fn << " (" << setw(5) << (100.0*fn)/(tn+fn) << "%) | " 
 	                       << setw(5) << tn << " (" << setw(5) << (100.0*tn)/(fn+tn) << "%) |" << endl;
    cout << "        +-------------+----------------+----------------+" << endl;
    cout << "        |   Total     | " << setw(5) << (tp+fn) << "          | " 
	                       << setw(5) << (fp+tn) << "          |" << endl;
    cout << "        +-------------+----------------+----------------+" << endl;
    cout << " NOTE:  The numbers in parentheses represent the probability" << endl;
    cout << "        that a given prediction will be accurate" << endl;

  }
  else {
    cout << "** No Results **" << endl;
  }
}

