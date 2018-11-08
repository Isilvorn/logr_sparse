/*
** Created by: Jason Orender
** (c) 2018 all rights reserved
**
** This program implements a logistic regression algorithm using a sparse vector format. Since
** it uses the list container to ensure that vectors of virtually any size can be used and resized
** at will, it incurs a fairly massive penalty (about 7x) over the static version for smaller
** vectors.  However, for extremely large vectors, there is both a speed and memory advantage if 
** most of the entries in the vector are zeroes.  The cache system also reduces some of the speed 
** disadvantage for smaller vectors.
*/

#include "../include/datamodule.h"

using namespace std;

int main(int argv, char **argc) {

  cout << endl << "Executing logr with command line arguments: " << endl << "  ";
  for (int i=0; i<argv; i++)
	cout << argc[i] << " ";
  cout << endl;

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

