#include <iostream>
#include <fstream>
#include <string>

using namespace std;

#define FEATURES 9

int main(int argv, char **argc) {
  ifstream infile;
  ofstream outfile1, outfile2, outfile3;
  string   strFile1, strFile2;
  int      id,n,ndata;
  char     c;

  cout << "Executing procdata with command line arguments: ";
  for (int i=0; i<argv; i++)
	cout << argc[i] << " ";
  cout << endl;

  if (argv == 3) {
	infile.open(argc[1]);
	if (infile.is_open()) {
	  ndata = 0;
	  strFile1 = strFile2 = argc[2];
	  strFile1 = "y_" + strFile1 + "_observed.dat";
	  strFile2 = "x_" + strFile2 + "_features.dat";
	  outfile1.open(strFile1);
	  if (outfile1.is_open()) {
		outfile2.open(strFile2);
		if (outfile2.is_open()) {
		  outfile3.open("weights_input.dat");
		  if (outfile3.is_open()) {
			while (!infile.eof()) {
			  infile >> id;
			  for (int i=0; i<FEATURES; i++) {
				infile >> c;
				c = infile.peek();
				if      (c == EOF)   break;
				else if (c == '?') { infile >> c; n = 0; }
				else                 infile >> n;
				outfile2 << n << " ";
			  }
			  if (c == EOF) break;
			  outfile2 << endl;
			  infile >> c >> n;
			  if (n == 2) n = 0;
			  if (n == 4) n = 1;
			  outfile1 << n << " ";
			  ndata++;
			}

			outfile3 << FEATURES << " " << ndata << endl;
			for (int i=0; i<ndata; i++) outfile3 << "0.5" << " ";
			outfile3 << endl;

			outfile1.close();
			outfile2.close();
			outfile3.close();
		  }
		  else cerr << "Could not open output file #3." << endl;
		}
		else cerr << "Could not open output file #2." << endl;
	  }
	  else cerr << "Could not open output file #1." << endl;
	}
	else cerr << "Bad input file name." << endl;
  }
  else {
	cerr << "Usage:  ./procdata [Input Data] [Processed Data base file name]" << endl;
  }

  return 0;
}
