/*
** Created by: Jason Orender
** (c) 2018 all rights reserved
**
** This library implements a sparse vector format using the list standard container.  The list
** container was chosen so that new elements could be added or subtracted with minimal cost,
** since this will happen quite often as elements become zeroes and need to be eliminated from
** explicit expression in the vector.
*/

#ifndef VECT_H
#define VECT_H

#include <iostream>
#include <cmath>
#include <list>
#include <cstring>

using namespace std;

// cache size for Svect
#define CSZ 64

/*
** Datapoint is a struct that contains the data for Svect.
*/
struct Datapoint {
public:
  Datapoint()         { d = 0.0; }  // default constructor initializes as zero
  Datapoint(double f) { d = f; }    // alternate constructor initializes with supplied value
  ~Datapoint()        { }           // destructor does nothing

  Datapoint& operator=(const Datapoint& in) { d = in.d; i = in.i; return *this; } 

  double d;                         // the double precision element of the data
  int    i;                         // the index of the data (like an array index)
  
};

/*
** Svect is a class set up to store and manipulate a Datapoint vector.
*/
class Svect {
public:
  Svect();                           // default constructor
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

  // defining a cache for array-like access speeds for vectors that have fewer explicit elements
  // than the size of the cache
  list<Datapoint>::iterator cache_it[CSZ];     // storing the iterator to directly access list elements
  int                       cache_index[CSZ];  // storing the index of the element in the vector
  double*                   cache_dp[CSZ];     // storing a pointer to the data element
};

ostream& operator<<(ostream&, const Svect&);
istream& operator>>(istream&, Svect&);

#endif
