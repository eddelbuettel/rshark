// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
//
// RShark -- R interface to the Shark libraries
//
// Copyright (C) 2010 Shane Conway
//
// This file is part of the RShark library for GNU R.
// It is made available under the terms of the GNU General Public
// License, version 2, or at your option, any later version,
// incorporated herein by reference.
//
// This program is distributed in the hope that it will be
// useful, but WITHOUT ANY WARRANTY; without even the implied
// warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
// PURPOSE.  See the GNU General Public License for more
// details.
//
// You should have received a copy of the GNU General Public
// License along with this program; if not, write to the Free
// Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
// MA 02111-1307, USA

//#include <rshark.hpp>

#include <Rng/GlobalRng.h>
#include <ReClaM/Svm.h>
#include <ReClaM/MeanSquaredError.h>
#include <iostream>

using namespace std;

RcppExport SEXP SVM(SEXP svmParameters) {

    try {
        Rcpp::List rparam(optionParameters);
        int examples = Rcpp::as<int>(rparam["examples"]);

		cout << "*** Support Vector Machine example program ***" << endl << endl;
		cout << "The regression training data are sampled from a sinc function" << endl;
		cout << "with additive Gaussian white noise." << endl;
		cout << endl;

		unsigned int e;
		Rng::seed(42);

		double C = 100.0;
		double epsilon = 0.1;
		double sigma = 2.0;
		unsigned int examples = 100;

		// create the sinc problem
		Array<double> x(examples, 1);
		Array<double> t(examples, 1);
		Array<double> y(examples, 1);
		for (e = 0; e < examples; e++)
		{
			x(e, 0) = Rng::uni(-12.0, 12.0);				// point
			t(e, 0) = sinc(x(e, 0));						// target
			y(e, 0) = t(e, 0) + Rng::gauss(0.0, 0.01);		// label
		}

		// create the SVM for prediction
		double gamma = 0.5 / (sigma * sigma);
		RBFKernel k(gamma);
		SVM svm(&k, false);

		// create a training scheme and an optimizer for learning
		Epsilon_SVM esvm(&svm, C, epsilon);
		SVM_Optimizer SVMopt;
		SVMopt.init(esvm);

		// train the SVM
		cout << "Support Vector Machine training ..." << flush;
		SVMopt.optimize(svm, x, y);
		cout << " done." << endl << endl;

		// compute the mean squared error on the training data:
		MeanSquaredError mse;
		double err = mse.error(svm, x, t);
		cout << "mean squared error on the training data: " << err << endl << endl;

        return Rcpp::wrap(err);

    } catch(std::exception &ex) {
        forward_exception_to_r(ex);
    } catch(...) {
        ::Rf_error("c++ exception (unknown reason)");
    }

    return R_NilValue;
}


