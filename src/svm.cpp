// -*- mode: c++; tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 4 -*-
//
// RShark -- R interface to the Shark libraries
//
// Copyright (C) 2010 Shane Conway and Dirk Eddelbuettel
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

// Exposing the SVM classes in Shark: http://shark-project.sourceforge.net/ReClaM/index.html#supportvectormachines

//#include <rshark.h>

#include <Rcpp.h>
#include <Rng/GlobalRng.h>
#include <ReClaM/Svm.h>
#include <ReClaM/MeanSquaredError.h>

using namespace std;

double sinc(double x)
{
    if (x == 0.0) return 1.0;
    else return sin(x) / x;
}

SVM R_Epsilon_SVM(SVM svm, double C, double epsilon, Array<double> x, Array<double> y) {

	Epsilon_SVM t_svm(&svm, C, epsilon);
	SVM_Optimizer SVMopt;
	SVMopt.init(t_svm);

	// train the SVM
	SVMopt.optimize(svm, x, y);

	return svm;
}

/*
SVM R_GaussianProcess(SVM svm, Array<double> x, Array<double> y) {

	GaussianProcess t_svm(&svm, x, y);
	SVM_Optimizer SVMopt;
	SVMopt.init(t_svm);

	// train the SVM
	SVMopt.optimize(svm, x, y);

	return svm;
}
*/

SVM R_RegularizationNetwork(SVM svm, double gamma, Array<double> x, Array<double> y) {

	RegularizationNetwork t_svm(&svm, gamma);
	SVM_Optimizer SVMopt;
	SVMopt.init(t_svm);

	// train the SVM
	SVMopt.optimize(svm, x, y);

	return svm;
}


RcppExport SEXP SVMregression(SEXP Xs, SEXP Ys, SEXP svmParameters) {

    try {
        Rcpp::NumericMatrix xR = Rcpp::NumericMatrix(Xs);
        Rcpp::NumericVector yR = Rcpp::NumericVector(Ys);
        Rcpp::List rparam(svmParameters);
        double C = Rcpp::as<double>(rparam["C"]);
        double epsilon = Rcpp::as<double>(rparam["epsilon"]);
        double gamma = Rcpp::as<double>(rparam["gamma"]);
        double sigma = Rcpp::as<double>(rparam["sigma"]);
        string type = Rcpp::as<string>(rparam["type"]);
        string kernel = Rcpp::as<string>(rparam["kernel"]);

		unsigned int examples = xR.rows();
        unsigned int e;

        // create the sinc problem
        Array<double> x(examples, 1);
        Array<double> y(examples, 1);
        for (e = 0; e < examples; e++)
        {
            x(e, 0) = xR[e];
            y(e, 0) = yR[e];
        }

        // create the kernel function
        double kgamma = 0.5 / (sigma * sigma);
        RBFKernel k(kgamma);

        // create the SVM for prediction
        SVM svm(&k, false);

        // create a training scheme and an optimizer for learning
        if(type=="Epsilon_SVM") {
            svm = R_Epsilon_SVM(svm, C, epsilon, x, y);
        } else if(type=="RegularizationNetwork") {
        	svm = R_RegularizationNetwork(svm, gamma, x, y);
        } else if(type=="GaussianProcess") {
        	//svm = R_GaussianProcess(svm, x, y);
        } else {
        	return R_NilValue;
        }

        // compute the mean squared error on the training data:
        MeanSquaredError mse;
        double err = mse.error(svm, x, y);

        unsigned int dimension = svm.getDimension();
        unsigned int offset = svm.getOffset();
        unsigned int nSV = svm.getExamples();

        // Find the support vector
        Rcpp::NumericVector alpha(nSV);
        for (size_t i=0; i<nSV; i++) alpha[i] = svm.getAlpha(i);

        Rcpp::List rl = R_NilValue;
        rl = Rcpp::List::create(Rcpp::Named("error") = err,
        		Rcpp::Named("offset") = offset,
        		Rcpp::Named("nSV") = nSV,
        		Rcpp::Named("alpha") = alpha,
        		Rcpp::Named("dimension") = dimension);
        return rl;

    } catch(std::exception &ex) {
        forward_exception_to_r(ex);
    } catch(...) {
        ::Rf_error("c++ exception (unknown reason)");
    }

    return R_NilValue;
}

