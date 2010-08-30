## rshark -- R interface to the Shark libraries
##
## Copyright (C) 2010  Shane Conway and Dirk Eddelbuettel
##
## This file is part of the RShark library for GNU R.
## It is made available under the terms of the GNU General Public
## License, version 2, or at your option, any later version,
## incorporated herein by reference.
##
## This program is distributed in the hope that it will be
## useful, but WITHOUT ANY WARRANTY; without even the implied
## warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
## PURPOSE.  See the GNU General Public License for more
## details.
##
## You should have received a copy of the GNU General Public
## License along with this program; if not, write to the Free
## Software Foundation, Inc., 59 Temple Place - Suite 330, Boston,
## MA 02111-1307, USA

ssvm <- function(x, y = NULL, scaled = TRUE, type = "C_SVM", kernel ="rbfdot", 
		kpar = "automatic", C = 1, nu = 0.2, epsilon = 0.1, gamma=1, 
		prob.model = FALSE, class.weights = NULL, cross = 0, fit = TRUE,
		cache = 40, tol = 0.001, shrinking = TRUE, sigma=1, ..., 
		subset, na.action = na.omit) {
	
	val <- .Call("SVMregression",
			list(	x=x,
					y=y,
					C=C,
					gamma=gamma,
					epsilon=epsilon,
					sigma=sigma,
					type=type,
					kernel=kernel
			),
			PACKAGE="rshark")
	class(val) <- c("shark.svm")
	val
}

# Generic methods
plot.svm <- function(x, ...) {
	warning("No plotting available for class", class(x)[1],"\n")
	invisible(x)
}

print.svm <- function(x, digits=5, ...) {
	cat("Error term", x, "\n")
	invisible(x)
}

summary.svm <- function(object, digits=5, ...) {
	cat("Detailed summary of SVM model for", class(object)[1], "\n")
	invisible(object)
}
