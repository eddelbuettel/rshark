## RShark -- R interface to the Shark libraries
##
## Copyright (C) 2010  Shane Conway	<shane.conway@gmail.com>
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

svm <- function(...){

     val <- .Call("SVM",
                     list(
   		          yield=as.double(yield),
	                  ),
                 PACKAGE="RShark")
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
