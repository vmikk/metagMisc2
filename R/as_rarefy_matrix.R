#' Coerce inputs to dgCMatrix (taxa as rows, samples as columns)
#'
#' @param x matrix-like object, phyloseq, TreeSummarizedExperiment, SparseArray, data.frame, or data.table
#' @param samples_are_rows If TRUE, rows are samples and will be transposed
#' @return A \code{dgCMatrix} with non-negative numeric counts
#' @export
#' 
as_rarefy_matrix <- function(x, samples_are_rows = FALSE) {

  if (inherits(x, "phyloseq")) {
    if (!requireNamespace("phyloseq", quietly = TRUE)) {
      stop("Package 'phyloseq' is required for phyloseq inputs", call. = FALSE)
    }
    ot <- phyloseq::otu_table(x)
    mat <- methods::as(ot, "matrix")
    if (phyloseq::taxa_are_rows(ot)) {
      mat <- Matrix::Matrix(mat, sparse = TRUE)
    } else {
      mat <- Matrix::t(Matrix::Matrix(mat, sparse = TRUE))
    }
    return(methods::as(mat, "dgCMatrix"))
  }

  if (inherits(x, "TreeSummarizedExperiment") || inherits(x, "SummarizedExperiment")) {
    if (!requireNamespace("SummarizedExperiment", quietly = TRUE)) {
      stop("Package 'SummarizedExperiment' is required for SE inputs", call. = FALSE)
    }
    mat <- SummarizedExperiment::assay(x, 1L)
    x <- mat
  }

  if (inherits(x, "SVT_SparseMatrix") || inherits(x, "SparseArray")) {
    if (!requireNamespace("SparseArray", quietly = TRUE)) {
      stop("Package 'SparseArray' is required for SparseArray inputs", call. = FALSE)
    }
    x <- methods::as(x, "dgCMatrix")
    if (isTRUE(samples_are_rows)) {
      x <- Matrix::t(x)
    }
    return(x)
  }

  if (inherits(x, "data.frame") || inherits(x, "data.table")) {
    if (requireNamespace("data.table", quietly = TRUE) && inherits(x, "data.table")) {
      x <- as.matrix(x)
    } else {
      x <- as.matrix(x)
    }
  }

  if (is.matrix(x) || inherits(x, "Matrix")) {
    if (!inherits(x, "dgCMatrix")) {
      x <- Matrix::Matrix(x, sparse = TRUE)
      x <- methods::as(x, "dgCMatrix")
    }
    if (isTRUE(samples_are_rows)) {
      x <- Matrix::t(x)
    }
    return(x)
  }

  stop("Unsupported input type: ", paste(class(x), collapse = ", "), call. = FALSE)
}
