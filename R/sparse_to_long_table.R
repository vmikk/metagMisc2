sparse_to_long_table <- function(x, samples_are_rows = FALSE) {

  mat <- as_rarefy_matrix(x, samples_are_rows = samples_are_rows)
  if(!inherits(mat, "dgCMatrix")){
    mat <- methods::as(mat, "dgCMatrix")
  }

  res <- .sparse_to_long_table_cpp(mat)

  if(requireNamespace("data.table", quietly = TRUE)){
    data.table::setDT(res)
  }

  return(res)
}
